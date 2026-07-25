import os from "node:os";
import process from "node:process";

const chunks = [];
for await (const chunk of process.stdin) chunks.push(chunk);

try {
  const request = JSON.parse(Buffer.concat(chunks).toString("utf8"));
  if (request.version !== 1) throw new Error(`unsupported version ${request.version}`);
  if (!["normalize", "canonical", "echo"].includes(request.operation)) {
    throw new Error(`unsupported operation ${request.operation}`);
  }
  process.stdout.write(JSON.stringify(normalize(request.value)));
} catch (error) {
  process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`);
  process.exitCode = 1;
}

function normalize(value) {
  const state = {
    ids: new Map(),
    nextId: 1,
    workspace: portable(process.cwd()),
    home: portable(os.homedir()),
    temp: portable(os.tmpdir()),
  };
  return visit(value, undefined, state);
}

function visit(value, key, state) {
  if (Array.isArray(value)) return value.map((item) => visit(item, key, state));
  if (value && typeof value === "object") return object(value, key, state);
  if (typeof value === "string") return text(value, key, state);
  if (typeof value === "number" && Object.is(value, -0)) return 0;
  return value;
}

function object(value, parent, state) {
  const headers = parent?.toLowerCase() === "headers";
  const output = {};
  for (const sourceKey of Object.keys(value).sort()) {
    if (volatile(sourceKey)) continue;
    const key = headers ? sourceKey.toLowerCase() : sourceKey;
    if (headers && sensitive(key)) {
      output[key] = "<redacted>";
    } else if (timestamp(key)) {
      output[key] = "<timestamp>";
    } else if (identifier(key)) {
      const raw = value[sourceKey];
      if (raw === null) {
        output[key] = null;
      } else {
        const text = typeof raw === "string" ? raw : JSON.stringify(raw);
        if (!state.ids.has(text)) state.ids.set(text, `id-${state.nextId++}`);
        output[key] = state.ids.get(text);
      }
    } else {
      output[key] = visit(value[sourceKey], key, state);
    }
  }
  return output;
}

function text(value, key, state) {
  let result = value.replace(/\r\n?/gu, "\n").normalize("NFC");
  if (!pathKey(key)) return result;
  result = portable(result);
  if (/^[A-Z]:/u.test(result)) result = result[0].toLowerCase() + result.slice(1);
  result = replaceRoot(result, state.workspace, "<workspace>");
  result = replaceRoot(result, state.home, "<home>");
  result = replaceRoot(result, state.temp, "<temp>");
  return result;
}

function replaceRoot(value, root, replacement) {
  if (!root || !value.toLowerCase().startsWith(root.toLowerCase())) return value;
  return replacement + value.slice(root.length);
}

function portable(value) {
  return String(value).replaceAll("\\", "/");
}

function volatile(key) {
  return ["duration", "durationms", "elapsed", "requestid", "traceid"].includes(
    key.toLowerCase(),
  );
}

function timestamp(key) {
  return ["timestamp", "createdat", "updatedat", "startedat", "endedat"].includes(
    key.toLowerCase(),
  );
}

function identifier(key) {
  return ["id", "sessionid", "parentid", "toolcallid", "messageid"].includes(
    key.toLowerCase(),
  );
}

function pathKey(key) {
  if (!key) return false;
  const lower = key.toLowerCase();
  return (
    lower === "path" ||
    lower === "cwd" ||
    lower.endsWith("path") ||
    lower.endsWith("dir") ||
    lower.endsWith("directory")
  );
}

function sensitive(key) {
  return [
    "authorization",
    "api-key",
    "x-api-key",
    "proxy-authorization",
    "cookie",
    "set-cookie",
  ].includes(key);
}
