//! Conformance manifest validation, canonical JSON, and reference-test inventory.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::ffi::OsStr;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Number, Value};
use unicode_normalization::UnicodeNormalization;
use walkdir::WalkDir;

/// Fixed reference commit used by this workspace.
pub const REFERENCE_COMMIT: &str = "518855dd502220d0c6480fb8863e2e7f8799893f";

/// Errors produced by the conformance runner.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A filesystem operation failed.
    #[error("filesystem operation failed for {path}: {source}")]
    Io {
        /// Relevant path.
        path: PathBuf,
        /// Original error.
        source: std::io::Error,
    },
    /// A YAML manifest could not be decoded.
    #[error("invalid manifest YAML: {0}")]
    Manifest(#[from] serde_yaml::Error),
    /// JSON input or output was malformed.
    #[error("invalid JSON: {0}")]
    Json(#[from] serde_json::Error),
    /// The manifest violates its schema contract.
    #[error("manifest validation failed:\n{0}")]
    Validation(String),
    /// A child runner failed.
    #[error("runner `{runner}` failed with status {status}: {stderr}")]
    Runner {
        /// Runner display name.
        runner: String,
        /// Exit status as text.
        status: String,
        /// Captured standard error.
        stderr: String,
    },
    /// A child runner emitted no JSON.
    #[error("runner `{0}` emitted an empty response")]
    Empty(String),
    /// Rust and reference output differed.
    #[error("fixture `{fixture}` differs\nrust: {rust}\nreference: {reference}")]
    Mismatch {
        /// Fixture path.
        fixture: PathBuf,
        /// Canonical Rust result.
        rust: String,
        /// Canonical reference result.
        reference: String,
    },
    /// A requested operation is not part of the protocol.
    #[error("unsupported conformance operation `{0}`")]
    Operation(String),
}

/// Result alias for this crate.
pub type Result<T> = std::result::Result<T, Error>;

/// Root conformance manifest.
#[derive(Clone, Debug, Deserialize)]
pub struct Manifest {
    /// Schema identifier.
    pub schema: String,
    /// Fixed reference metadata.
    pub baseline: Baseline,
    /// Reference-test inventory policy.
    pub test_inventory: TestInventory,
    /// Canonical JSON policy.
    pub canonical_json: CanonicalJson,
    /// Gate declarations.
    pub gates: BTreeMap<String, Gate>,
    /// Status declarations.
    pub statuses: Statuses,
    /// Runner declarations.
    pub runners: Runners,
    /// Observable compatibility rows.
    pub features: Vec<Feature>,
    /// Explicitly excluded items.
    pub out_of_scope: Vec<Excluded>,
}

/// Reference metadata.
#[derive(Clone, Debug, Deserialize)]
pub struct Baseline {
    /// Repository path.
    pub repository: PathBuf,
    /// Pinned git commit.
    pub commit: String,
    /// Upstream package version.
    pub package_version: String,
    /// Expected reference test-file count.
    pub expected_test_files: usize,
    /// Ordered source authority.
    pub authority: Vec<String>,
}

/// Test inventory configuration.
#[derive(Clone, Debug, Deserialize)]
pub struct TestInventory {
    /// Roots to scan.
    pub roots: Vec<PathBuf>,
    /// Test globs.
    pub globs: Vec<String>,
    /// Generated inventory path.
    pub output: PathBuf,
    /// Whether each test needs a feature owner.
    pub require_every_test_mapped: bool,
    /// Broad ownership rules.
    pub ownership: Vec<Ownership>,
}

/// Test ownership rule.
#[derive(Clone, Debug, Deserialize)]
pub struct Ownership {
    /// Relative glob.
    pub glob: String,
    /// Required feature prefix.
    pub prefix: String,
}

/// Canonical JSON policy.
#[derive(Clone, Debug, Deserialize)]
pub struct CanonicalJson {
    /// Text encoding.
    pub encoding: String,
    /// Record line ending.
    pub line_ending: String,
    /// Whether output ends with a newline.
    pub trailing_newline: bool,
    /// Unicode normalization form.
    pub unicode: String,
    /// Object-key order.
    pub object_keys: String,
    /// Array policy.
    pub arrays: String,
    /// Number policy.
    pub numbers: NumberPolicy,
    /// Timestamp policy.
    pub timestamps: Replacement,
    /// Identifier policy.
    pub identifiers: IdentifierPolicy,
    /// Path policy.
    pub paths: PathPolicy,
    /// Header policy.
    pub headers: HeaderPolicy,
    /// Fields removed from snapshots.
    pub volatile_fields: Vec<String>,
    /// Usage normalization policy.
    pub usage: UsagePolicy,
}

/// Number normalization.
#[derive(Clone, Debug, Deserialize)]
pub struct NumberPolicy {
    /// Finite representation.
    pub finite: String,
    /// Replacement for negative zero.
    pub negative_zero: i64,
    /// Behavior for non-finite values.
    pub non_finite: String,
}

/// Generic replacement policy.
#[derive(Clone, Debug, Deserialize)]
pub struct Replacement {
    /// Action name.
    pub action: String,
    /// Replacement value.
    pub value: String,
}

/// Stable identifier policy.
#[derive(Clone, Debug, Deserialize)]
pub struct IdentifierPolicy {
    /// Mapping policy.
    pub action: String,
    /// Stable value prefix.
    pub prefix: String,
}

/// Portable path policy.
#[derive(Clone, Debug, Deserialize)]
pub struct PathPolicy {
    /// Separator policy.
    pub separators: String,
    /// Drive-letter policy.
    pub drive_letter: String,
    /// Workspace replacement.
    pub workspace: String,
    /// Home replacement.
    pub home: String,
    /// Temporary-directory replacement.
    pub temp: String,
}

/// HTTP header policy.
#[derive(Clone, Debug, Deserialize)]
pub struct HeaderPolicy {
    /// Header name casing.
    pub names: String,
    /// Sensitive names.
    pub redact: Vec<String>,
}

/// Usage/cost policy.
#[derive(Clone, Debug, Deserialize)]
pub struct UsagePolicy {
    /// Optional field behavior.
    pub absent_optional_fields: String,
    /// Cost decimal precision.
    pub cost_precision: u8,
}

/// One execution gate.
#[derive(Clone, Debug, Deserialize)]
pub struct Gate {
    /// Gate kind.
    pub kind: String,
    /// Network policy encoded as boolean or name.
    pub network: serde_yaml::Value,
}

/// Valid status configuration.
#[derive(Clone, Debug, Deserialize)]
pub struct Statuses {
    /// Allowed states.
    pub allowed: Vec<String>,
    /// Required row fields for passing.
    pub passing_requires: Vec<String>,
}

/// Runner configuration.
#[derive(Clone, Debug, Deserialize)]
pub struct Runners {
    /// Rust command.
    pub rust: Vec<String>,
    /// Reference command.
    pub reference: Vec<String>,
    /// Input fixture directory.
    pub fixture_dir: PathBuf,
    /// Snapshot directory.
    pub snapshot_dir: PathBuf,
}

/// One observable behavior.
#[derive(Clone, Debug, Deserialize)]
pub struct Feature {
    /// Stable identifier.
    pub id: String,
    /// Required gate.
    pub gate: String,
    /// Upstream evidence.
    pub reference: Vec<String>,
    /// Rust test target.
    pub rust_test: String,
    /// Current state.
    pub status: String,
}

/// Explicit non-goal.
#[derive(Clone, Debug, Deserialize)]
pub struct Excluded {
    /// Stable identifier.
    pub id: String,
    /// Exclusion reason.
    pub reason: String,
}

/// A generated reference test mapping.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct TestMap {
    /// Path relative to the reference repository.
    pub path: String,
    /// Owning compatibility feature.
    pub feature: String,
    /// Gate inherited from that feature.
    pub gate: String,
}

/// Versioned request understood by both runners.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Request {
    /// Protocol version.
    pub version: u32,
    /// Operation name.
    pub operation: String,
    /// Operation value.
    pub value: Value,
}

/// Load a manifest from disk.
///
/// # Errors
///
/// Returns an error when the file cannot be read or its YAML is invalid.
pub fn load(path: impl AsRef<Path>) -> Result<Manifest> {
    let path = path.as_ref();
    let source = fs::read_to_string(path).map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })?;
    serde_yaml::from_str(&source).map_err(Error::from)
}

/// Validate manifest structure and return reference-test mappings.
///
/// # Errors
///
/// Returns an error for invalid manifest relationships or when the reference
/// test inventory cannot be read.
pub fn validate(root: &Path, manifest: &Manifest) -> Result<Vec<TestMap>> {
    let mut errors = Vec::new();
    if manifest.schema != "ri.conformance/v1" {
        errors.push(format!("unsupported schema `{}`", manifest.schema));
    }
    if manifest.baseline.commit != REFERENCE_COMMIT {
        errors.push(format!(
            "reference commit is `{}`, expected `{REFERENCE_COMMIT}`",
            manifest.baseline.commit
        ));
    }
    if manifest.baseline.authority.is_empty() {
        errors.push("baseline authority cannot be empty".to_owned());
    }
    let allowed: BTreeSet<_> = manifest
        .statuses
        .allowed
        .iter()
        .map(String::as_str)
        .collect();
    let mut ids = BTreeSet::new();
    for feature in &manifest.features {
        if !ids.insert(feature.id.as_str()) {
            errors.push(format!("duplicate feature id `{}`", feature.id));
        }
        if !manifest.gates.contains_key(&feature.gate) {
            errors.push(format!(
                "feature `{}` references unknown gate `{}`",
                feature.id, feature.gate
            ));
        }
        if !allowed.contains(feature.status.as_str()) {
            errors.push(format!(
                "feature `{}` uses unknown status `{}`",
                feature.id, feature.status
            ));
        }
        if feature.reference.is_empty() {
            errors.push(format!(
                "feature `{}` has no reference evidence",
                feature.id
            ));
        }
        if feature.rust_test.trim().is_empty() {
            errors.push(format!("feature `{}` has no Rust test target", feature.id));
        }
    }
    for item in &manifest.out_of_scope {
        if item.reason.trim().is_empty() {
            errors.push(format!("out-of-scope item `{}` has no reason", item.id));
        }
    }
    let inventory = inventory(root, manifest)?;
    if inventory.len() != manifest.baseline.expected_test_files {
        errors.push(format!(
            "reference test inventory contains {} files, expected {}",
            inventory.len(),
            manifest.baseline.expected_test_files
        ));
    }
    if manifest.test_inventory.require_every_test_mapped {
        let known: BTreeSet<_> = manifest
            .features
            .iter()
            .map(|feature| feature.id.as_str())
            .collect();
        for item in &inventory {
            if !known.contains(item.feature.as_str()) {
                errors.push(format!(
                    "reference test `{}` maps to unknown feature `{}`",
                    item.path, item.feature
                ));
            }
        }
    }
    if errors.is_empty() {
        Ok(inventory)
    } else {
        Err(Error::Validation(errors.join("\n")))
    }
}

/// Build a deterministic map from every reference test to a compatibility row.
///
/// # Errors
///
/// Returns an error when a configured test root is missing or cannot be read.
pub fn inventory(root: &Path, manifest: &Manifest) -> Result<Vec<TestMap>> {
    let reference_root = root.join(&manifest.baseline.repository);
    let features: BTreeMap<_, _> = manifest
        .features
        .iter()
        .map(|feature| (feature.id.as_str(), feature.gate.as_str()))
        .collect();
    let mut mappings = Vec::new();
    for relative_root in &manifest.test_inventory.roots {
        let absolute_root = root.join(relative_root);
        if !absolute_root.exists() {
            return Err(Error::Validation(format!(
                "test root `{}` does not exist",
                relative_root.display()
            )));
        }
        for item in WalkDir::new(&absolute_root).follow_links(false) {
            let item = item.map_err(|source| Error::Io {
                path: absolute_root.clone(),
                source: std::io::Error::other(source.to_string()),
            })?;
            if !item.file_type().is_file() || !is_test_file(item.path()) {
                continue;
            }
            let relative = item
                .path()
                .strip_prefix(&reference_root)
                .unwrap_or(item.path());
            let path = portable(relative);
            let feature = classify(&path);
            let gate = features
                .get(feature)
                .copied()
                .unwrap_or("transform")
                .to_owned();
            mappings.push(TestMap {
                path,
                feature: feature.to_owned(),
                gate,
            });
        }
    }
    mappings.sort_by(|left, right| left.path.cmp(&right.path));
    mappings.dedup_by(|left, right| left.path == right.path);
    Ok(mappings)
}

/// Persist a generated inventory as stable pretty JSON.
///
/// # Errors
///
/// Returns an error when JSON encoding, directory creation, or writing fails.
pub fn write_inventory(root: &Path, manifest: &Manifest, mappings: &[TestMap]) -> Result<()> {
    let output = root.join(&manifest.test_inventory.output);
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent).map_err(|source| Error::Io {
            path: parent.to_path_buf(),
            source,
        })?;
    }
    let mut bytes = serde_json::to_vec_pretty(mappings)?;
    bytes.push(b'\n');
    fs::write(&output, bytes).map_err(|source| Error::Io {
        path: output,
        source,
    })
}

fn is_test_file(path: &Path) -> bool {
    let name = path.file_name().and_then(OsStr::to_str).unwrap_or_default();
    name.ends_with(".test.ts") || name.ends_with(".test.mjs")
}

fn classify(path: &str) -> &'static str {
    let lower = path.to_ascii_lowercase();
    if lower.starts_with("scripts/") {
        "cli.modes-shared-runtime"
    } else if lower.starts_with("packages/ai/") {
        if lower.contains("anthropic") {
            "ai.wire.anthropic"
        } else if lower.contains("codex") {
            "ai.wire.openai-codex"
        } else if lower.contains("responses") || lower.contains("openai") {
            "ai.wire.openai-responses"
        } else if lower.contains("vertex") {
            "ai.wire.vertex"
        } else if lower.contains("google") {
            "ai.wire.google"
        } else if lower.contains("mistral") {
            "ai.wire.mistral"
        } else if lower.contains("bedrock") {
            "ai.wire.bedrock"
        } else if lower.contains("image") {
            "ai.wire.openrouter-images"
        } else if lower.contains("auth") || lower.contains("oauth") || lower.contains("credential")
        {
            "ai.auth.precedence"
        } else if lower.contains("transform") || lower.contains("handoff") {
            "ai.handoff.thinking"
        } else if lower.contains("partial") || lower.contains("tool") {
            "ai.tool.schema"
        } else if lower.contains("cost") || lower.contains("usage") {
            "ai.usage.cache-cost"
        } else if lower.contains("stream") {
            "ai.stream.interleaving"
        } else {
            "ai.message.types"
        }
    } else if lower.starts_with("packages/storage/sqlite-node/") {
        "session.sqlite.transaction"
    } else if lower.starts_with("packages/agent/") {
        if lower.contains("harness") && lower.contains("compact") {
            "harness.compaction"
        } else if lower.contains("harness") && lower.contains("session") {
            "session.context"
        } else if lower.contains("harness") {
            "harness.snapshot"
        } else if lower.contains("queue") || lower.contains("steer") || lower.contains("follow") {
            "agent.queue.steer"
        } else if lower.contains("tool") {
            "agent.tools.preflight"
        } else {
            "agent.events.basic"
        }
    } else if lower.starts_with("packages/tui/") {
        if lower.contains("overlay") {
            "tui.overlay"
        } else if lower.contains("editor")
            || lower.contains("undo")
            || lower.contains("kill-ring")
            || lower.contains("word-navigation")
            || lower.contains("autocomplete")
        {
            "tui.editor"
        } else if lower.contains("key") || lower.contains("stdin") {
            "tui.keys"
        } else {
            "tui.diff"
        }
    } else if lower.contains("/tools/") || lower.contains("tool-") {
        if lower.contains("read") {
            "tools.read"
        } else if lower.contains("write") {
            "tools.write"
        } else if lower.contains("edit") {
            "tools.edit"
        } else if lower.contains("bash") {
            "tools.bash"
        } else if lower.contains("grep") {
            "tools.grep"
        } else if lower.contains("find") {
            "tools.find"
        } else if lower.contains("ls") {
            "tools.ls"
        } else {
            "tools.truncation"
        }
    } else if lower.contains("extension") {
        "extensions.reducer"
    } else if lower.contains("rpc") {
        "rpc.commands"
    } else if lower.contains("skill") {
        "resources.skills"
    } else if lower.contains("prompt") || lower.contains("resource") {
        "resources.prompts"
    } else if lower.contains("setting") {
        "settings.layers"
    } else if lower.contains("trust") {
        "trust.gate"
    } else if lower.contains("compact") {
        "harness.compaction"
    } else if lower.contains("branch") {
        "harness.branch-summary"
    } else if lower.contains("session") {
        "session.tree"
    } else {
        "harness.snapshot"
    }
}

/// Normalize a JSON value according to the stable cross-runner rules.
pub fn normalize(value: Value) -> Value {
    let workspace = std::env::current_dir().ok().map(|path| portable(&path));
    let home = std::env::var_os(if cfg!(windows) { "USERPROFILE" } else { "HOME" })
        .map(PathBuf::from)
        .map(|path| portable(&path));
    let temp = Some(portable(std::env::temp_dir()));
    let mut state = NormalizeState {
        ids: HashMap::new(),
        next_id: 1,
        workspace,
        home,
        temp,
    };
    state.value(value, None)
}

struct NormalizeState {
    ids: HashMap<String, String>,
    next_id: usize,
    workspace: Option<String>,
    home: Option<String>,
    temp: Option<String>,
}

impl NormalizeState {
    fn value(&mut self, value: Value, key: Option<&str>) -> Value {
        match value {
            Value::Object(object) => self.object(object, key),
            Value::Array(items) => Value::Array(
                items
                    .into_iter()
                    .map(|item| self.value(item, key))
                    .collect(),
            ),
            Value::String(text) => Value::String(self.string(&text, key)),
            Value::Number(number) => Value::Number(normalize_number(number)),
            other => other,
        }
    }

    fn object(&mut self, object: Map<String, Value>, parent: Option<&str>) -> Value {
        let mut entries: Vec<_> = object.into_iter().collect();
        entries.sort_by(|left, right| left.0.cmp(&right.0));
        let is_headers = parent.is_some_and(|key| key.eq_ignore_ascii_case("headers"));
        let mut output = Map::new();
        for (mut key, value) in entries {
            if is_volatile(&key) {
                continue;
            }
            if is_headers {
                key = key.to_ascii_lowercase();
                if is_sensitive_header(&key) {
                    output.insert(key, Value::String("<redacted>".to_owned()));
                    continue;
                }
            }
            if is_timestamp(&key) {
                output.insert(key, Value::String("<timestamp>".to_owned()));
            } else if is_identifier(&key) {
                output.insert(key, self.identifier(value));
            } else {
                let normalized = self.value(value, Some(&key));
                output.insert(key, normalized);
            }
        }
        Value::Object(output)
    }

    fn identifier(&mut self, value: Value) -> Value {
        let raw = match value {
            Value::Null => return Value::Null,
            Value::String(text) => text,
            other => other.to_string(),
        };
        let stable = if let Some(stable) = self.ids.get(&raw) {
            stable.clone()
        } else {
            let stable = format!("id-{}", self.next_id);
            self.next_id += 1;
            self.ids.insert(raw, stable.clone());
            stable
        };
        Value::String(stable)
    }

    fn string(&self, text: &str, key: Option<&str>) -> String {
        let mut normalized: String = text
            .replace("\r\n", "\n")
            .replace('\r', "\n")
            .nfc()
            .collect();
        if key.is_some_and(is_path_key) {
            normalized = normalized.replace('\\', "/");
            if normalized.len() >= 2 && normalized.as_bytes()[1] == b':' {
                normalized.replace_range(0..1, &normalized[0..1].to_ascii_lowercase());
            }
            normalized = replace_root(normalized, self.workspace.as_deref(), "<workspace>");
            normalized = replace_root(normalized, self.home.as_deref(), "<home>");
            normalized = replace_root(normalized, self.temp.as_deref(), "<temp>");
        }
        normalized
    }
}

fn normalize_number(number: Number) -> Number {
    if number
        .as_f64()
        .is_some_and(|value| value == 0.0 && value.is_sign_negative())
    {
        Number::from(0)
    } else {
        number
    }
}

fn is_volatile(key: &str) -> bool {
    matches!(
        key.to_ascii_lowercase().as_str(),
        "duration" | "durationms" | "elapsed" | "requestid" | "traceid"
    )
}

fn is_timestamp(key: &str) -> bool {
    let key = key.to_ascii_lowercase();
    key == "timestamp"
        || key == "createdat"
        || key == "updatedat"
        || key == "startedat"
        || key == "endedat"
}

fn is_identifier(key: &str) -> bool {
    matches!(
        key.to_ascii_lowercase().as_str(),
        "id" | "sessionid" | "parentid" | "toolcallid" | "messageid"
    )
}

fn is_path_key(key: &str) -> bool {
    let key = key.to_ascii_lowercase();
    key == "path"
        || key == "cwd"
        || key.ends_with("path")
        || key.ends_with("dir")
        || key.ends_with("directory")
}

fn is_sensitive_header(key: &str) -> bool {
    matches!(
        key,
        "authorization" | "api-key" | "x-api-key" | "proxy-authorization" | "cookie" | "set-cookie"
    )
}

fn replace_root(mut value: String, root: Option<&str>, replacement: &str) -> String {
    if let Some(root) = root {
        if !root.is_empty() {
            let lower_value = value.to_ascii_lowercase();
            let lower_root = root.to_ascii_lowercase();
            if lower_value.starts_with(&lower_root) {
                value.replace_range(..root.len(), replacement);
            }
        }
    }
    value
}

fn portable(path: impl AsRef<Path>) -> String {
    path.as_ref().to_string_lossy().replace('\\', "/")
}

/// Execute a versioned request.
///
/// # Errors
///
/// Returns an error for an unsupported protocol version or operation.
pub fn execute(request: Request) -> Result<Value> {
    if request.version != 1 {
        return Err(Error::Operation(format!("version {}", request.version)));
    }
    match request.operation.as_str() {
        "normalize" | "canonical" | "echo" => Ok(normalize(request.value)),
        operation => Err(Error::Operation(operation.to_owned())),
    }
}

/// Encode canonical JSON without a trailing newline.
///
/// # Errors
///
/// Returns an error when the JSON value cannot be encoded.
pub fn encode(value: &Value) -> Result<Vec<u8>> {
    serde_json::to_vec(value).map_err(Error::from)
}

/// Compare all JSON fixtures with the Rust and Node reference runners.
///
/// # Errors
///
/// Returns an error for missing fixtures, runner I/O or failures, invalid
/// output, or a semantic mismatch.
pub fn compare(root: &Path, manifest: &Manifest) -> Result<usize> {
    let fixture_dir = root.join(&manifest.runners.fixture_dir);
    let mut fixtures = Vec::new();
    if fixture_dir.exists() {
        for item in WalkDir::new(&fixture_dir).follow_links(false) {
            let item = item.map_err(|source| Error::Io {
                path: fixture_dir.clone(),
                source: std::io::Error::other(source.to_string()),
            })?;
            if item.file_type().is_file() && item.path().extension() == Some(OsStr::new("json")) {
                fixtures.push(item.path().to_path_buf());
            }
        }
    }
    fixtures.sort();
    if fixtures.is_empty() {
        return Err(Error::Validation(format!(
            "no JSON fixtures found under `{}`",
            manifest.runners.fixture_dir.display()
        )));
    }
    let current = std::env::current_exe().map_err(|source| Error::Io {
        path: PathBuf::from("<current-executable>"),
        source,
    })?;
    let mut compared = 0;
    for fixture in fixtures {
        let input = fs::read(&fixture).map_err(|source| Error::Io {
            path: fixture.clone(),
            source,
        })?;
        let rust = run_child(root, "rust", &current, &[OsStr::new("run")], &input)?;
        let reference_program = manifest
            .runners
            .reference
            .first()
            .ok_or_else(|| Error::Validation("reference command is empty".to_owned()))?;
        let reference_args: Vec<_> = manifest
            .runners
            .reference
            .iter()
            .skip(1)
            .map(OsStr::new)
            .collect();
        let reference = run_child(
            root,
            "reference",
            Path::new(reference_program),
            &reference_args,
            &input,
        )?;
        let rust_value: Value = serde_json::from_slice(&rust)?;
        let reference_value: Value = serde_json::from_slice(&reference)?;
        let rust_value = normalize(rust_value);
        let reference_value = normalize(reference_value);
        if rust_value != reference_value {
            return Err(Error::Mismatch {
                fixture,
                rust: String::from_utf8_lossy(&encode(&rust_value)?).into_owned(),
                reference: String::from_utf8_lossy(&encode(&reference_value)?).into_owned(),
            });
        }
        compared += 1;
    }
    Ok(compared)
}

fn run_child(
    root: &Path,
    name: &str,
    program: &Path,
    args: &[&OsStr],
    input: &[u8],
) -> Result<Vec<u8>> {
    let mut child = Command::new(program)
        .args(args)
        .current_dir(root)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|source| Error::Io {
            path: program.to_path_buf(),
            source,
        })?;
    child
        .stdin
        .take()
        .expect("piped standard input is present")
        .write_all(input)
        .map_err(|source| Error::Io {
            path: program.to_path_buf(),
            source,
        })?;
    let output = child.wait_with_output().map_err(|source| Error::Io {
        path: program.to_path_buf(),
        source,
    })?;
    if !output.status.success() {
        return Err(Error::Runner {
            runner: name.to_owned(),
            status: output.status.to_string(),
            stderr: String::from_utf8_lossy(&output.stderr).trim().to_owned(),
        });
    }
    if output.stdout.is_empty() {
        return Err(Error::Empty(name.to_owned()));
    }
    Ok(output.stdout)
}

/// Read all bytes from a file or standard input.
///
/// # Errors
///
/// Returns an error when the selected file or standard input cannot be read.
pub fn read_input(path: Option<&Path>) -> Result<Vec<u8>> {
    if let Some(path) = path {
        fs::read(path).map_err(|source| Error::Io {
            path: path.to_path_buf(),
            source,
        })
    } else {
        let mut bytes = Vec::new();
        std::io::stdin()
            .read_to_end(&mut bytes)
            .map_err(|source| Error::Io {
                path: PathBuf::from("<stdin>"),
                source,
            })?;
        Ok(bytes)
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{Request, encode, execute, normalize};

    #[test]
    fn sorts_keys_and_normalizes_text() {
        let value = json!({"z": "a\r\nb", "a": {"y": 2, "x": 1}});
        let bytes = encode(&normalize(value)).unwrap();
        assert_eq!(
            String::from_utf8(bytes).unwrap(),
            r#"{"a":{"x":1,"y":2},"z":"a\nb"}"#
        );
    }

    #[test]
    fn maps_identifiers_by_first_observation() {
        let value = json!({
            "id": "real-a",
            "children": [
                {"parentId": "real-a", "id": "real-b"},
                {"id": "real-a"}
            ]
        });
        assert_eq!(
            normalize(value),
            json!({
                "children": [
                    {"id": "id-1", "parentId": "id-2"},
                    {"id": "id-2"}
                ],
                "id": "id-2"
            })
        );
    }

    #[test]
    fn redacts_case_insensitive_headers() {
        let value = json!({
            "headers": {
                "Authorization": "Bearer secret",
                "X-Api-Key": "secret",
                "Accept": "application/json"
            }
        });
        assert_eq!(
            normalize(value),
            json!({
                "headers": {
                    "accept": "application/json",
                    "authorization": "<redacted>",
                    "x-api-key": "<redacted>"
                }
            })
        );
    }

    #[test]
    fn executes_versioned_request() {
        let result = execute(Request {
            version: 1,
            operation: "normalize".to_owned(),
            value: json!({"b": 2, "a": 1}),
        })
        .unwrap();
        assert_eq!(result, json!({"a": 1, "b": 2}));
    }
}
