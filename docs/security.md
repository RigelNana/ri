# Security model

Ri treats models, extensions, tools, projects, and credentials as separate trust
boundaries.

## Credentials

- Credentials are resolved explicitly per Provider.
- An existing stored credential owns its Provider; refresh or type errors never
  silently fall through to ambient environment credentials.
- Refresh uses a serialized read-modify-write lock with a second freshness check.
- Logs, protocol snapshots, and conformance fixtures redact authorization, API-key,
  proxy authorization, and cookie headers.
- Pi credentials are not imported automatically.

## Projects and resources

- Project settings, context files, prompts, skills, extensions, and packages load
  only after the project trust decision.
- Global trust extensions may participate in the decision, but project extensions
  cannot decide their own trust.
- Resource collisions retain provenance and follow deterministic first-wins
  precedence.

## Tools

- Tool parameters are schema-validated before hooks and execution.
- Filesystem mutations serialize by canonical path.
- Shell processes receive cancellation and timeout signals; platform adapters
  terminate process trees.
- Tool output is bounded before it enters model context. Full shell output may be
  written to an explicit temporary spill file.

## Extensions

Embedded Rust extensions run with the host process's authority and should be
reviewed like any other dependency.

Dynamic packages use the WASM Component Model. They receive no ambient filesystem,
network, process, UI, Session, or Provider access. Each capability is declared in
`ri-package.toml`, approved by policy, and represented by a host resource. The host
enforces fuel, memory, and deadline limits. Reload invalidates prior context handles,
so stale code cannot mutate a replacement Session.

## Network

Production Provider requests use real protocol adapters. Scripted Providers and
loopback servers live only in `ri-testkit`; they are not fallback paths. Proxy,
timeout, TLS, retry, OAuth, and Provider error behavior remains observable through
typed errors rather than hidden Provider switching.

## Persistence

JSONL Sessions are append-only. SQLite writes use WAL, `synchronous=FULL`, a busy
timeout, and one transaction for entry, sequence, active leaf, branch, and
materialized statistics. A failed transaction cannot advance in-memory state.
