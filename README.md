# Ri

Ri is an Edition 2024 Rust workspace for a behavior-compatible, native implementation of
the Pi agent stack. The fixed reference source is `ref/pi` at commit
`518855dd502220d0c6480fb8863e2e7f8799893f`; `docs/pi-reference-analysis.md` and
`conformance/manifest.yaml` define the compatibility baseline.

This workspace baseline establishes crate boundaries, dependency direction, lint policy,
continuous integration, and an executable conformance manifest contract. Runtime crates
whose implementation belongs to later milestones intentionally expose only accurate
crate-level documentation at this stage.

## Crates

- `ri-ai` — AI data, provider, and wire-protocol layer.
- `ri-agent` — low-level agent loop and state machine.
- `ri-session` and `ri-storage-sqlite` — session model and optional SQLite storage.
- `ri-tools` — built-in coding tools and execution environment.
- `ri-ext` and `ri-ext-wasm` — native extension contracts and optional WASM host.
- `ri-harness` — the single high-level orchestration layer.
- `ri-sdk` and `ri` — user-facing SDK and facade.
- `ri-rpc`, `ri-tui`, `ri-compat`, and `ri-cli` — product protocols and interfaces.
- `ri-macros` — procedural macros.
- `ri-testkit` — test-only support.
- `ri-conformance` — manifest validation and deterministic runner comparison.

Heavy boundary crates are opt-in: `ri-session/sqlite`, `ri-ext/wasm`, and the
`ri-cli` interface features do not enter a default build unless selected.

## Conformance commands

```text
cargo run -p ri-conformance -- validate
cargo run -p ri-conformance -- normalize --input value.json
cargo run -p ri-conformance -- run --input request.json
cargo run -p ri-conformance -- compare
```

`normalize` accepts any JSON value. `run` accepts a versioned request:

```json
{"version":1,"operation":"normalize","value":{"b":2,"a":1}}
```

Both commands emit canonical JSON without a trailing newline. `compare` sends each JSON
fixture to the configured Rust and reference commands and fails on a command error,
empty output, invalid JSON, or a normalized mismatch.

## Development checks

```text
cargo fmt --all -- --check
cargo check --workspace --all-targets --all-features
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
cargo deny check
```
