# Testing and conformance

Ri uses the pinned Pi source and tests as a behavioral oracle, not as a build-time
dependency.

## Gates

1. **Transform** — deterministic message, schema, partial JSON, cost, Session tree,
   compaction boundary, prompt, and terminal-width behavior.
2. **Agent** — scripted assistant streams, exact event traces, tools, queue drains,
   retry, cancellation, save points, and compaction.
3. **Wire** — loopback HTTP/SSE/WebSocket servers verify payloads, headers, arbitrary
   chunk boundaries, malformed frames, usage, retry, and abort.
4. **Session/RPC** — JSONL framing, migration, transaction, reopen, import/export,
   command/response, and extension UI traces.
5. **Live** — explicit credential-gated Provider, OAuth, cache affinity, thinking
   signature, Bedrock, terminal, and platform shell contracts.

`conformance/manifest.yaml` owns every compatibility row. The generated
`conformance/reference-tests.json` maps every reference test file to one row and
gate. A row can become `passing` only when its reference, Rust test target, and gate
are all present.

## Determinism

- IDs are mapped by first observation.
- Timestamps and volatile durations are normalized.
- Paths use `/` and stable workspace/home/temp roots.
- Sensitive headers are redacted.
- JSON object keys are lexical; array and event order is preserved.
- Async tests use barriers, paused time, and explicit cancellation rather than
  timing sleeps.

## Commands

```text
cargo fmt --all -- --check
cargo check --workspace --all-targets --all-features
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo nextest run --workspace --all-features
cargo test --workspace --all-features --doc
cargo llvm-cov --workspace --all-features
cargo run -p ri-conformance -- validate
cargo run -p ri-conformance -- compare
cargo deny check
```

Live suites require both a feature-specific opt-in variable and the corresponding
credential. Missing credentials skip the live contract; they never redirect a test
to a fake production Provider.
