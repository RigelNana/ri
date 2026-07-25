# Conformance

`manifest.yaml` is the machine-readable compatibility contract for the fixed Pi
reference commit. It separates deterministic transforms, scripted Agent traces,
local wire protocols, durable Session/RPC protocols, and credential-gated live
contracts.

Commands:

```text
cargo run -p ri-conformance -- validate
cargo run -p ri-conformance -- inventory
cargo run -p ri-conformance -- normalize --input value.json
cargo run -p ri-conformance -- run --input request.json
cargo run -p ri-conformance -- compare
```

`inventory` maps every upstream `*.test.ts`/`*.test.mjs` file to a stable feature
row. `compare` sends every JSON fixture to both the Rust executable and the
Node reference runner. Both sides emit canonical JSON without a trailing line
ending, so byte differences indicate a behavioral or normalization mismatch.

Loopback HTTP, SSE, and WebSocket test servers belong in `ri-testkit`; they are
never linked as production Provider fallbacks. Live contracts are skipped unless
their explicit credential and opt-in environment variables are present.
