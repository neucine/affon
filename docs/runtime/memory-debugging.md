# Memory Debugging

Affon now exposes memory diagnostics through `Affon.metrics()` and cache trimming through `Affon.trimMemory()`.

- `Affon.metrics()` returns a flat metric snapshot (`domain`, `group`, `name`, `kind`, `unit`, `value`).
- `Affon.trimMemory()` trims internal caches and returns released bytes.

For long-running CLI jobs, Affon can also expose a local read-only observer:

- enable with `AFFON_OBSERVER_ENABLED=1`
- optional port override with `AFFON_OBSERVER_PORT=<port>`
- if the port is `0` or omitted, Affon binds an ephemeral local port on `127.0.0.1`

The local HTTP observer depends on the native networking substrate being enabled in the build. If that substrate is unavailable, `Affon.metrics()`, `Affon.trace(...)`, and `Affon.span(...)` remain available in-process, but the HTTP observer endpoint is not started.

Current Phase 1 endpoints:

- `GET /health`
- `GET /runtime`
- `GET /metrics`
- `GET /traces?since=<cursor>&limit=<n>`
- `PATCH /runtime`

Example:

```sh
AFFON_OBSERVER_ENABLED=1 AFFON_OBSERVER_PORT=0 affon run script.ts
```

Affon prints the bound URL to stderr, for example:

```txt
Affon observer listening on http://127.0.0.1:56513
```

Then you can poll it externally:

```sh
curl http://127.0.0.1:56513/metrics
```

Inspect current runtime state and runtime-mutable config:

```sh
curl http://127.0.0.1:56513/runtime
```

Or use the repo helper for periodic polling:

```sh
tools/observe_runtime.py http://127.0.0.1:56513 --endpoint metrics --interval 2
```

For recent completed spans:

```sh
curl "http://127.0.0.1:56513/traces?since=0&limit=64"
```

The trace response includes:

- `events`: recent completed spans
- `next_cursor`: the next cursor to use for incremental polling
- `dropped_before_cursor`: whether the requested cursor was older than the bounded in-memory window

Trace filters are optional and apply after the bounded recent-span snapshot is copied:

- `domain=<runtime|compute|memory|support>`
- `group=<group>`
- `name=<span-name>`
- `min_elapsed_ns=<threshold>`

Example:

```sh
curl "http://127.0.0.1:56513/traces?since=120&limit=32&domain=compute&group=execution&min_elapsed_ns=1000000"
```

The same filters work through the helper:

```sh
tools/observe_runtime.py http://127.0.0.1:56513 --endpoint traces --interval 1 --limit 32 --domain compute --group execution --min-elapsed-ns 1000000
```

## Runtime Config Control

`GET /runtime` returns:

- process/runtime metadata
- observer bind info
- current runtime-mutable config

Today the returned config includes:

- `device`
- `csv`
- `repr`
- `trace`

`PATCH /runtime` supports controlled live mutation of the runtime-mutable trace
emit policy.

Example:

```sh
curl -X PATCH http://127.0.0.1:56513/runtime \
  -H 'Content-Type: application/json' \
  --data '{
    "config": {
      "trace": {
        "enabled": true,
        "min_elapsed_ns": 1000000,
        "sample_rate": 0.25
      }
    }
  }'
```

Current writable trace fields:

- `enabled`
- `min_elapsed_ns`
- `sample_rate`

Rules:

- `sample_rate` must be between `0` and `1`
- `min_elapsed_ns` must be a non-negative integer
- bootstrap-only settings are not writable through `/runtime`

## Quick Workflow

1. Capture a metrics snapshot (`before`).
2. Run workload.
3. Capture another snapshot (`after`).
4. Optionally call `Affon.trimMemory()` and snapshot again.

Example:

```ts
const before = Affon.metrics();
runWork();
const after = Affon.metrics();
const trimmed = Affon.trimMemory();
const postTrim = Affon.metrics();

console.log('trimmed bytes', trimmed);
console.log(before.length, after.length, postTrim.length);
```

## Recommended Memory Signals

Look for metrics with:

- `domain === "memory"`
- `group === "owned.current_bytes"`
- `group === "observed.current_bytes"`
- `group === "allocator.events"`

Typical examples include CPU/Metal live bytes, pool bytes, process footprint, and allocator event counters.

Additional useful categories:

- `group === "owned.current_count"`
- `group === "observed.current_count"`
- `group === "owned.peak_bytes"`
- compute-domain `execution.graph_fusion` counters when checking graph execution plans against fusion execution

## Regression Fixture

See:

- [test/e2e/config/memory-regression.test.ts](/Users/chao.yang/Private/affon/test/e2e/config/memory-regression.test.ts)
