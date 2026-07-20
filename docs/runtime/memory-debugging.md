# Memory Debugging

Affon exposes runtime and compute diagnostics through `std:telemetry` and cache
trimming through `Affon.trimMemory()`.

- `telemetry.metrics()` returns a flat metric snapshot (`id`, `scope`, `name`, `kind`, `unit`, `value`, `count`, `sum`, `min`, `max`).
- `Affon.trimMemory()` trims internal caches and returns released bytes.

Example:

```ts
import telemetry from 'std:telemetry'

const snapshot = telemetry.metrics()
const metalPoolBytes = snapshot.find(
  (metric) => metric.scope === 'compute.memory' && metric.name === 'metal_pool_live_bytes',
)?.value ?? 0
```

The old `src/obs` API has been retired in this package boundary. Runtime
metrics and traces are exposed through Hao's `std:telemetry` module, using
`scope` strings such as `runtime.memory`, `compute.storage`, and
`compute.execution`.

## Quick Workflow

1. Capture a metrics snapshot (`before`).
2. Run workload.
3. Capture another snapshot (`after`).
4. Optionally call `Affon.trimMemory()` and snapshot again.

Example:

```ts
import telemetry from 'std:telemetry'

const before = telemetry.metrics();
runWork();
const after = telemetry.metrics();
const trimmed = Affon.trimMemory();
const postTrim = telemetry.metrics();

console.log('trimmed bytes', trimmed);
console.log(before.length, after.length, postTrim.length);
```

## Recommended Memory Signals

Look for metrics with:

- `scope === "runtime.memory"` for Hao runtime allocator/process memory
- `scope === "compute.storage"` for tensor storage allocations and live bytes
- `scope === "compute.memory"` for Affon compute memory regions and Metal pool activity

Typical examples include CPU/Metal live bytes, CPU/Metal peak bytes, pool bytes,
process footprint, and allocator event counters.

Useful compute storage metrics:

- `scope === "compute.storage" && name === "live_bytes"`
- `scope === "compute.storage" && name === "peak_bytes"`
- `scope === "compute.storage" && name === "live_cpu_bytes"`
- `scope === "compute.storage" && name === "peak_cpu_bytes"`
- `scope === "compute.storage" && name === "live_metal_bytes"`
- `scope === "compute.storage" && name === "peak_metal_bytes"`

Useful compute memory metrics:

- `scope === "compute.memory" && name === "metal_pool_live_bytes"`
- `scope === "compute.memory" && name === "metal_pool_peak_bytes"`
- `scope === "compute.memory" && name === "metal_pool_live_buffers"`
- `scope === "compute.memory" && name === "metal_pool_hits"`
- `scope === "compute.memory" && name === "metal_pool_misses"`
- `scope === "compute.memory" && name === "metal_pool_trim_bytes"`

Use `scope === "compute.execution"` counters such as `fusion_fallback`,
`transfer_to_host_bytes`, and `contiguity_fixup_bytes` when checking execution
plans, backend transfer churn, or graph fusion behavior.

## Regression Fixture

See:

- [packages/lm/test/memory.test.ts](/Users/chao.yang/Private/affon-next/packages/lm/test/memory.test.ts)
