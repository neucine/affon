# Memory Debugging

Affon exposes process, tensor-storage, and device-pool diagnostics through
`std:telemetry`.

```ts
import telemetry from 'std:telemetry'

const snapshot = telemetry.metrics()
const liveTensorBytes = snapshot.find(
  metric => metric.scope === 'compute.storage' && metric.name === 'live_bytes',
)?.value ?? 0
```

Metric records include `scope`, `name`, `kind`, `unit`, and the current value;
counter and distribution records can also include aggregate fields. Match on
`scope` and `name`, not array position.

## Start with ownership

The Program API makes resource ownership explicit:

- a `Session` owns tensors, compiled executables, and execution state;
- `Executable.run(...)` returns tensors owned by that Session;
- `Tensor`, `Executable`, `ExecutionState`, and `Session` provide idempotent
  `dispose()` and `Symbol.dispose`;
- disposing a Session prevents new work, but live child objects keep their
  storage until they are also disposed.

For a suspected leak, first ensure long-lived collections are not retaining
output tensors or old `ExecutionState` objects. In bounded loops, dispose
per-iteration inputs and outputs deterministically.

## Compare snapshots

Capture the same metrics before warmup, after warmup, and after repeated steady
work:

```ts
import telemetry from 'std:telemetry'

const before = telemetry.metrics()
runWarmup()
const warm = telemetry.metrics()
runRepeatedWork()
const after = telemetry.metrics()

console.log({ before, warm, after })
```

Useful scopes are:

- `compute.storage` — live and peak tensor storage plus allocation counts
- `compute.memory` — compute allocator and accelerator-pool diagnostics
- `compute.execution` — transfers, fixups, and execution counters
- `runtime.memory` — JavaScript runtime and process memory

Read metric names from the snapshot produced by your build. Diagnostic metrics
can grow over time and are not themselves stable application APIs.

## Interpret trends

- A steadily rising `compute.storage/live_bytes` value usually means evaluated
  tensors or state remain reachable or undisposed.
- A stable live-byte value with a rising peak is normal when later requests use
  larger shapes.
- Accelerator pools deliberately retain reusable storage, so pool bytes can
  plateau above live tensor bytes without being an ownership leak.
- Process resident or physical footprint includes the JavaScript runtime,
  libraries, allocator arenas, driver mappings, and caches. It need not fall
  immediately when tensors are released.
- `item()` and `to_array()` synchronize accelerator work. Measure after a
  deliberate synchronization point when comparing iterations.

Warm up compilation and representative shapes before judging a steady-state
plateau. Varying shapes can populate additional executable or allocator cache
entries even when ownership is correct.

## Metal-specific checks

On macOS, compare three layers rather than relying on Activity Monitor alone:

1. Affon live tensor storage (`compute.storage`)
2. Metal allocator and pool metrics (`compute.memory`)
3. process physical footprint (`runtime.memory` or `vmmap`)

If live tensor bytes stabilize while pool bytes warm up and then stabilize, the
behavior is retention for reuse. If live tensor bytes rise with every request,
inspect application references and disposal first. If only process footprint
rises, investigate Metal resource churn and driver residency before calling it
a tensor leak.

The Metal pool is bounded by `AFFON_METAL_POOL_MAX_TOTAL_BYTES`. Advanced users
can tune `AFFON_METAL_POOL_OVERSIZE_THRESHOLD_BYTES` and
`AFFON_METAL_POOL_SIZE_CLASSES`; see [Runtime Configuration](./configuration.md).
Change one setting at a time and compare the same workload after warmup.

## Reproducible reports

A useful memory report includes:

- Affon revision and build mode
- operating system, device, and relevant environment settings
- Program name, input shapes, dtypes, and iteration count
- snapshot values at baseline, post-warmup, and steady state
- whether host synchronization occurred before each snapshot
- the ownership and disposal pattern for inputs, outputs, states, executables,
  and Sessions

This separates an application retention bug, an Affon storage leak, bounded
pool growth, and process/driver accounting instead of treating every rising
number as the same failure.
