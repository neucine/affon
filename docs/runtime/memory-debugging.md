# Memory Debugging

Affon exposes runtime and compute diagnostics through `std:telemetry`.

- `telemetry.metrics()` returns a flat metric snapshot (`id`, `scope`, `name`, `kind`, `unit`, `value`, `count`, `sum`, `min`, `max`).

Example:

```ts
import telemetry from 'std:telemetry'

const snapshot = telemetry.metrics()
const metalPoolBytes = snapshot.find(
  (metric) => metric.scope === 'compute.memory' && metric.name === 'metal_pool_live_bytes',
)?.value ?? 0
```

Runtime metrics and traces are exposed through the telemetry console and
Hao's `std:telemetry` module, using `scope` strings such as `runtime.memory`,
`compute.storage`, and `compute.execution`.

## Quick Workflow

1. Capture a metrics snapshot (`before`).
2. Run workload.
3. Capture another snapshot (`after`).
4. Compare the snapshots and investigate growing live-byte or peak-byte metrics.

Example:

```ts
import telemetry from 'std:telemetry'

const before = telemetry.metrics();
runWork();
const after = telemetry.metrics();

console.log(before.length, after.length);
```

## Recommended Memory Signals

Look for metrics with:

- `scope === "runtime.memory"` for Hao runtime allocator/process memory
- `scope === "compute.storage"` for tensor storage allocation_count and live bytes
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
- `scope === "compute.memory" && name === "metal_pool_live_buffer_count"`
- `scope === "compute.memory" && name === "metal_pool_hit_count"`
- `scope === "compute.memory" && name === "metal_pool_miss_count"`
- `scope === "compute.memory" && name === "metal_pool_trim_bytes"`
- `scope === "compute.memory" && name === "metal_device_current_allocated_bytes"`

Use `scope === "compute.execution"` counters such as `fusion_fallback_count`,
`transfer_to_host_bytes`, and `contiguity_fixup_bytes` when checking execution
plans, backend transfer churn, or graph fusion behavior.

## Troubleshooting Guide

### Activity Monitor Memory Is High During Metal Training

Symptom:

- macOS Activity Monitor reports a large memory footprint.
- `vmmap -summary <pid>` attributes much of the footprint to
  `IOAccelerator (graphics)`.
- `compute.storage.live_metal_bytes` is stable or much smaller than Activity
  Monitor.
- `runtime.memory.qjs_heap_used_bytes` and CPU live-byte metrics are stable.

This usually means the pressure is on the Metal/driver side, not the JavaScript
heap or ordinary CPU allocations. A common cause is high-rate creation and
release of large transient `MTLBuffer` objects. Metal buffer ownership may still
be correct while process footprint remains high because the driver and virtual
memory system can retain mappings, residency, and allocation metadata around
recent GPU resources.

Check:

1. Compare `compute.storage.live_metal_bytes` with
   `compute.memory.metal_device_current_allocated_bytes`.
2. Compare both with `runtime.memory.physical_footprint_bytes` or `vmmap`.
3. Inspect pool activity:
   - low `metal_pool_hit_count`
   - high `metal_pool_miss_count`
   - repeated over-threshold or scratch allocations

If `metal_device_current_allocated_bytes` is stable but Activity Monitor grows,
the likely issue is Metal resource churn or IOAccelerator VM residency rather
than unreleased Affon tensor storage.

### Large Metal Allocations Bypass Reuse

Affon pools Metal storage to avoid repeatedly creating large `MTLBuffer`
resources. Allocations above `AFFON_METAL_POOL_OVERSIZE_THRESHOLD_BYTES` bypass
that pool and are destroyed directly. Repeated direct allocation of large
buffers is risky in long-running training loops, especially for unfused language
model loss paths that materialize `[tokens, vocab]` logits.

Known example:

- `batch=4`
- `seq=255`
- `vocab=50257`
- `dtype=f32`

The logits buffer alone is roughly:

```text
4 * 255 * 50257 * 4 bytes ~= 195 MiB
```

Additional transpose, contiguous, backward, or workspace buffers can add more
large transient allocations.

Action:

- Prefer kernels or graph plans that avoid materializing full logits when
  possible.
- Increase `AFFON_METAL_POOL_OVERSIZE_THRESHOLD_BYTES` only when the model shape
  is expected and the machine has enough memory.
- Watch `metal_pool_live_bytes`, `metal_pool_peak_bytes`, and
  `metal_device_current_allocated_bytes` after changing the threshold.
- Treat repeated over-threshold allocations as a performance and footprint risk,
  even if there is no ownership leak.

Example:

```sh
AFFON_METAL_POOL_OVERSIZE_THRESHOLD_BYTES=536870912 \
  ./zig-out/bin/affon run train.ts
```

### Scratch/Workspace Churn

Scratch storage is logically temporary, but on Metal it can still be expensive
if each temporary maps to a fresh `MTLBuffer`. If Activity Monitor grows while
`compute.storage.live_metal_bytes` remains bounded, inspect workspace-heavy
paths such as:

- contiguity fixups
- unfused elementwise chains
- unfused LM-head loss
- reductions with temporary row buffers

The expected stable behavior is:

- `metal_pool_hit_count` increases after warmup.
- `metal_pool_live_bytes` reaches a bounded plateau.
- `metal_device_current_allocated_bytes` reaches a bounded plateau.
- `IOAccelerator (graphics)` region count stops growing.

If pool misses keep increasing without later hits, the workload is probably
generating many incompatible exact sizes or bypassing pooled storage.

### Confirming With `vmmap`

On macOS, use the helper script for a repeatable process-level check:

```sh
AFFON_METAL_MAX_PHYS_GROWTH_MB=512 \
AFFON_METAL_MAX_IOACCEL_REGION_GROWTH=5000 \
  tools/metal-footprint-watch.sh -i 15 -s 20 -o /tmp/affon-metal.jsonl -- \
  bash -lc 'RUNTIME_PACKAGE_PATH=packages apps/decoder-lm/run.sh wikitext-debug'
```

For the focused LM-head pressure path:

```sh
tools/metal-footprint-watch.sh -i 5 -s 12 -o /tmp/affon-lmhead.jsonl -- \
  bash -lc 'RUNTIME_PACKAGE_PATH=packages zig-out/bin/affon run tools/lm-head-loss-pressure-repro.ts'
```

Interpretation:

- current physical footprint should plateau after warmup.
- `IOAccelerator (graphics)` region count should not grow steadily.
- `metal_device_current_allocated_bytes` should be close to the expected live
  Metal resource budget plus the bounded pool.

## Regression Fixture

See:

- [packages/lm/test/memory.test.ts](../../packages/lm/test/memory.test.ts)
- [tools/metal-footprint-watch.sh](../../tools/metal-footprint-watch.sh)
- [tools/lm-head-loss-pressure-repro.ts](../../tools/lm-head-loss-pressure-repro.ts)
