# MM Public API Contract

`mm/*` is the memory-management subsystem contract for allocation paths across compute and host integrations.

This document is the current contract for `mm/*`.

## Public Entry

- `src/mm/index.zig`

All new callsites should prefer `mm` (from `index.zig`) over direct imports of internal implementation modules.

## Telemetry Contract

`mm` does not define a separate telemetry surface.

- Aggregates are exported through `obs.metrics`.
- Event/span semantics are exported through `obs.trace`.
- `mm.events` is removed from `mm/*`.
- `mm.metrics` is an internal implementation detail for metric updates, not a public read API.

## Stable Types

- `AllocationPolicy`
- `Region`
- `RegionMetadata`
- `Intention`
- `RegionConfig`

## Lifecycle Model

`Region` is the primary lifecycle abstraction.

A region defines:

- lifecycle boundary for allocations
- backing `device` (for example `cpu`, `metal`, future `gpu/*`)
- allocation `policy` (for example `owned`, `scratch`, `pool`)

Allocations are made from a region and may carry an `intention` tag for attribution,
diagnostics, and telemetry.

### Region-Centric Semantics

- Lifecycle ownership is attached to region, not raw device alone.
- Device describes where memory is backed/executed.
- Policy describes allocator strategy within that region.
- Intention describes callsite purpose of the allocation.
- Allocator state belongs to region.

### Deterministic Execution Rule

- Region identity determines allocation behavior deterministically.
- There is no per-allocation heuristic rerouting once region is chosen.
- If a region is not ready, allocation fails explicitly.
- No silent fallback to another region/device.

### Region Registry and Readiness

Regions are explicitly declared and registered.

- A region may be declared even when its device is not currently available.
- A region has readiness state (`ready` / `not_ready`).
- Readiness is checked before allocation.
- For non-ready regions, callers receive deterministic errors.

This preserves stable region identity while supporting optional backends.

### Allocator Access Pattern

Public model is region-first.

- Region setup:
  - `registerRegion(region, config)`
  - `setRegionReady(region, ready)`
  - `isRegionReady(region)`
- Zig integration path:
  - `mm.allocator(region, intention)` for APIs requiring `std.mem.Allocator`

The allocator is derived from region, never the other way around.
Intention is expressed when providing the allocator handle.

### Intention Model

- Intention is attached at allocator-handle creation time.
- Multiple allocator handles may be created from the same region with different intentions.
- All handles from the same region still allocate from the same region backend/state.
- Generic downstream code can keep accepting plain `std.mem.Allocator`.
- Attribution remains centralized because the mm-provided allocator wrapper carries intention metadata.

### Backend Extensibility

- Different devices may have multiple internal allocators/subregions.
- Those internals are implementation details behind region semantics.
- Public API remains region-centric and backend-agnostic.
- Future GPU backends should fit by adding region configurations, not by changing lifecycle semantics.
- region readiness (`isRegionReady`) is the capability gate.

### Explicit Limits and Tradeoffs

- Startup/initialization is more explicit (region registration lifecycle).
- Callers must handle non-ready region errors.
- Cross-region operations must be explicit (copy/move APIs), not implicit.
- Policy tuning is still required per workload (deterministic does not remove tuning).
- Runtime device availability transitions need explicit state rules.

## Stable Functions

- region registry:
  - `registerRegion(region, config)`
  - `setRegionReady(region, ready)`
  - `getRegionConfig(region)`
  - `isRegionReady(region)`
- allocator handoff:
  - `allocator(region, intention)`
- metadata:
  - `regionMetadata(region)`

Public reads should use `obs.metrics.snapshot(...)` rather than `mm` snapshots.

## Guardrails

- Keep region registration/readiness deterministic.
- Preserve explicit failure for non-ready regions; no hidden fallback.
- Keep telemetry routed through `obs.metrics` / `obs.trace`.
- Keep `mm.metrics` internal-only; do not add external consumers of `mm` snapshots.
