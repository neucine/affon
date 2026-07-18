# Observability (`obs`) API Contract

`obs` is the stable observability subsystem for Affon.

It defines taxonomy and API contracts that other subsystems use for metrics and traces.

## Public Modules

- `obs.common`
- `obs.metrics`
- `obs.trace`

## Taxonomy (`obs.common`)

Stable enums:

- `MetricKind`: `gauge | counter`
- `MetricUnit`: `count | bytes | nanoseconds`
- `Mechanism`: `diagnostic | telemetry`
- `TelemetryChannel`: `traces | logs | metrics`
- `Domain`: `runtime | compute | memory | support`

## Metrics API (`obs.metrics`)

Stable types:

- `Id` (opaque metric identifier)
- `Definition`
- `Snapshot`

Stable functions:

- `register(definition: Definition) !Id`
- `set(id: Id, value: i64) void`
- `add(id: Id, delta: i64) void`
- `snapshot(buffer: []Snapshot) []const Snapshot`

Contract:

- Registering the same definition is idempotent and returns the same `Id`.
- `snapshot` returns a view over caller-provided buffer.

### Taxonomy Direction

Metrics identity should be modeled as:

- `domain` (enum taxonomy)
- `group` (free-form subsystem label)
- `name` (metric name)

## Trace API (`obs.trace`)

Stable types:

- `Scope`
- `Event`

Stable functions:

- `begin(domain: Domain, group: []const u8, name: []const u8) Scope`
- `recent(buffer: []Event) []const Event`

Contract:

- `Scope.end()` records one completed event.
- `begin()` requires explicit `domain` and `group`.

### Taxonomy Direction

Trace identity should follow the same model:

- `domain` (enum taxonomy)
- `group` (free-form subsystem label)
- `name` (span/operation name)

No scope-based trace entrypoint is part of the public API.

## Status

`obs/*` is the active observability surface across host and compute paths.
Legacy `telemetry/*` has been retired from active runtime wiring.
