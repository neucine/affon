# Runtime Configuration

Affon reads public runtime configuration from environment variables during
startup. Settings are schema-backed: each field has a parser, default, and
mutability class.

- Startup settings are read during process initialization and locked after the
  runtime starts.
- Runtime settings are read during startup and may also be changed through the
  runtime config store when Affon exposes a control surface for that field.
- Invalid values are ignored or rejected by the field parser rather than being
  treated as arbitrary strings.

## Environment Variables

| Variable | Field | Mutability | Default | Values |
| --- | --- | --- | --- | --- |
| `AFFON_DEVICE` | `device.default` | runtime | `cpu` | `cpu`, `metal` |
| `AFFON_QJS_STACK_SIZE` | `quickjs.stack_size` | startup | `8388608` | positive integer bytes |
| `AFFON_LIBUV_THREADPOOL_SIZE` | `libuv.thread_pool_size` | startup | unset | positive integer |
| `AFFON_NATIVE_STACK_TRACE` | `debug.native_stack_trace` | runtime | `false` | boolean |
| `AFFON_METAL_THREADGROUP_SIZE` | `device.metal.threadgroup_size` | startup | `256` | positive integer |
| `AFFON_METAL_REDUCE_ALL_THRESHOLD` | `device.metal.reduce_all_threshold` | startup | `512` | positive integer |
| `AFFON_METAL_REDUCE_AXIS_THRESHOLD` | `device.metal.reduce_axis_threshold` | startup | `512` | positive integer |
| `AFFON_METAL_POOL_OVERSIZE_THRESHOLD_BYTES` | `device.metal.pool_oversize_threshold_bytes` | startup | `67108864` | positive integer bytes |
| `AFFON_CPU_PARALLEL_THRESHOLD` | `device.cpu.parallel_threshold` | runtime | `65536` | positive integer |
| `AFFON_CSV_CHUNK_SIZE` | `csv.chunk_size` | runtime | `65536` | positive integer bytes |
| `AFFON_REPR_MAX_ITEMS` | `repr.max_items` | runtime | `6` | positive integer |
| `AFFON_REPR_MAX_ROWS` | `repr.repr_max_rows` | runtime | `20` | positive integer |
| `AFFON_REPR_MAX_COLS` | `repr.repr_max_cols` | runtime | `12` | positive integer |
| `AFFON_OBSERVER_ENABLED` | `observer.enabled` | runtime | `false` | boolean |
| `AFFON_OBSERVER_PORT` | `observer.port` | runtime | `0` | integer, zero allowed |

Boolean values accept the shared config parser's normal boolean forms, including
`1`, `true`, `yes`, and `on` for enabled values, and `0`, `false`, `no`, and
`off` for disabled values.

## Legacy Aliases

The old ndarray representation environment names still populate the canonical
`repr` fields, but they are deprecated:

- `AFFON_NDARRAY_FORMAT_MAX_ITEMS` -> `AFFON_REPR_MAX_ITEMS`
- `AFFON_NDARRAY_REPR_MAX_ROWS` -> `AFFON_REPR_MAX_ROWS`
- `AFFON_NDARRAY_REPR_MAX_COLS` -> `AFFON_REPR_MAX_COLS`

Prefer the `AFFON_REPR_*` names in scripts and docs.

## Libuv Thread Pool

When `AFFON_LIBUV_THREADPOOL_SIZE` is set, Affon mirrors the parsed value into
`UV_THREADPOOL_SIZE` before the runtime starts. Set it in the environment before
launching Affon:

```sh
AFFON_LIBUV_THREADPOOL_SIZE=8 ./zig-out/bin/affon script.ts
```

## Examples

Run on Metal by default and keep tensor displays compact:

```sh
AFFON_DEVICE=metal AFFON_REPR_MAX_ITEMS=4 ./zig-out/bin/affon script.ts
```

The observer fields remain in the schema, but the retired `src/obs` HTTP
observer is not the current public diagnostics surface. Use `std:telemetry` for
in-process metrics and traces.
