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
| `AFFON_DEVICE` | `device.default` | runtime | auto-detected (`metal` on macOS when available, `cuda` on other platforms when available, otherwise `cpu`) | `cpu`, `metal`, `cuda` |
| `AFFON_CUDA_DEVICE` | CUDA device ordinal | startup | `0` | non-negative integer |
| `AFFON_NATIVE_STACK_TRACE` | `debug.native_stack_trace` | runtime | `false` | boolean |
| `AFFON_METAL_THREADGROUP_SIZE` | `device.metal.threadgroup_size` | startup | `256` | positive integer |
| `AFFON_METAL_REDUCE_ALL_THRESHOLD` | `device.metal.reduce_all_threshold` | startup | `512` | positive integer |
| `AFFON_METAL_REDUCE_AXIS_THRESHOLD` | `device.metal.reduce_axis_threshold` | startup | `512` | positive integer |
| `AFFON_METAL_POOL_OVERSIZE_THRESHOLD_BYTES` | `device.metal.pool_oversize_threshold_bytes` | startup | `536870912` | positive integer bytes |
| `AFFON_METAL_POOL_MAX_TOTAL_BYTES` | `device.metal.pool_max_total_bytes` | startup | `1073741824` | positive integer bytes |
| `AFFON_METAL_POOL_SIZE_CLASSES` | `device.metal.pool_size_classes` | startup | `65536:512,1048576:512,*:8` | comma-separated `threshold:max_buffers` entries plus final `*:max_buffers` catch-all |
| `AFFON_CPU_PARALLEL_THRESHOLD` | `device.cpu.parallel_threshold` | runtime | `65536` | positive integer |
| `AFFON_CSV_CHUNK_SIZE` | `csv.chunk_size` | runtime | `65536` | positive integer bytes |
| `AFFON_REPR_MAX_ITEMS` | `repr.max_items` | runtime | `6` | positive integer |
| `AFFON_REPR_MAX_ROWS` | `repr.repr_max_rows` | runtime | `20` | positive integer |
| `AFFON_REPR_MAX_COLS` | `repr.repr_max_cols` | runtime | `12` | positive integer |

Affon also loads these inherited Hao runtime settings before creating the
runtime:

| Variable | Field | Mutability | Default | Values |
| --- | --- | --- | --- | --- |
| `RUNTIME_QJS_STACK_SIZE` | `qjs.stack_size` | startup | `8388608` | positive integer bytes |
| `RUNTIME_LIBUV_THREADPOOL_SIZE` | `libuv.thread_pool_size` | startup | unset | positive integer |

Boolean values accept the shared config parser's normal boolean forms, including
`1`, `true`, `yes`, and `on` for enabled values, and `0`, `false`, `no`, and
`off` for disabled values.

## Libuv Thread Pool

When `RUNTIME_LIBUV_THREADPOOL_SIZE` is set, Affon mirrors the parsed value into
`UV_THREADPOOL_SIZE` before the runtime starts. Set it in the environment before
launching Affon:

```sh
RUNTIME_LIBUV_THREADPOOL_SIZE=8 ./zig-out/bin/affon script.ts
```

## Metal Pool Size Classes

Metal pool allocations are rounded into bucket sizes before reuse. Small
requests round to 4 KiB buckets, requests up to 1 MiB round to 64 KiB buckets,
requests up to 16 MiB round to 1 MiB buckets, and larger requests round to 16
MiB buckets.

`AFFON_METAL_POOL_SIZE_CLASSES` controls how many retained buffers each bucket
may keep. Entries are checked in order against the rounded bucket size:

```sh
AFFON_METAL_POOL_SIZE_CLASSES='65536:512,1048576:512,*:8'
```

This keeps up to 512 buffers per bucket for bucket sizes up to 64 KiB, up to
512 buffers per bucket for bucket sizes up to 1 MiB, and up to 8 buffers per
bucket for larger pooled buckets. Thresholds must be ascending, and `*` must be
the final catch-all entry. `AFFON_METAL_POOL_MAX_TOTAL_BYTES` remains the global
pool budget.

## Examples

When `AFFON_DEVICE` is unset, Affon detects a usable accelerator on startup:

- macOS selects Metal when the system Metal device is available.
- Other platforms select CUDA when the CUDA runtime and a device are available.
- CPU is the fallback when no supported accelerator is available.

Set `AFFON_DEVICE=cpu` to force CPU, or explicitly select `metal` or `cuda`.

For normal CLI commands, Affon prints the selected device, selection source, and
platform once to stderr at startup. `affon --version` remains machine-friendly
and prints only the version.

Run on Metal explicitly and keep tensor displays compact:

```sh
AFFON_DEVICE=metal AFFON_REPR_MAX_ITEMS=4 ./zig-out/bin/affon script.ts
```

Use the runtime telemetry console and `std:telemetry` for diagnostics, metrics,
and traces.

## CUDA selection

Use `AFFON_DEVICE=cuda` or `setDevice('cuda')` to place new tensors on CUDA.
Select a GPU with `AFFON_CUDA_DEVICE=N` before startup, or `setDevice('cuda:N')`
before the first CUDA allocation. The selected ordinal is fixed for the process.
Changing the ordinal after initialization is rejected.

Linux CUDA execution needs a working NVIDIA driver (`nvidia-smi`), plus
`libnvrtc.so.12` and `libcublas.so.12` on the dynamic loader path. CUDA is loaded
at runtime, so CPU execution does not require these libraries. Device-to-host
reads such as `item()` and `to_array()` synchronize the CUDA execution stream.
