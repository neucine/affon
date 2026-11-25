# Full-catalog parity and performance evidence

This page records a full sweep of Affon's Cyna-native compute path against PyTorch. The sweep runs every case in the master capability catalog for Metal and every case in the CUDA mirror catalog.

## Protocol

- Affon branch `compute-cyna-routing` at `ce2185cce`.
- Compute Core at `da314e7`.
- Hao at `42d5770`.
- Released Cyna 0.5.4.
- Affon binary SHA-256 `Metal 8dc2bba6… / CUDA c18f1d54…`.
- Resumable sweep with three phases: capability probe, operator peer, Program peer.
- 10 iterations per sample, 5 samples, 3 rounds. Ratios are Affon elapsed time divided by PyTorch elapsed time, so lower is better. Timing covers a resident operation, a synchronized host materialization, and output allocation and disposal.

## Coverage

| Target | Cases | Numerical correct | Terminal unsupported | Operator passed | Program passed |
| --- | ---: | ---: | ---: | ---: | ---: |
| Metal (`Mac16,1/26.6.2/arm64`) | 11989 | 11950 | 39 | 11950 | 11950 |
| CUDA (`NVIDIA GeForce RTX 3090/7.0.0-43-generic/x86_64`) | 11946 | 11945 | 1 | 11945 | 11945 |

A case counts as covered only when the capability probe reaches a terminal status. Every numerical-correct case also has a passing operator peer and a passing Program peer with recorded timings. Unsupported cases are listed below and are never counted as passes.

## Performance versus PyTorch

| Target | Check | Timed cells | Geomean | Median | Maximum | Slowest cell |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Metal | operator | 11950 | 2.03× | 2.05× | 63.77× | `metal/f32/std_all/strided/r64_c127` |
| Metal | program | 11950 | 2.07× | 2.06× | 57.59× | `metal/f32/std_all/strided/r64_c127` |
| CUDA | operator | 11945 | 5.06× | 4.87× | 710.00× | `cuda/bf16/topk/large/dense_axis0_r128_c257_k1_smallest` |
| CUDA | program | 11945 | 5.26× | 5.19× | 728.45× | `cuda/f32/topk/large/dense_axis0_r128_c257_k1_smallest` |

### Metal operator: slowest cells

| Case | Affon ms | PyTorch ms | Ratio |
| --- | ---: | ---: | ---: |
| `metal/f32/std_all/strided/r64_c127` | 14.1426 | 0.2218 | 63.77× |
| `metal/f32/variance_all/strided/r64_c127` | 14.1356 | 0.2279 | 62.02× |
| `metal/f32/variance_all/outer_axis` | 13.2621 | 0.2417 | 54.87× |
| `metal/f32/std_all/outer_axis` | 13.2464 | 0.2487 | 53.25× |
| `metal/f32/std_all/long_axis` | 5.6736 | 0.2086 | 27.19× |
| `metal/f32/variance_all/long_axis` | 5.6448 | 0.2224 | 25.39× |
| `metal/bf16/eq/offset/r129_c257` | 4.9361 | 0.4056 | 12.17× |
| `metal/bf16/gt/offset/r129_c257` | 4.9252 | 0.4072 | 12.10× |
| `metal/bf16/lt/strided/r129_c257` | 4.9875 | 0.4126 | 12.09× |
| `metal/bf16/eq/strided/r129_c257` | 4.9701 | 0.4121 | 12.06× |

### Metal program: slowest cells

| Case | Affon ms | PyTorch ms | Ratio |
| --- | ---: | ---: | ---: |
| `metal/f32/std_all/strided/r64_c127` | 13.2576 | 0.2302 | 57.59× |
| `metal/f32/variance_all/strided/r64_c127` | 13.3060 | 0.2326 | 57.20× |
| `metal/f32/variance_all/outer_axis` | 12.4327 | 0.2274 | 54.68× |
| `metal/f32/std_all/outer_axis` | 12.4277 | 0.2291 | 54.24× |
| `metal/f32/std_all/long_axis` | 5.1160 | 0.2181 | 23.46× |
| `metal/f32/variance_all/long_axis` | 5.1163 | 0.2208 | 23.17× |
| `metal/bf16/gt/strided/r129_c257` | 5.0744 | 0.4151 | 12.23× |
| `metal/bf16/eq/offset/r129_c257` | 5.0851 | 0.4185 | 12.15× |
| `metal/bf16/lt/strided/r129_c257` | 5.0456 | 0.4173 | 12.09× |
| `metal/bf16/lt/offset/r129_c257` | 5.0562 | 0.4194 | 12.05× |

### CUDA operator: slowest cells

| Case | Affon ms | PyTorch ms | Ratio |
| --- | ---: | ---: | ---: |
| `cuda/bf16/topk/large/dense_axis0_r128_c257_k1_smallest` | 26.5019 | 0.0373 | 710.00× |
| `cuda/bf16/topk/large/strided_axis0_r128_c257_k1_smallest` | 26.9527 | 0.0394 | 684.42× |
| `cuda/bf16/topk/large/offset_axis0_r128_c257_k1_smallest` | 26.0148 | 0.0398 | 652.89× |
| `cuda/f32/topk/large/dense_axis0_r128_c257_k1_smallest` | 29.6228 | 0.0456 | 650.26× |
| `cuda/f32/topk/large/strided_axis0_r128_c257_k1_smallest` | 29.7964 | 0.0476 | 626.51× |
| `cuda/f32/topk/large/offset_axis0_r128_c257_k1_smallest` | 29.2509 | 0.0487 | 601.25× |
| `cuda/f32/topk/batched/dense_axis1_b4_r33_c129_k16` | 48.4821 | 0.1614 | 300.33× |
| `cuda/bf16/topk/batched/dense_axis1_b4_r33_c129_k16` | 41.7789 | 0.1642 | 254.49× |
| `cuda/f32/topk/batched/dense_axis1_b2_r33_c129_k16` | 23.7006 | 0.1021 | 232.16× |
| `cuda/f32/topk/batched/dense_axis1_b4_r33_c65_k16` | 24.3550 | 0.1073 | 226.92× |

### CUDA program: slowest cells

| Case | Affon ms | PyTorch ms | Ratio |
| --- | ---: | ---: | ---: |
| `cuda/f32/topk/large/dense_axis0_r128_c257_k1_smallest` | 29.8002 | 0.0409 | 728.45× |
| `cuda/f32/topk/large/strided_axis0_r128_c257_k1_smallest` | 29.8253 | 0.0434 | 687.02× |
| `cuda/bf16/topk/large/dense_axis0_r128_c257_k1_smallest` | 26.8523 | 0.0407 | 659.57× |
| `cuda/f32/topk/large/offset_axis0_r128_c257_k1_smallest` | 29.5894 | 0.0463 | 639.43× |
| `cuda/bf16/topk/large/strided_axis0_r128_c257_k1_smallest` | 26.9391 | 0.0427 | 630.27× |
| `cuda/bf16/topk/large/offset_axis0_r128_c257_k1_smallest` | 26.2546 | 0.0436 | 601.70× |
| `cuda/f32/topk/batched/dense_axis1_b4_r33_c129_k16` | 47.7033 | 0.1646 | 289.89× |
| `cuda/bf16/topk/batched/dense_axis1_b4_r33_c129_k16` | 42.9784 | 0.1568 | 274.17× |
| `cuda/f32/topk/batched/dense_axis1_b2_r33_c129_k16` | 24.1809 | 0.1031 | 234.58× |
| `cuda/f32/topk/batched/dense_axis1_b4_r33_c65_k16` | 24.5546 | 0.1108 | 221.60× |

## Unsupported cases

### Metal

| Case | Probe status | First error |
| --- | --- | --- |
| `metal/f32/sum_all/scalar` | probe_failed | Program lowering failed: InvalidShape |
| `metal/f32/mean_all/scalar` | probe_failed | Program lowering failed: InvalidShape |
| `metal/f32/min_all/scalar` | probe_failed | Program lowering failed: InvalidShape |
| `metal/f32/max_all/scalar` | probe_failed | Program lowering failed: InvalidShape |
| `metal/f32/variance_all/scalar` | probe_failed | Program lowering failed: InvalidShape |
| `metal/f32/std_all/scalar` | probe_failed | Program lowering failed: InvalidShape |
| `metal/bf16/sum_all/scalar` | probe_failed | Program lowering failed: InvalidShape |
| `metal/bf16/mean_all/scalar` | probe_failed | Program lowering failed: InvalidShape |
| `metal/bf16/min_all/scalar` | probe_failed | Program lowering failed: InvalidShape |
| `metal/bf16/max_all/scalar` | probe_failed | Program lowering failed: InvalidShape |
| `metal/bf16/variance_all/scalar` | probe_failed | Program lowering failed: InvalidShape |
| `metal/bf16/std_all/scalar` | probe_failed | Program lowering failed: InvalidShape |
| `metal/f32/embedding/invalid_index` | probe_failed | Program execution failed: InvalidCommand (BackendFailure) |
| `metal/f32/gather/invalid_index` | probe_failed | Program execution failed: InvalidCommand (BackendFailure) |
| `metal/f32/index_select/invalid_index` | probe_failed | Program execution failed: InvalidCommand (BackendFailure) |
| `metal/f32/argmin_all/scalar` | probe_failed | Program lowering failed: InvalidShape |
| `metal/f32/argmax_all/scalar` | probe_failed | Program lowering failed: InvalidShape |
| `metal/f32/sum_axis/scalar` | probe_failed | sum axis is out of range |
| `metal/f32/mean_axis/scalar` | probe_failed | mean axis is out of range |
| `metal/f32/min_axis/scalar` | probe_failed | min axis is out of range |
| `metal/f32/max_axis/scalar` | probe_failed | max axis is out of range |
| `metal/f32/variance_axis/scalar` | probe_failed | variance axis is out of range |
| `metal/f32/std_axis/scalar` | probe_failed | std axis is out of range |
| `metal/f32/argmin_axis/scalar` | probe_failed | argmin axis is out of range |
| `metal/f32/argmax_axis/scalar` | probe_failed | argmax axis is out of range |
| `metal/bf16/embedding/invalid_index` | probe_failed | Program execution failed: InvalidCommand (BackendFailure) |
| `metal/bf16/gather/invalid_index` | probe_failed | Program execution failed: InvalidCommand (BackendFailure) |
| `metal/bf16/index_select/invalid_index` | probe_failed | Program execution failed: InvalidCommand (BackendFailure) |
| `metal/bf16/argmin_all/scalar` | probe_failed | Program lowering failed: InvalidShape |
| `metal/bf16/argmax_all/scalar` | probe_failed | Program lowering failed: InvalidShape |
| `metal/bf16/sum_axis/scalar` | probe_failed | sum axis is out of range |
| `metal/bf16/mean_axis/scalar` | probe_failed | mean axis is out of range |
| `metal/bf16/min_axis/scalar` | probe_failed | min axis is out of range |
| `metal/bf16/max_axis/scalar` | probe_failed | max axis is out of range |
| `metal/bf16/variance_axis/scalar` | probe_failed | variance axis is out of range |
| `metal/bf16/std_axis/scalar` | probe_failed | std axis is out of range |
| `metal/bf16/argmin_axis/scalar` | probe_failed | argmin axis is out of range |
| `metal/bf16/argmax_axis/scalar` | probe_failed | argmax axis is out of range |
| `metal/bf16/scatter_add/invalid_index` | probe_failed | Program execution failed: InvalidCommand (BackendFailure) |

### CUDA

| Case | Probe status | First error |
| --- | --- | --- |
| `cuda/f32/scatter_add/invalid_index` | probe_failed | Program execution failed: InvalidCommand (InvalidValue) |

## Raw evidence

- Metal sweep: `test/benchmark/results/parity-metal-full-2026-11-25-final3/`.
- CUDA sweep: `test/benchmark/results/parity-cuda-full-2026-11-25-final3/`.
- Each sweep directory keeps `capability.json`, `manifest.json`, per-cell numerical checkpoints under `numerical/`, and per-cell peer reports under `operator_peer/` and `program_peer/`.

## Scope notes

- The master catalog has 23,943 cases: 11,989 Metal, 11,948 CUDA, and 6 CPU. This sweep covers every Metal case and every case in the reviewed CUDA mirror.
- The 6 CPU cases are outside this GPU sweep and are not claimed here.
- The CUDA mirror omits `cuda/f32/reshape/view_only` and `cuda/bf16/reshape/view_only` because the mirror generator only copies Metal contracts present in the reviewed supported-contract ledger. Those two master entries are not claimed here.

