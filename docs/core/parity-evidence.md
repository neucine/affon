# Full-catalog parity and performance evidence

This page records a full sweep of Affon's Cyna-native compute path against PyTorch. The sweep runs every case in the master capability catalog for Metal and every case in the CUDA mirror catalog.

## Protocol

- Affon branch `compute-cyna-routing` at `3f3974ce38c1d6625f3c986a0a1bb6ee9076a442`.
- Compute Core at `0bcc364d3b1d18334ebd1439b78d285ec3310b7f`.
- Hao at `42d57709117a7414547ef60808d9c6e31bf084e8`.
- Released Cyna 0.5.4.
- Affon binary SHA-256 `Metal f4ae82b3d17129a66fbeb0ce822c5723c81cea8517510b6be8a3c0f4a9c16660 / CUDA 11a51d4313a44ce7a22410013c90bd0662c79a4bca487ce3464c18257de17fcd`.
- Resumable sweep with three phases: capability probe, operator peer, Program peer.
- 10 iterations per sample, 5 samples, 3 rounds. Ratios are Affon elapsed time divided by PyTorch elapsed time, so lower is better. Timing covers a resident operation, a synchronized host materialization, and output allocation and disposal.

## Coverage

| Target | Cases | Numerical correct | Terminal unsupported | Operator passed | Program passed |
| --- | ---: | ---: | ---: | ---: | ---: |
| Metal (`Mac16,1/26.6.2/arm64`) | 11989 | 11950 | 39 | 11950 | 11950 |
| CUDA (`NVIDIA GeForce RTX 3090/7.0.0-43-generic/x86_64`) | 11946 | 11946 | 0 | 11946 | 11946 |

A case counts as covered only when the capability probe reaches a terminal status. Every numerical-correct case also has a passing operator peer and a passing Program peer with recorded timings. Unsupported cases are listed below and are never counted as passes.

## Performance versus PyTorch

| Target | Check | Timed cells | Geomean | Median | Maximum | Slowest cell |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Metal | operator | 11950 | 2.07× | 2.03× | 60.47× | `metal/f32/std_all/strided/r64_c127` |
| Metal | program | 11950 | 2.06× | 2.02× | 58.76× | `metal/f32/variance_all/strided/r64_c127` |
| CUDA | operator | 11946 | 4.24× | 3.92× | 820.87× | `cuda/bf16/topk/large/dense_axis0_r128_c257_k1_smallest` |
| CUDA | program | 11946 | 4.25× | 3.83× | 754.84× | `cuda/f32/topk/large/dense_axis0_r128_c257_k1_smallest` |

### Metal operator: slowest cells

| Case | Affon ms | PyTorch ms | Ratio |
| --- | ---: | ---: | ---: |
| `metal/f32/std_all/strided/r64_c127` | 14.6973 | 0.2431 | 60.47× |
| `metal/f32/variance_all/strided/r64_c127` | 14.5761 | 0.2415 | 60.36× |
| `metal/f32/variance_all/outer_axis` | 12.4936 | 0.2259 | 55.30× |
| `metal/f32/std_all/outer_axis` | 12.5338 | 0.2278 | 55.02× |
| `metal/f32/std_all/long_axis` | 5.1747 | 0.2094 | 24.71× |
| `metal/f32/variance_all/long_axis` | 5.1700 | 0.2119 | 24.40× |
| `metal/bf16/eq/offset/r129_c257` | 4.9062 | 0.3915 | 12.53× |
| `metal/bf16/lt/offset/r129_c257` | 4.9648 | 0.4060 | 12.23× |
| `metal/bf16/eq/strided/r129_c257` | 4.9659 | 0.4067 | 12.21× |
| `metal/bf16/lt/strided/r129_c257` | 4.9825 | 0.4089 | 12.19× |

### Metal program: slowest cells

| Case | Affon ms | PyTorch ms | Ratio |
| --- | ---: | ---: | ---: |
| `metal/f32/variance_all/strided/r64_c127` | 14.7274 | 0.2506 | 58.76× |
| `metal/f32/std_all/strided/r64_c127` | 14.6881 | 0.2501 | 58.73× |
| `metal/f32/std_all/outer_axis` | 12.5367 | 0.2399 | 52.27× |
| `metal/f32/variance_all/outer_axis` | 12.5161 | 0.2409 | 51.96× |
| `metal/f32/variance_all/long_axis` | 5.2020 | 0.2217 | 23.47× |
| `metal/f32/std_all/long_axis` | 5.2072 | 0.2270 | 22.94× |
| `metal/bf16/lt/offset/r129_c257` | 5.0477 | 0.4142 | 12.19× |
| `metal/bf16/eq/offset/r129_c257` | 5.0271 | 0.4182 | 12.02× |
| `metal/bf16/gt/strided/r129_c257` | 5.0617 | 0.4213 | 12.01× |
| `metal/bf16/gt/offset/r129_c257` | 5.0211 | 0.4189 | 11.99× |

### CUDA operator: slowest cells

| Case | Affon ms | PyTorch ms | Ratio |
| --- | ---: | ---: | ---: |
| `cuda/bf16/topk/large/dense_axis0_r128_c257_k1_smallest` | 30.1258 | 0.0367 | 820.87× |
| `cuda/bf16/topk/large/strided_axis0_r128_c257_k1_smallest` | 29.2396 | 0.0411 | 711.52× |
| `cuda/bf16/topk/large/offset_axis0_r128_c257_k1_smallest` | 29.8170 | 0.0426 | 700.48× |
| `cuda/f32/topk/large/dense_axis0_r128_c257_k1_smallest` | 30.8426 | 0.0456 | 676.16× |
| `cuda/f32/topk/large/offset_axis0_r128_c257_k1_smallest` | 31.1380 | 0.0477 | 653.03× |
| `cuda/f32/topk/large/strided_axis0_r128_c257_k1_smallest` | 31.2844 | 0.0487 | 642.11× |
| `cuda/f32/topk/batched/dense_axis1_b4_r33_c129_k16` | 48.0194 | 0.1642 | 292.36× |
| `cuda/bf16/topk/batched/dense_axis1_b4_r33_c129_k16` | 42.6290 | 0.1668 | 255.63× |
| `cuda/f32/topk/large/dense_axis0_r128_c65_k1_smallest` | 8.1303 | 0.0342 | 237.94× |
| `cuda/f32/topk/large/strided_axis0_r128_c65_k1_smallest` | 8.3965 | 0.0361 | 232.68× |

### CUDA program: slowest cells

| Case | Affon ms | PyTorch ms | Ratio |
| --- | ---: | ---: | ---: |
| `cuda/f32/topk/large/dense_axis0_r128_c257_k1_smallest` | 31.9213 | 0.0423 | 754.84× |
| `cuda/f32/topk/large/strided_axis0_r128_c257_k1_smallest` | 32.0058 | 0.0434 | 737.63× |
| `cuda/bf16/topk/large/dense_axis0_r128_c257_k1_smallest` | 29.5263 | 0.0404 | 731.29× |
| `cuda/bf16/topk/large/strided_axis0_r128_c257_k1_smallest` | 30.1879 | 0.0423 | 713.06× |
| `cuda/bf16/topk/large/offset_axis0_r128_c257_k1_smallest` | 29.4233 | 0.0419 | 702.19× |
| `cuda/f32/topk/large/offset_axis0_r128_c257_k1_smallest` | 30.8100 | 0.0459 | 671.41× |
| `cuda/f32/topk/batched/dense_axis1_b4_r33_c129_k16` | 46.4261 | 0.1593 | 291.47× |
| `cuda/bf16/topk/batched/dense_axis1_b4_r33_c129_k16` | 43.7081 | 0.1623 | 269.30× |
| `cuda/f32/topk/batched/dense_axis1_b2_r33_c129_k16` | 23.8250 | 0.1014 | 234.95× |
| `cuda/f32/topk/batched/dense_axis1_b4_r33_c65_k16` | 24.4846 | 0.1078 | 227.12× |

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

CUDA: none.

## Kernel reuse and the remaining gap

Both backends call the same accelerator kernels as the earlier Zig runtime. Metal reaches the Objective-C/Metal kernels through Compute Core's Metal backend. CUDA compiles the original CUDA kernel sources from `compute-core/src/backend/cuda/zig/` and calls them through the flat C adapter in `compute-core/src/backend/cuda/compat.c`.

Reuse fixed the largest CUDA regressions. BF16 prefill matmul execution fell from 45.6 ms to 1.96 ms (23× faster) and the full CUDA geomean fell from 5.06× to 4.24×. Two integration costs still dominate the reported ratios:

- Per-invocation overhead. A rank-2 view such as `squeeze` does no compute, yet takes 0.213 ms on Metal and 0.048 ms on CUDA in this harness, against 0.0089 ms and 0.007 ms in the earlier Zig report. That gap is Cyna Program invocation and output allocation, not a kernel.
- Output materialization. `Tensor.to_array()` builds the nested result through one JavaScript bridge call per element. For the 98,304-element 128×768 matmul output, BF16 prefill readback alone is 9.5 ms on CUDA and 6.5 ms on Metal. The Zig runtime built whole array rows in one call, which is why its total was far smaller.

Two compute gaps remain and are not kernel-reuse failures:

- CUDA axis-0 and batched top-k still run on the CPU. The Cyna lowering produces two separate single-output nodes (`top_k_values` and `top_k_indices`), so the CUDA backend's fused-pair GPU path is never entered. Axis-0 k=1 takes 30.1 ms versus PyTorch's 0.037 ms. The earlier Zig path took 0.030 ms.
- CUDA BF16 matmul still selects `matmul_small`. Execution is 1.96 ms against 0.027 ms in the earlier Zig report, which used `cuda_matmul_strided`. The shared kernels contain a faster GEMM path; the adapter needs to select it the way the Zig backend did.

## Raw evidence

- Metal sweep: `test/benchmark/results/parity-metal-full-2026-11-26-final2/`.
- CUDA sweep: `test/benchmark/results/parity-cuda-full-2026-11-26-final2/`.
- Each sweep directory keeps `capability.json`, `manifest.json`, per-cell numerical checkpoints under `numerical/`, and per-cell peer reports under `operator_peer/` and `program_peer/`.

## Scope notes

- The master catalog has 23,943 cases: 11,989 Metal, 11,948 CUDA, and 6 CPU. This sweep covers every Metal case and every case in the reviewed CUDA mirror.
- The 6 CPU cases are outside this GPU sweep and are not claimed here.
- The CUDA mirror omits `cuda/f32/reshape/view_only` and `cuda/bf16/reshape/view_only` because the mirror generator only copies Metal contracts present in the reviewed supported-contract ledger. Those two master entries are not claimed here.

