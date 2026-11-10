# CUDA top-k controlled before/after

Target: `NVIDIA GeForce RTX 3090/7.0.0-38-generic/x86_64`. Catalog: 520 cells. Protocol: 10 iterations, 5 samples, 3 rounds.

The PyTorch peer uses a full stable sort followed by narrow to preserve lower-index ties. Its timing is not native `torch.topk`. Ratios here compare Affon binaries on the same target and protocol. No regression baseline is accepted by this analysis.

| Tier | Dtype | Median execution ratio | Median whole-operation ratio | Faster execution cells | Material execution / total regressions |
| --- | --- | ---: | ---: | ---: | ---: |
| operator_peer | f32 | 0.255× | 0.653× | 259/260 | 0 / 0 |
| operator_peer | bf16 | 1.008× | 1.004× | 53/260 | 0 / 0 |
| program_peer | f32 | 0.305× | 0.673× | 255/260 | 0 / 0 |
| program_peer | bf16 | 1.008× | 1.005× | 36/260 | 0 / 0 |

Material regression threshold: at least 5% and 0.05 ms slower. Every cell is in `controlled-before-after-nan.json`.

## operator_peer largest execution gains

- `cuda/f32/topk/large/strided_axis1_r128_c257_k256_smallest`: 8.0854 → 0.0445 ms execution; 9.2433 → 1.1876 ms total.
- `cuda/f32/topk/large/dense_axis1_r128_c257_k256_smallest`: 8.0290 → 0.0446 ms execution; 9.1825 → 1.1988 ms total.
- `cuda/f32/topk/large/offset_axis1_r128_c257_k256_smallest`: 8.0263 → 0.0446 ms execution; 9.1879 → 1.1965 ms total.
- `cuda/f32/topk/large/strided_axis1_r64_c257_k256_smallest`: 4.8120 → 0.0370 ms execution; 5.3887 → 0.6223 ms total.
- `cuda/f32/topk/large/strided_axis1_r32_c257_k256_smallest`: 4.8060 → 0.0351 ms execution; 5.0987 → 0.3290 ms total.

Largest execution slowdown: `cuda/bf16/topk/large/dense_axis1_r8_c31_k30_smallest`, 0.036689 ms.

## program_peer largest execution gains

- `cuda/f32/topk/large/strided_axis1_r128_c257_k256_smallest`: 8.0912 → 0.0519 ms execution; 9.2492 → 1.2121 ms total.
- `cuda/f32/topk/large/dense_axis1_r128_c257_k256_smallest`: 8.0348 → 0.0518 ms execution; 9.1963 → 1.2107 ms total.
- `cuda/f32/topk/large/offset_axis1_r128_c257_k256_smallest`: 8.0283 → 0.0516 ms execution; 9.1844 → 1.2094 ms total.
- `cuda/f32/topk/large/strided_axis1_r32_c257_k256_smallest`: 4.8139 → 0.0405 ms execution; 5.1065 → 0.3352 ms total.
- `cuda/f32/topk/large/strided_axis1_r64_c257_k256_smallest`: 4.8160 → 0.0429 ms execution; 5.3884 → 0.6211 ms total.

Largest execution slowdown: `cuda/bf16/topk/large/dense_axis1_r8_c31_k30_smallest`, 0.036744 ms.

