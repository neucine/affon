# CUDA top-k controlled before/after

Target: `NVIDIA GeForce RTX 3090/7.0.0-38-generic/x86_64`. Catalog: 520 cells. Protocol: 10 iterations, 5 samples, 3 rounds.

The PyTorch peer uses a full stable sort followed by narrow to preserve lower-index ties. Its timing is not native `torch.topk`. Ratios below compare Affon binaries on the same target and protocol. No regression baseline is accepted by this analysis.

| Tier | Dtype | Median execution ratio | Median whole-operation ratio | Faster execution cells | Regressions ≥25% and ≥0.01 ms |
| --- | --- | ---: | ---: | ---: | ---: |
| operator_peer | f32 | 0.256× | 0.654× | 252/260 | 0 execution / 0 total |
| operator_peer | bf16 | 1.007× | 1.004× | 35/260 | 0 execution / 0 total |
| program_peer | f32 | 0.306× | 0.679× | 254/260 | 0 execution / 0 total |
| program_peer | bf16 | 1.007× | 1.004× | 40/260 | 0 execution / 0 total |

The JSON retains all 520 cell comparisons in each tier, including small slowdowns. The earlier NaN-aware candidate and its 47 BF16 regression cells remain in `controlled-before-after-nan.json` and `reviewed-regressions-nan.json`.

## operator_peer largest execution gains

- `cuda/f32/topk/large/strided_axis1_r128_c257_k256_smallest`: 8.0854 → 0.0445 ms execution; 9.2433 → 1.2026 ms total.
- `cuda/f32/topk/large/dense_axis1_r128_c257_k256_smallest`: 8.0290 → 0.0443 ms execution; 9.1825 → 1.2054 ms total.
- `cuda/f32/topk/large/offset_axis1_r128_c257_k256_smallest`: 8.0263 → 0.0447 ms execution; 9.1879 → 1.2012 ms total.
- `cuda/f32/topk/large/strided_axis1_r64_c257_k256_smallest`: 4.8120 → 0.0371 ms execution; 5.3887 → 0.6192 ms total.
- `cuda/f32/topk/large/strided_axis1_r32_c257_k256_smallest`: 4.8060 → 0.0351 ms execution; 5.0987 → 0.3267 ms total.

Largest execution slowdown: `cuda/bf16/topk/large/strided_axis1_r8_c257_k256_smallest`, 0.002124 ms.

## program_peer largest execution gains

- `cuda/f32/topk/large/strided_axis1_r128_c257_k256_smallest`: 8.0912 → 0.0517 ms execution; 9.2492 → 1.2098 ms total.
- `cuda/f32/topk/large/dense_axis1_r128_c257_k256_smallest`: 8.0348 → 0.0516 ms execution; 9.1963 → 1.2154 ms total.
- `cuda/f32/topk/large/offset_axis1_r128_c257_k256_smallest`: 8.0283 → 0.0520 ms execution; 9.1844 → 1.2006 ms total.
- `cuda/f32/topk/large/strided_axis1_r32_c257_k256_smallest`: 4.8139 → 0.0409 ms execution; 5.1065 → 0.3374 ms total.
- `cuda/f32/topk/large/strided_axis1_r64_c257_k256_smallest`: 4.8160 → 0.0430 ms execution; 5.3884 → 0.6223 ms total.

Largest execution slowdown: `cuda/bf16/topk/large/offset_axis1_r128_c257_k256_smallest`, 0.002149 ms.

