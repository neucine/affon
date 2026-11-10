# CUDA f32 unary dispatch

The reviewed Compute change routes f32 unary operations through the existing unary kernel that reads strided inputs directly. For rank-two inputs this removes a separate contiguous-copy launch. The source change is in the sibling Compute checkout's `src/backend/cuda/device.zig`.

The complete 519-cell catalog covers 13 unary operations across scalar, dense, offset, ragged, and strided cases on an RTX 3090. Baseline and candidate used 10 iterations, 5 samples, and 3 rounds. Both complete audits passed all 519 numerical, operator, and Program checks, bound to the exact catalog and PyTorch 2.11.0+cu128. The candidate binary SHA-256 is `de9f587b9528b946e0e3f046708cf7c80f0590b84574ce3e9be84c03d6c2ff57`. The regression comparison found no changed case definitions or correctness failures.

Candidate Affon total medians improved in 504 of 519 operator cells and 502 Program cells. Median candidate/baseline ratios were 0.9081 operator and 0.9209 Program. Every operation improved at the median. The largest Affon slowdown was 3.4% operator and 2.6% Program. One diagnostic timing suspect was caused by a PyTorch peer shift; Affon improved 7.7% operator and 7.3% Program in that cell. These are per-call measurements that include output materialization.

`unary-f32-full-baseline/` and `unary-f32-full-candidate/` retain the complete raw reports. The audits, regression output, logs, catalog, and build log are alongside them. A 26-cell probe preceded the full sweep. Prior aggregate entries are retained in `../archive-cuda-unary-before-f32-one-launch-2026-11-07/`.
