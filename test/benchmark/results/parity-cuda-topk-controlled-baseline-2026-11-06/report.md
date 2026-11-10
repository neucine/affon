# Parity batch report

Target: `NVIDIA GeForce RTX 3090/7.0.0-38-generic/x86_64`  
Catalog SHA-256: `95fd5fcca4989135d6805365714bb135d1822d09a976b741b03f3317fb40de4e`  
Binary SHA-256: `d352a015ea131602d664b217ee6a7e1f5084c2785fb4ee0c02ee8ec8c87be1ad`  
Timing: 10 iterations × 5 samples × 3 rounds

## Checks

- Cases run: 520
- Numerical check passed: 520
- Operator check and timing passed: 32
- Program check and timing passed: 0
- Numerical failures: 0
- Operator failures: 0
- Program failures: 0
- Operator checks pending: 488
- Program checks pending: 520
- Checkpoint updated: 2026-11-06T11:05:08.081146+00:00
- Validation is in progress; pending checks have not passed.

## Cases

| Case | Family | Shape | Layout | Numerical | Operator | Affon / PyTorch ms | Program | Affon / PyTorch ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `cuda/bf16/topk/batched` | indexing | 3 × 11 × 7 | dense | correct | [passed](operator_peer/cuda__bf16__topk__batched.json) | 0.04536 / 0.04255 | not_run | — |
| `cuda/bf16/topk/batched/dense_axis1_b2_r16_c129_k8` | indexing | 2 × 16 × 129 | dense | correct | [passed](operator_peer/cuda__bf16__topk__batched__dense_axis1_b2_r16_c129_k8.json) | 0.1249 / 0.08566 | not_run | — |
| `cuda/bf16/topk/batched/dense_axis1_b2_r16_c17_k8` | indexing | 2 × 16 × 17 | dense | correct | [passed](operator_peer/cuda__bf16__topk__batched__dense_axis1_b2_r16_c17_k8.json) | 0.06068 / 0.04783 | not_run | — |
| `cuda/bf16/topk/batched/dense_axis1_b2_r16_c5_k8` | indexing | 2 × 16 × 5 | dense | correct | [passed](operator_peer/cuda__bf16__topk__batched__dense_axis1_b2_r16_c5_k8.json) | 0.04795 / 0.0421 | not_run | — |
| `cuda/bf16/topk/batched/dense_axis1_b2_r16_c65_k8` | indexing | 2 × 16 × 65 | dense | correct | [passed](operator_peer/cuda__bf16__topk__batched__dense_axis1_b2_r16_c65_k8.json) | 0.09129 / 0.06932 | not_run | — |
| `cuda/bf16/topk/batched/dense_axis1_b2_r33_c129_k16` | indexing | 2 × 33 × 129 | dense | correct | [passed](operator_peer/cuda__bf16__topk__batched__dense_axis1_b2_r33_c129_k16.json) | 0.2148 / 0.1006 | not_run | — |
| `cuda/bf16/topk/batched/dense_axis1_b2_r33_c17_k16` | indexing | 2 × 33 × 17 | dense | correct | [passed](operator_peer/cuda__bf16__topk__batched__dense_axis1_b2_r33_c17_k16.json) | 0.08839 / 0.05874 | not_run | — |
| `cuda/bf16/topk/batched/dense_axis1_b2_r33_c5_k16` | indexing | 2 × 33 × 5 | dense | correct | [passed](operator_peer/cuda__bf16__topk__batched__dense_axis1_b2_r33_c5_k16.json) | 0.06888 / 0.04507 | not_run | — |
| `cuda/bf16/topk/batched/dense_axis1_b2_r33_c65_k16` | indexing | 2 × 33 × 65 | dense | correct | [passed](operator_peer/cuda__bf16__topk__batched__dense_axis1_b2_r33_c65_k16.json) | 0.149 / 0.08818 | not_run | — |
| `cuda/bf16/topk/batched/dense_axis1_b2_r3_c129_k1` | indexing | 2 × 3 × 129 | dense | correct | [passed](operator_peer/cuda__bf16__topk__batched__dense_axis1_b2_r3_c129_k1.json) | 0.03835 / 0.04651 | not_run | — |
| `cuda/bf16/topk/batched/dense_axis1_b2_r3_c17_k1` | indexing | 2 × 3 × 17 | dense | correct | [passed](operator_peer/cuda__bf16__topk__batched__dense_axis1_b2_r3_c17_k1.json) | 0.0299 / 0.04029 | not_run | — |
| `cuda/bf16/topk/batched/dense_axis1_b2_r3_c5_k1` | indexing | 2 × 3 × 5 | dense | correct | [passed](operator_peer/cuda__bf16__topk__batched__dense_axis1_b2_r3_c5_k1.json) | 0.02847 / 0.03984 | not_run | — |
| `cuda/bf16/topk/batched/dense_axis1_b2_r3_c65_k1` | indexing | 2 × 3 × 65 | dense | correct | [passed](operator_peer/cuda__bf16__topk__batched__dense_axis1_b2_r3_c65_k1.json) | 0.03352 / 0.04252 | not_run | — |
| `cuda/bf16/topk/batched/dense_axis1_b2_r7_c129_k3` | indexing | 2 × 7 × 129 | dense | correct | [passed](operator_peer/cuda__bf16__topk__batched__dense_axis1_b2_r7_c129_k3.json) | 0.06357 / 0.06193 | not_run | — |
| `cuda/bf16/topk/batched/dense_axis1_b2_r7_c17_k3` | indexing | 2 × 7 × 17 | dense | correct | [passed](operator_peer/cuda__bf16__topk__batched__dense_axis1_b2_r7_c17_k3.json) | 0.03767 / 0.04305 | not_run | — |
| `cuda/bf16/topk/batched/dense_axis1_b2_r7_c5_k3` | indexing | 2 × 7 × 5 | dense | correct | [passed](operator_peer/cuda__bf16__topk__batched__dense_axis1_b2_r7_c5_k3.json) | 0.03401 / 0.04094 | not_run | — |
| `cuda/bf16/topk/batched/dense_axis1_b2_r7_c65_k3` | indexing | 2 × 7 × 65 | dense | correct | [passed](operator_peer/cuda__bf16__topk__batched__dense_axis1_b2_r7_c65_k3.json) | 0.04941 / 0.05152 | not_run | — |
| `cuda/bf16/topk/batched/dense_axis1_b4_r16_c129_k8` | indexing | 4 × 16 × 129 | dense | correct | [passed](operator_peer/cuda__bf16__topk__batched__dense_axis1_b4_r16_c129_k8.json) | 0.2067 / 0.1017 | not_run | — |
| `cuda/bf16/topk/batched/dense_axis1_b4_r16_c17_k8` | indexing | 4 × 16 × 17 | dense | correct | [passed](operator_peer/cuda__bf16__topk__batched__dense_axis1_b4_r16_c17_k8.json) | 0.08273 / 0.05875 | not_run | — |
| `cuda/bf16/topk/batched/dense_axis1_b4_r16_c5_k8` | indexing | 4 × 16 × 5 | dense | correct | [passed](operator_peer/cuda__bf16__topk__batched__dense_axis1_b4_r16_c5_k8.json) | 0.06069 / 0.04456 | not_run | — |
| `cuda/bf16/topk/batched/dense_axis1_b4_r16_c65_k8` | indexing | 4 × 16 × 65 | dense | correct | [passed](operator_peer/cuda__bf16__topk__batched__dense_axis1_b4_r16_c65_k8.json) | 0.1415 / 0.08843 | not_run | — |
| `cuda/bf16/topk/batched/dense_axis1_b4_r33_c129_k16` | indexing | 4 × 33 × 129 | dense | correct | [passed](operator_peer/cuda__bf16__topk__batched__dense_axis1_b4_r33_c129_k16.json) | 0.3931 / 0.1526 | not_run | — |
| `cuda/bf16/topk/batched/dense_axis1_b4_r33_c17_k16` | indexing | 4 × 33 × 17 | dense | correct | [passed](operator_peer/cuda__bf16__topk__batched__dense_axis1_b4_r33_c17_k16.json) | 0.1277 / 0.07094 | not_run | — |
| `cuda/bf16/topk/batched/dense_axis1_b4_r33_c5_k16` | indexing | 4 × 33 × 5 | dense | correct | [passed](operator_peer/cuda__bf16__topk__batched__dense_axis1_b4_r33_c5_k16.json) | 0.09331 / 0.04876 | not_run | — |
| `cuda/bf16/topk/batched/dense_axis1_b4_r33_c65_k16` | indexing | 4 × 33 × 65 | dense | correct | [passed](operator_peer/cuda__bf16__topk__batched__dense_axis1_b4_r33_c65_k16.json) | 0.2379 / 0.1032 | not_run | — |
| `cuda/bf16/topk/batched/dense_axis1_b4_r3_c129_k1` | indexing | 4 × 3 × 129 | dense | correct | [passed](operator_peer/cuda__bf16__topk__batched__dense_axis1_b4_r3_c129_k1.json) | 0.04919 / 0.05732 | not_run | — |
| `cuda/bf16/topk/batched/dense_axis1_b4_r3_c17_k1` | indexing | 4 × 3 × 17 | dense | correct | [passed](operator_peer/cuda__bf16__topk__batched__dense_axis1_b4_r3_c17_k1.json) | 0.03336 / 0.04149 | not_run | — |
| `cuda/bf16/topk/batched/dense_axis1_b4_r3_c5_k1` | indexing | 4 × 3 × 5 | dense | correct | [passed](operator_peer/cuda__bf16__topk__batched__dense_axis1_b4_r3_c5_k1.json) | 0.03039 / 0.04069 | not_run | — |
| `cuda/bf16/topk/batched/dense_axis1_b4_r3_c65_k1` | indexing | 4 × 3 × 65 | dense | correct | [passed](operator_peer/cuda__bf16__topk__batched__dense_axis1_b4_r3_c65_k1.json) | 0.04041 / 0.04635 | not_run | — |
| `cuda/bf16/topk/batched/dense_axis1_b4_r7_c129_k3` | indexing | 4 × 7 × 129 | dense | correct | [passed](operator_peer/cuda__bf16__topk__batched__dense_axis1_b4_r7_c129_k3.json) | 0.09598 / 0.0679 | not_run | — |
| `cuda/bf16/topk/batched/dense_axis1_b4_r7_c17_k3` | indexing | 4 × 7 × 17 | dense | correct | [passed](operator_peer/cuda__bf16__topk__batched__dense_axis1_b4_r7_c17_k3.json) | 0.04613 / 0.04553 | not_run | — |
| `cuda/bf16/topk/batched/dense_axis1_b4_r7_c5_k3` | indexing | 4 × 7 × 5 | dense | correct | [passed](operator_peer/cuda__bf16__topk__batched__dense_axis1_b4_r7_c5_k3.json) | 0.03805 / 0.0415 | not_run | — |
| `cuda/bf16/topk/batched/dense_axis1_b4_r7_c65_k3` | indexing | 4 × 7 × 65 | dense | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/batched/dense_axis2_b2_r16_c129_k64` | indexing | 2 × 16 × 129 | dense | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/batched/dense_axis2_b2_r16_c17_k8` | indexing | 2 × 16 × 17 | dense | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/batched/dense_axis2_b2_r16_c5_k2` | indexing | 2 × 16 × 5 | dense | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/batched/dense_axis2_b2_r16_c65_k32` | indexing | 2 × 16 × 65 | dense | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/batched/dense_axis2_b2_r33_c129_k64` | indexing | 2 × 33 × 129 | dense | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/batched/dense_axis2_b2_r33_c17_k8` | indexing | 2 × 33 × 17 | dense | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/batched/dense_axis2_b2_r33_c5_k2` | indexing | 2 × 33 × 5 | dense | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/batched/dense_axis2_b2_r33_c65_k32` | indexing | 2 × 33 × 65 | dense | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/batched/dense_axis2_b2_r3_c129_k64` | indexing | 2 × 3 × 129 | dense | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/batched/dense_axis2_b2_r3_c17_k8` | indexing | 2 × 3 × 17 | dense | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/batched/dense_axis2_b2_r3_c5_k2` | indexing | 2 × 3 × 5 | dense | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/batched/dense_axis2_b2_r3_c65_k32` | indexing | 2 × 3 × 65 | dense | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/batched/dense_axis2_b2_r7_c129_k64` | indexing | 2 × 7 × 129 | dense | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/batched/dense_axis2_b2_r7_c17_k8` | indexing | 2 × 7 × 17 | dense | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/batched/dense_axis2_b2_r7_c5_k2` | indexing | 2 × 7 × 5 | dense | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/batched/dense_axis2_b2_r7_c65_k32` | indexing | 2 × 7 × 65 | dense | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/batched/dense_axis2_b4_r16_c129_k64` | indexing | 4 × 16 × 129 | dense | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/batched/dense_axis2_b4_r16_c17_k8` | indexing | 4 × 16 × 17 | dense | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/batched/dense_axis2_b4_r16_c5_k2` | indexing | 4 × 16 × 5 | dense | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/batched/dense_axis2_b4_r16_c65_k32` | indexing | 4 × 16 × 65 | dense | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/batched/dense_axis2_b4_r33_c129_k64` | indexing | 4 × 33 × 129 | dense | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/batched/dense_axis2_b4_r33_c17_k8` | indexing | 4 × 33 × 17 | dense | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/batched/dense_axis2_b4_r33_c5_k2` | indexing | 4 × 33 × 5 | dense | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/batched/dense_axis2_b4_r33_c65_k32` | indexing | 4 × 33 × 65 | dense | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/batched/dense_axis2_b4_r3_c129_k64` | indexing | 4 × 3 × 129 | dense | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/batched/dense_axis2_b4_r3_c17_k8` | indexing | 4 × 3 × 17 | dense | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/batched/dense_axis2_b4_r3_c5_k2` | indexing | 4 × 3 × 5 | dense | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/batched/dense_axis2_b4_r3_c65_k32` | indexing | 4 × 3 × 65 | dense | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/batched/dense_axis2_b4_r7_c129_k64` | indexing | 4 × 7 × 129 | dense | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/batched/dense_axis2_b4_r7_c17_k8` | indexing | 4 × 7 × 17 | dense | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/batched/dense_axis2_b4_r7_c5_k2` | indexing | 4 × 7 × 5 | dense | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/batched/dense_axis2_b4_r7_c65_k32` | indexing | 4 × 7 × 65 | dense | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/large` | indexing | 8 × 7 | dense | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/large/dense_axis0_r128_c257_k1_smallest` | indexing | 128 × 257 | dense | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/large/dense_axis0_r128_c31_k1_smallest` | indexing | 128 × 31 | dense | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/large/dense_axis0_r128_c65_k1_smallest` | indexing | 128 × 65 | dense | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/large/dense_axis0_r128_c7_k1_smallest` | indexing | 128 × 7 | dense | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/large/dense_axis0_r32_c257_k1_smallest` | indexing | 32 × 257 | dense | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/large/dense_axis0_r32_c31_k1_smallest` | indexing | 32 × 31 | dense | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/large/dense_axis0_r32_c65_k1_smallest` | indexing | 32 × 65 | dense | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/large/dense_axis0_r32_c7_k1_smallest` | indexing | 32 × 7 | dense | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/large/dense_axis0_r64_c257_k1_smallest` | indexing | 64 × 257 | dense | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/large/dense_axis0_r64_c31_k1_smallest` | indexing | 64 × 31 | dense | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/large/dense_axis0_r64_c65_k1_smallest` | indexing | 64 × 65 | dense | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/large/dense_axis0_r64_c7_k1_smallest` | indexing | 64 × 7 | dense | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/large/dense_axis0_r8_c257_k1_smallest` | indexing | 8 × 257 | dense | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/large/dense_axis0_r8_c31_k1_smallest` | indexing | 8 × 31 | dense | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/large/dense_axis0_r8_c65_k1_smallest` | indexing | 8 × 65 | dense | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/large/dense_axis0_r8_c7_k1_smallest` | indexing | 8 × 7 | dense | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/large/dense_axis1_r128_c257_k256_smallest` | indexing | 128 × 257 | dense | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/large/dense_axis1_r128_c31_k30_smallest` | indexing | 128 × 31 | dense | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/large/dense_axis1_r128_c65_k64_smallest` | indexing | 128 × 65 | dense | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/large/dense_axis1_r128_c7_k6_smallest` | indexing | 128 × 7 | dense | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/large/dense_axis1_r32_c257_k256_smallest` | indexing | 32 × 257 | dense | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/large/dense_axis1_r32_c31_k30_smallest` | indexing | 32 × 31 | dense | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/large/dense_axis1_r32_c65_k64_smallest` | indexing | 32 × 65 | dense | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/large/dense_axis1_r32_c7_k6_smallest` | indexing | 32 × 7 | dense | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/large/dense_axis1_r64_c257_k256_smallest` | indexing | 64 × 257 | dense | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/large/dense_axis1_r64_c31_k30_smallest` | indexing | 64 × 31 | dense | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/large/dense_axis1_r64_c65_k64_smallest` | indexing | 64 × 65 | dense | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/large/dense_axis1_r64_c7_k6_smallest` | indexing | 64 × 7 | dense | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/large/dense_axis1_r8_c257_k256_smallest` | indexing | 8 × 257 | dense | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/large/dense_axis1_r8_c31_k30_smallest` | indexing | 8 × 31 | dense | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/large/dense_axis1_r8_c65_k64_smallest` | indexing | 8 × 65 | dense | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/large/dense_axis1_r8_c7_k6_smallest` | indexing | 8 × 7 | dense | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/large/offset_axis0_r128_c257_k1_smallest` | indexing | 128 × 257 | offset | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/large/offset_axis0_r128_c31_k1_smallest` | indexing | 128 × 31 | offset | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/large/offset_axis0_r128_c65_k1_smallest` | indexing | 128 × 65 | offset | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/large/offset_axis0_r128_c7_k1_smallest` | indexing | 128 × 7 | offset | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/large/offset_axis0_r32_c257_k1_smallest` | indexing | 32 × 257 | offset | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/large/offset_axis0_r32_c31_k1_smallest` | indexing | 32 × 31 | offset | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/large/offset_axis0_r32_c65_k1_smallest` | indexing | 32 × 65 | offset | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/large/offset_axis0_r32_c7_k1_smallest` | indexing | 32 × 7 | offset | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/large/offset_axis0_r64_c257_k1_smallest` | indexing | 64 × 257 | offset | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/large/offset_axis0_r64_c31_k1_smallest` | indexing | 64 × 31 | offset | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/large/offset_axis0_r64_c65_k1_smallest` | indexing | 64 × 65 | offset | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/large/offset_axis0_r64_c7_k1_smallest` | indexing | 64 × 7 | offset | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/large/offset_axis0_r8_c257_k1_smallest` | indexing | 8 × 257 | offset | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/large/offset_axis0_r8_c31_k1_smallest` | indexing | 8 × 31 | offset | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/large/offset_axis0_r8_c65_k1_smallest` | indexing | 8 × 65 | offset | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/large/offset_axis0_r8_c7_k1_smallest` | indexing | 8 × 7 | offset | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/large/offset_axis1_r128_c257_k256_smallest` | indexing | 128 × 257 | offset | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/large/offset_axis1_r128_c31_k30_smallest` | indexing | 128 × 31 | offset | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/large/offset_axis1_r128_c65_k64_smallest` | indexing | 128 × 65 | offset | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/large/offset_axis1_r128_c7_k6_smallest` | indexing | 128 × 7 | offset | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/large/offset_axis1_r32_c257_k256_smallest` | indexing | 32 × 257 | offset | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/large/offset_axis1_r32_c31_k30_smallest` | indexing | 32 × 31 | offset | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/large/offset_axis1_r32_c65_k64_smallest` | indexing | 32 × 65 | offset | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/large/offset_axis1_r32_c7_k6_smallest` | indexing | 32 × 7 | offset | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/large/offset_axis1_r64_c257_k256_smallest` | indexing | 64 × 257 | offset | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/large/offset_axis1_r64_c31_k30_smallest` | indexing | 64 × 31 | offset | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/large/offset_axis1_r64_c65_k64_smallest` | indexing | 64 × 65 | offset | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/large/offset_axis1_r64_c7_k6_smallest` | indexing | 64 × 7 | offset | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/large/offset_axis1_r8_c257_k256_smallest` | indexing | 8 × 257 | offset | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/large/offset_axis1_r8_c31_k30_smallest` | indexing | 8 × 31 | offset | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/large/offset_axis1_r8_c65_k64_smallest` | indexing | 8 × 65 | offset | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/large/offset_axis1_r8_c7_k6_smallest` | indexing | 8 × 7 | offset | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/large/strided_axis0_r128_c257_k1_smallest` | indexing | 128 × 257 | strided | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/large/strided_axis0_r128_c31_k1_smallest` | indexing | 128 × 31 | strided | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/large/strided_axis0_r128_c65_k1_smallest` | indexing | 128 × 65 | strided | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/large/strided_axis0_r128_c7_k1_smallest` | indexing | 128 × 7 | strided | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/large/strided_axis0_r32_c257_k1_smallest` | indexing | 32 × 257 | strided | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/large/strided_axis0_r32_c31_k1_smallest` | indexing | 32 × 31 | strided | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/large/strided_axis0_r32_c65_k1_smallest` | indexing | 32 × 65 | strided | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/large/strided_axis0_r32_c7_k1_smallest` | indexing | 32 × 7 | strided | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/large/strided_axis0_r64_c257_k1_smallest` | indexing | 64 × 257 | strided | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/large/strided_axis0_r64_c31_k1_smallest` | indexing | 64 × 31 | strided | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/large/strided_axis0_r64_c65_k1_smallest` | indexing | 64 × 65 | strided | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/large/strided_axis0_r64_c7_k1_smallest` | indexing | 64 × 7 | strided | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/large/strided_axis0_r8_c257_k1_smallest` | indexing | 8 × 257 | strided | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/large/strided_axis0_r8_c31_k1_smallest` | indexing | 8 × 31 | strided | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/large/strided_axis0_r8_c65_k1_smallest` | indexing | 8 × 65 | strided | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/large/strided_axis0_r8_c7_k1_smallest` | indexing | 8 × 7 | strided | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/large/strided_axis1_r128_c257_k256_smallest` | indexing | 128 × 257 | strided | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/large/strided_axis1_r128_c31_k30_smallest` | indexing | 128 × 31 | strided | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/large/strided_axis1_r128_c65_k64_smallest` | indexing | 128 × 65 | strided | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/large/strided_axis1_r128_c7_k6_smallest` | indexing | 128 × 7 | strided | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/large/strided_axis1_r32_c257_k256_smallest` | indexing | 32 × 257 | strided | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/large/strided_axis1_r32_c31_k30_smallest` | indexing | 32 × 31 | strided | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/large/strided_axis1_r32_c65_k64_smallest` | indexing | 32 × 65 | strided | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/large/strided_axis1_r32_c7_k6_smallest` | indexing | 32 × 7 | strided | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/large/strided_axis1_r64_c257_k256_smallest` | indexing | 64 × 257 | strided | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/large/strided_axis1_r64_c31_k30_smallest` | indexing | 64 × 31 | strided | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/large/strided_axis1_r64_c65_k64_smallest` | indexing | 64 × 65 | strided | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/large/strided_axis1_r64_c7_k6_smallest` | indexing | 64 × 7 | strided | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/large/strided_axis1_r8_c257_k256_smallest` | indexing | 8 × 257 | strided | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/large/strided_axis1_r8_c31_k30_smallest` | indexing | 8 × 31 | strided | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/large/strided_axis1_r8_c65_k64_smallest` | indexing | 8 × 65 | strided | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/large/strided_axis1_r8_c7_k6_smallest` | indexing | 8 × 7 | strided | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/small` | indexing | 3 × 3 | dense | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/small/dense_axis0_r16_c129_k9` | indexing | 16 × 129 | dense | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/small/dense_axis0_r16_c17_k9` | indexing | 16 × 17 | dense | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/small/dense_axis0_r16_c3_k9` | indexing | 16 × 3 | dense | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/small/dense_axis0_r16_c64_k9` | indexing | 16 × 64 | dense | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/small/dense_axis0_r33_c129_k17` | indexing | 33 × 129 | dense | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/small/dense_axis0_r33_c17_k17` | indexing | 33 × 17 | dense | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/small/dense_axis0_r33_c3_k17` | indexing | 33 × 3 | dense | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/small/dense_axis0_r33_c64_k17` | indexing | 33 × 64 | dense | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/small/dense_axis0_r3_c129_k2` | indexing | 3 × 129 | dense | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/small/dense_axis0_r3_c17_k2` | indexing | 3 × 17 | dense | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/small/dense_axis0_r3_c3_k2` | indexing | 3 × 3 | dense | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/small/dense_axis0_r3_c64_k2` | indexing | 3 × 64 | dense | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/small/dense_axis0_r7_c129_k4` | indexing | 7 × 129 | dense | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/small/dense_axis0_r7_c17_k4` | indexing | 7 × 17 | dense | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/small/dense_axis0_r7_c3_k4` | indexing | 7 × 3 | dense | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/small/dense_axis0_r7_c64_k4` | indexing | 7 × 64 | dense | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/small/dense_axis1_r16_c129_k65` | indexing | 16 × 129 | dense | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/small/dense_axis1_r16_c17_k9` | indexing | 16 × 17 | dense | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/small/dense_axis1_r16_c3_k2` | indexing | 16 × 3 | dense | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/small/dense_axis1_r16_c64_k33` | indexing | 16 × 64 | dense | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/small/dense_axis1_r33_c129_k65` | indexing | 33 × 129 | dense | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/small/dense_axis1_r33_c17_k9` | indexing | 33 × 17 | dense | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/small/dense_axis1_r33_c3_k2` | indexing | 33 × 3 | dense | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/small/dense_axis1_r33_c64_k33` | indexing | 33 × 64 | dense | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/small/dense_axis1_r3_c129_k65` | indexing | 3 × 129 | dense | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/small/dense_axis1_r3_c17_k9` | indexing | 3 × 17 | dense | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/small/dense_axis1_r3_c3_k2` | indexing | 3 × 3 | dense | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/small/dense_axis1_r3_c64_k33` | indexing | 3 × 64 | dense | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/small/dense_axis1_r7_c129_k65` | indexing | 7 × 129 | dense | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/small/dense_axis1_r7_c17_k9` | indexing | 7 × 17 | dense | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/small/dense_axis1_r7_c3_k2` | indexing | 7 × 3 | dense | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/small/dense_axis1_r7_c64_k33` | indexing | 7 × 64 | dense | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/small/offset_axis0_r16_c129_k9` | indexing | 16 × 129 | offset | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/small/offset_axis0_r16_c17_k9` | indexing | 16 × 17 | offset | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/small/offset_axis0_r16_c3_k9` | indexing | 16 × 3 | offset | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/small/offset_axis0_r16_c64_k9` | indexing | 16 × 64 | offset | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/small/offset_axis0_r33_c129_k17` | indexing | 33 × 129 | offset | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/small/offset_axis0_r33_c17_k17` | indexing | 33 × 17 | offset | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/small/offset_axis0_r33_c3_k17` | indexing | 33 × 3 | offset | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/small/offset_axis0_r33_c64_k17` | indexing | 33 × 64 | offset | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/small/offset_axis0_r3_c129_k2` | indexing | 3 × 129 | offset | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/small/offset_axis0_r3_c17_k2` | indexing | 3 × 17 | offset | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/small/offset_axis0_r3_c3_k2` | indexing | 3 × 3 | offset | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/small/offset_axis0_r3_c64_k2` | indexing | 3 × 64 | offset | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/small/offset_axis0_r7_c129_k4` | indexing | 7 × 129 | offset | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/small/offset_axis0_r7_c17_k4` | indexing | 7 × 17 | offset | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/small/offset_axis0_r7_c3_k4` | indexing | 7 × 3 | offset | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/small/offset_axis0_r7_c64_k4` | indexing | 7 × 64 | offset | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/small/offset_axis1_r16_c129_k65` | indexing | 16 × 129 | offset | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/small/offset_axis1_r16_c17_k9` | indexing | 16 × 17 | offset | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/small/offset_axis1_r16_c3_k2` | indexing | 16 × 3 | offset | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/small/offset_axis1_r16_c64_k33` | indexing | 16 × 64 | offset | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/small/offset_axis1_r33_c129_k65` | indexing | 33 × 129 | offset | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/small/offset_axis1_r33_c17_k9` | indexing | 33 × 17 | offset | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/small/offset_axis1_r33_c3_k2` | indexing | 33 × 3 | offset | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/small/offset_axis1_r33_c64_k33` | indexing | 33 × 64 | offset | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/small/offset_axis1_r3_c129_k65` | indexing | 3 × 129 | offset | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/small/offset_axis1_r3_c17_k9` | indexing | 3 × 17 | offset | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/small/offset_axis1_r3_c3_k2` | indexing | 3 × 3 | offset | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/small/offset_axis1_r3_c64_k33` | indexing | 3 × 64 | offset | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/small/offset_axis1_r7_c129_k65` | indexing | 7 × 129 | offset | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/small/offset_axis1_r7_c17_k9` | indexing | 7 × 17 | offset | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/small/offset_axis1_r7_c3_k2` | indexing | 7 × 3 | offset | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/small/offset_axis1_r7_c64_k33` | indexing | 7 × 64 | offset | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/small/strided_axis0_r16_c129_k9` | indexing | 16 × 129 | strided | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/small/strided_axis0_r16_c17_k9` | indexing | 16 × 17 | strided | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/small/strided_axis0_r16_c3_k9` | indexing | 16 × 3 | strided | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/small/strided_axis0_r16_c64_k9` | indexing | 16 × 64 | strided | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/small/strided_axis0_r33_c129_k17` | indexing | 33 × 129 | strided | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/small/strided_axis0_r33_c17_k17` | indexing | 33 × 17 | strided | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/small/strided_axis0_r33_c3_k17` | indexing | 33 × 3 | strided | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/small/strided_axis0_r33_c64_k17` | indexing | 33 × 64 | strided | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/small/strided_axis0_r3_c129_k2` | indexing | 3 × 129 | strided | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/small/strided_axis0_r3_c17_k2` | indexing | 3 × 17 | strided | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/small/strided_axis0_r3_c3_k2` | indexing | 3 × 3 | strided | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/small/strided_axis0_r3_c64_k2` | indexing | 3 × 64 | strided | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/small/strided_axis0_r7_c129_k4` | indexing | 7 × 129 | strided | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/small/strided_axis0_r7_c17_k4` | indexing | 7 × 17 | strided | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/small/strided_axis0_r7_c3_k4` | indexing | 7 × 3 | strided | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/small/strided_axis0_r7_c64_k4` | indexing | 7 × 64 | strided | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/small/strided_axis1_r16_c129_k65` | indexing | 16 × 129 | strided | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/small/strided_axis1_r16_c17_k9` | indexing | 16 × 17 | strided | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/small/strided_axis1_r16_c3_k2` | indexing | 16 × 3 | strided | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/small/strided_axis1_r16_c64_k33` | indexing | 16 × 64 | strided | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/small/strided_axis1_r33_c129_k65` | indexing | 33 × 129 | strided | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/small/strided_axis1_r33_c17_k9` | indexing | 33 × 17 | strided | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/small/strided_axis1_r33_c3_k2` | indexing | 33 × 3 | strided | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/small/strided_axis1_r33_c64_k33` | indexing | 33 × 64 | strided | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/small/strided_axis1_r3_c129_k65` | indexing | 3 × 129 | strided | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/small/strided_axis1_r3_c17_k9` | indexing | 3 × 17 | strided | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/small/strided_axis1_r3_c3_k2` | indexing | 3 × 3 | strided | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/small/strided_axis1_r3_c64_k33` | indexing | 3 × 64 | strided | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/small/strided_axis1_r7_c129_k65` | indexing | 7 × 129 | strided | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/small/strided_axis1_r7_c17_k9` | indexing | 7 × 17 | strided | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/small/strided_axis1_r7_c3_k2` | indexing | 7 × 3 | strided | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/small/strided_axis1_r7_c64_k33` | indexing | 7 × 64 | strided | correct | not_run | — | not_run | — |
| `cuda/bf16/topk/strided` | indexing | 33 × 65 | strided | correct | not_run | — | not_run | — |
| `cuda/f32/topk/batched` | indexing | 3 × 11 × 7 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/batched/dense_axis1_b2_r16_c129_k8` | indexing | 2 × 16 × 129 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/batched/dense_axis1_b2_r16_c17_k8` | indexing | 2 × 16 × 17 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/batched/dense_axis1_b2_r16_c5_k8` | indexing | 2 × 16 × 5 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/batched/dense_axis1_b2_r16_c65_k8` | indexing | 2 × 16 × 65 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/batched/dense_axis1_b2_r33_c129_k16` | indexing | 2 × 33 × 129 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/batched/dense_axis1_b2_r33_c17_k16` | indexing | 2 × 33 × 17 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/batched/dense_axis1_b2_r33_c5_k16` | indexing | 2 × 33 × 5 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/batched/dense_axis1_b2_r33_c65_k16` | indexing | 2 × 33 × 65 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/batched/dense_axis1_b2_r3_c129_k1` | indexing | 2 × 3 × 129 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/batched/dense_axis1_b2_r3_c17_k1` | indexing | 2 × 3 × 17 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/batched/dense_axis1_b2_r3_c5_k1` | indexing | 2 × 3 × 5 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/batched/dense_axis1_b2_r3_c65_k1` | indexing | 2 × 3 × 65 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/batched/dense_axis1_b2_r7_c129_k3` | indexing | 2 × 7 × 129 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/batched/dense_axis1_b2_r7_c17_k3` | indexing | 2 × 7 × 17 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/batched/dense_axis1_b2_r7_c5_k3` | indexing | 2 × 7 × 5 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/batched/dense_axis1_b2_r7_c65_k3` | indexing | 2 × 7 × 65 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/batched/dense_axis1_b4_r16_c129_k8` | indexing | 4 × 16 × 129 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/batched/dense_axis1_b4_r16_c17_k8` | indexing | 4 × 16 × 17 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/batched/dense_axis1_b4_r16_c5_k8` | indexing | 4 × 16 × 5 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/batched/dense_axis1_b4_r16_c65_k8` | indexing | 4 × 16 × 65 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/batched/dense_axis1_b4_r33_c129_k16` | indexing | 4 × 33 × 129 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/batched/dense_axis1_b4_r33_c17_k16` | indexing | 4 × 33 × 17 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/batched/dense_axis1_b4_r33_c5_k16` | indexing | 4 × 33 × 5 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/batched/dense_axis1_b4_r33_c65_k16` | indexing | 4 × 33 × 65 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/batched/dense_axis1_b4_r3_c129_k1` | indexing | 4 × 3 × 129 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/batched/dense_axis1_b4_r3_c17_k1` | indexing | 4 × 3 × 17 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/batched/dense_axis1_b4_r3_c5_k1` | indexing | 4 × 3 × 5 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/batched/dense_axis1_b4_r3_c65_k1` | indexing | 4 × 3 × 65 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/batched/dense_axis1_b4_r7_c129_k3` | indexing | 4 × 7 × 129 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/batched/dense_axis1_b4_r7_c17_k3` | indexing | 4 × 7 × 17 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/batched/dense_axis1_b4_r7_c5_k3` | indexing | 4 × 7 × 5 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/batched/dense_axis1_b4_r7_c65_k3` | indexing | 4 × 7 × 65 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/batched/dense_axis2_b2_r16_c129_k64` | indexing | 2 × 16 × 129 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/batched/dense_axis2_b2_r16_c17_k8` | indexing | 2 × 16 × 17 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/batched/dense_axis2_b2_r16_c5_k2` | indexing | 2 × 16 × 5 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/batched/dense_axis2_b2_r16_c65_k32` | indexing | 2 × 16 × 65 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/batched/dense_axis2_b2_r33_c129_k64` | indexing | 2 × 33 × 129 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/batched/dense_axis2_b2_r33_c17_k8` | indexing | 2 × 33 × 17 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/batched/dense_axis2_b2_r33_c5_k2` | indexing | 2 × 33 × 5 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/batched/dense_axis2_b2_r33_c65_k32` | indexing | 2 × 33 × 65 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/batched/dense_axis2_b2_r3_c129_k64` | indexing | 2 × 3 × 129 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/batched/dense_axis2_b2_r3_c17_k8` | indexing | 2 × 3 × 17 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/batched/dense_axis2_b2_r3_c5_k2` | indexing | 2 × 3 × 5 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/batched/dense_axis2_b2_r3_c65_k32` | indexing | 2 × 3 × 65 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/batched/dense_axis2_b2_r7_c129_k64` | indexing | 2 × 7 × 129 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/batched/dense_axis2_b2_r7_c17_k8` | indexing | 2 × 7 × 17 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/batched/dense_axis2_b2_r7_c5_k2` | indexing | 2 × 7 × 5 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/batched/dense_axis2_b2_r7_c65_k32` | indexing | 2 × 7 × 65 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/batched/dense_axis2_b4_r16_c129_k64` | indexing | 4 × 16 × 129 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/batched/dense_axis2_b4_r16_c17_k8` | indexing | 4 × 16 × 17 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/batched/dense_axis2_b4_r16_c5_k2` | indexing | 4 × 16 × 5 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/batched/dense_axis2_b4_r16_c65_k32` | indexing | 4 × 16 × 65 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/batched/dense_axis2_b4_r33_c129_k64` | indexing | 4 × 33 × 129 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/batched/dense_axis2_b4_r33_c17_k8` | indexing | 4 × 33 × 17 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/batched/dense_axis2_b4_r33_c5_k2` | indexing | 4 × 33 × 5 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/batched/dense_axis2_b4_r33_c65_k32` | indexing | 4 × 33 × 65 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/batched/dense_axis2_b4_r3_c129_k64` | indexing | 4 × 3 × 129 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/batched/dense_axis2_b4_r3_c17_k8` | indexing | 4 × 3 × 17 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/batched/dense_axis2_b4_r3_c5_k2` | indexing | 4 × 3 × 5 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/batched/dense_axis2_b4_r3_c65_k32` | indexing | 4 × 3 × 65 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/batched/dense_axis2_b4_r7_c129_k64` | indexing | 4 × 7 × 129 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/batched/dense_axis2_b4_r7_c17_k8` | indexing | 4 × 7 × 17 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/batched/dense_axis2_b4_r7_c5_k2` | indexing | 4 × 7 × 5 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/batched/dense_axis2_b4_r7_c65_k32` | indexing | 4 × 7 × 65 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/large` | indexing | 8 × 7 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/large/dense_axis0_r128_c257_k1_smallest` | indexing | 128 × 257 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/large/dense_axis0_r128_c31_k1_smallest` | indexing | 128 × 31 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/large/dense_axis0_r128_c65_k1_smallest` | indexing | 128 × 65 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/large/dense_axis0_r128_c7_k1_smallest` | indexing | 128 × 7 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/large/dense_axis0_r32_c257_k1_smallest` | indexing | 32 × 257 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/large/dense_axis0_r32_c31_k1_smallest` | indexing | 32 × 31 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/large/dense_axis0_r32_c65_k1_smallest` | indexing | 32 × 65 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/large/dense_axis0_r32_c7_k1_smallest` | indexing | 32 × 7 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/large/dense_axis0_r64_c257_k1_smallest` | indexing | 64 × 257 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/large/dense_axis0_r64_c31_k1_smallest` | indexing | 64 × 31 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/large/dense_axis0_r64_c65_k1_smallest` | indexing | 64 × 65 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/large/dense_axis0_r64_c7_k1_smallest` | indexing | 64 × 7 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/large/dense_axis0_r8_c257_k1_smallest` | indexing | 8 × 257 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/large/dense_axis0_r8_c31_k1_smallest` | indexing | 8 × 31 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/large/dense_axis0_r8_c65_k1_smallest` | indexing | 8 × 65 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/large/dense_axis0_r8_c7_k1_smallest` | indexing | 8 × 7 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/large/dense_axis1_r128_c257_k256_smallest` | indexing | 128 × 257 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/large/dense_axis1_r128_c31_k30_smallest` | indexing | 128 × 31 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/large/dense_axis1_r128_c65_k64_smallest` | indexing | 128 × 65 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/large/dense_axis1_r128_c7_k6_smallest` | indexing | 128 × 7 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/large/dense_axis1_r32_c257_k256_smallest` | indexing | 32 × 257 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/large/dense_axis1_r32_c31_k30_smallest` | indexing | 32 × 31 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/large/dense_axis1_r32_c65_k64_smallest` | indexing | 32 × 65 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/large/dense_axis1_r32_c7_k6_smallest` | indexing | 32 × 7 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/large/dense_axis1_r64_c257_k256_smallest` | indexing | 64 × 257 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/large/dense_axis1_r64_c31_k30_smallest` | indexing | 64 × 31 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/large/dense_axis1_r64_c65_k64_smallest` | indexing | 64 × 65 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/large/dense_axis1_r64_c7_k6_smallest` | indexing | 64 × 7 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/large/dense_axis1_r8_c257_k256_smallest` | indexing | 8 × 257 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/large/dense_axis1_r8_c31_k30_smallest` | indexing | 8 × 31 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/large/dense_axis1_r8_c65_k64_smallest` | indexing | 8 × 65 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/large/dense_axis1_r8_c7_k6_smallest` | indexing | 8 × 7 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/large/offset_axis0_r128_c257_k1_smallest` | indexing | 128 × 257 | offset | correct | not_run | — | not_run | — |
| `cuda/f32/topk/large/offset_axis0_r128_c31_k1_smallest` | indexing | 128 × 31 | offset | correct | not_run | — | not_run | — |
| `cuda/f32/topk/large/offset_axis0_r128_c65_k1_smallest` | indexing | 128 × 65 | offset | correct | not_run | — | not_run | — |
| `cuda/f32/topk/large/offset_axis0_r128_c7_k1_smallest` | indexing | 128 × 7 | offset | correct | not_run | — | not_run | — |
| `cuda/f32/topk/large/offset_axis0_r32_c257_k1_smallest` | indexing | 32 × 257 | offset | correct | not_run | — | not_run | — |
| `cuda/f32/topk/large/offset_axis0_r32_c31_k1_smallest` | indexing | 32 × 31 | offset | correct | not_run | — | not_run | — |
| `cuda/f32/topk/large/offset_axis0_r32_c65_k1_smallest` | indexing | 32 × 65 | offset | correct | not_run | — | not_run | — |
| `cuda/f32/topk/large/offset_axis0_r32_c7_k1_smallest` | indexing | 32 × 7 | offset | correct | not_run | — | not_run | — |
| `cuda/f32/topk/large/offset_axis0_r64_c257_k1_smallest` | indexing | 64 × 257 | offset | correct | not_run | — | not_run | — |
| `cuda/f32/topk/large/offset_axis0_r64_c31_k1_smallest` | indexing | 64 × 31 | offset | correct | not_run | — | not_run | — |
| `cuda/f32/topk/large/offset_axis0_r64_c65_k1_smallest` | indexing | 64 × 65 | offset | correct | not_run | — | not_run | — |
| `cuda/f32/topk/large/offset_axis0_r64_c7_k1_smallest` | indexing | 64 × 7 | offset | correct | not_run | — | not_run | — |
| `cuda/f32/topk/large/offset_axis0_r8_c257_k1_smallest` | indexing | 8 × 257 | offset | correct | not_run | — | not_run | — |
| `cuda/f32/topk/large/offset_axis0_r8_c31_k1_smallest` | indexing | 8 × 31 | offset | correct | not_run | — | not_run | — |
| `cuda/f32/topk/large/offset_axis0_r8_c65_k1_smallest` | indexing | 8 × 65 | offset | correct | not_run | — | not_run | — |
| `cuda/f32/topk/large/offset_axis0_r8_c7_k1_smallest` | indexing | 8 × 7 | offset | correct | not_run | — | not_run | — |
| `cuda/f32/topk/large/offset_axis1_r128_c257_k256_smallest` | indexing | 128 × 257 | offset | correct | not_run | — | not_run | — |
| `cuda/f32/topk/large/offset_axis1_r128_c31_k30_smallest` | indexing | 128 × 31 | offset | correct | not_run | — | not_run | — |
| `cuda/f32/topk/large/offset_axis1_r128_c65_k64_smallest` | indexing | 128 × 65 | offset | correct | not_run | — | not_run | — |
| `cuda/f32/topk/large/offset_axis1_r128_c7_k6_smallest` | indexing | 128 × 7 | offset | correct | not_run | — | not_run | — |
| `cuda/f32/topk/large/offset_axis1_r32_c257_k256_smallest` | indexing | 32 × 257 | offset | correct | not_run | — | not_run | — |
| `cuda/f32/topk/large/offset_axis1_r32_c31_k30_smallest` | indexing | 32 × 31 | offset | correct | not_run | — | not_run | — |
| `cuda/f32/topk/large/offset_axis1_r32_c65_k64_smallest` | indexing | 32 × 65 | offset | correct | not_run | — | not_run | — |
| `cuda/f32/topk/large/offset_axis1_r32_c7_k6_smallest` | indexing | 32 × 7 | offset | correct | not_run | — | not_run | — |
| `cuda/f32/topk/large/offset_axis1_r64_c257_k256_smallest` | indexing | 64 × 257 | offset | correct | not_run | — | not_run | — |
| `cuda/f32/topk/large/offset_axis1_r64_c31_k30_smallest` | indexing | 64 × 31 | offset | correct | not_run | — | not_run | — |
| `cuda/f32/topk/large/offset_axis1_r64_c65_k64_smallest` | indexing | 64 × 65 | offset | correct | not_run | — | not_run | — |
| `cuda/f32/topk/large/offset_axis1_r64_c7_k6_smallest` | indexing | 64 × 7 | offset | correct | not_run | — | not_run | — |
| `cuda/f32/topk/large/offset_axis1_r8_c257_k256_smallest` | indexing | 8 × 257 | offset | correct | not_run | — | not_run | — |
| `cuda/f32/topk/large/offset_axis1_r8_c31_k30_smallest` | indexing | 8 × 31 | offset | correct | not_run | — | not_run | — |
| `cuda/f32/topk/large/offset_axis1_r8_c65_k64_smallest` | indexing | 8 × 65 | offset | correct | not_run | — | not_run | — |
| `cuda/f32/topk/large/offset_axis1_r8_c7_k6_smallest` | indexing | 8 × 7 | offset | correct | not_run | — | not_run | — |
| `cuda/f32/topk/large/strided_axis0_r128_c257_k1_smallest` | indexing | 128 × 257 | strided | correct | not_run | — | not_run | — |
| `cuda/f32/topk/large/strided_axis0_r128_c31_k1_smallest` | indexing | 128 × 31 | strided | correct | not_run | — | not_run | — |
| `cuda/f32/topk/large/strided_axis0_r128_c65_k1_smallest` | indexing | 128 × 65 | strided | correct | not_run | — | not_run | — |
| `cuda/f32/topk/large/strided_axis0_r128_c7_k1_smallest` | indexing | 128 × 7 | strided | correct | not_run | — | not_run | — |
| `cuda/f32/topk/large/strided_axis0_r32_c257_k1_smallest` | indexing | 32 × 257 | strided | correct | not_run | — | not_run | — |
| `cuda/f32/topk/large/strided_axis0_r32_c31_k1_smallest` | indexing | 32 × 31 | strided | correct | not_run | — | not_run | — |
| `cuda/f32/topk/large/strided_axis0_r32_c65_k1_smallest` | indexing | 32 × 65 | strided | correct | not_run | — | not_run | — |
| `cuda/f32/topk/large/strided_axis0_r32_c7_k1_smallest` | indexing | 32 × 7 | strided | correct | not_run | — | not_run | — |
| `cuda/f32/topk/large/strided_axis0_r64_c257_k1_smallest` | indexing | 64 × 257 | strided | correct | not_run | — | not_run | — |
| `cuda/f32/topk/large/strided_axis0_r64_c31_k1_smallest` | indexing | 64 × 31 | strided | correct | not_run | — | not_run | — |
| `cuda/f32/topk/large/strided_axis0_r64_c65_k1_smallest` | indexing | 64 × 65 | strided | correct | not_run | — | not_run | — |
| `cuda/f32/topk/large/strided_axis0_r64_c7_k1_smallest` | indexing | 64 × 7 | strided | correct | not_run | — | not_run | — |
| `cuda/f32/topk/large/strided_axis0_r8_c257_k1_smallest` | indexing | 8 × 257 | strided | correct | not_run | — | not_run | — |
| `cuda/f32/topk/large/strided_axis0_r8_c31_k1_smallest` | indexing | 8 × 31 | strided | correct | not_run | — | not_run | — |
| `cuda/f32/topk/large/strided_axis0_r8_c65_k1_smallest` | indexing | 8 × 65 | strided | correct | not_run | — | not_run | — |
| `cuda/f32/topk/large/strided_axis0_r8_c7_k1_smallest` | indexing | 8 × 7 | strided | correct | not_run | — | not_run | — |
| `cuda/f32/topk/large/strided_axis1_r128_c257_k256_smallest` | indexing | 128 × 257 | strided | correct | not_run | — | not_run | — |
| `cuda/f32/topk/large/strided_axis1_r128_c31_k30_smallest` | indexing | 128 × 31 | strided | correct | not_run | — | not_run | — |
| `cuda/f32/topk/large/strided_axis1_r128_c65_k64_smallest` | indexing | 128 × 65 | strided | correct | not_run | — | not_run | — |
| `cuda/f32/topk/large/strided_axis1_r128_c7_k6_smallest` | indexing | 128 × 7 | strided | correct | not_run | — | not_run | — |
| `cuda/f32/topk/large/strided_axis1_r32_c257_k256_smallest` | indexing | 32 × 257 | strided | correct | not_run | — | not_run | — |
| `cuda/f32/topk/large/strided_axis1_r32_c31_k30_smallest` | indexing | 32 × 31 | strided | correct | not_run | — | not_run | — |
| `cuda/f32/topk/large/strided_axis1_r32_c65_k64_smallest` | indexing | 32 × 65 | strided | correct | not_run | — | not_run | — |
| `cuda/f32/topk/large/strided_axis1_r32_c7_k6_smallest` | indexing | 32 × 7 | strided | correct | not_run | — | not_run | — |
| `cuda/f32/topk/large/strided_axis1_r64_c257_k256_smallest` | indexing | 64 × 257 | strided | correct | not_run | — | not_run | — |
| `cuda/f32/topk/large/strided_axis1_r64_c31_k30_smallest` | indexing | 64 × 31 | strided | correct | not_run | — | not_run | — |
| `cuda/f32/topk/large/strided_axis1_r64_c65_k64_smallest` | indexing | 64 × 65 | strided | correct | not_run | — | not_run | — |
| `cuda/f32/topk/large/strided_axis1_r64_c7_k6_smallest` | indexing | 64 × 7 | strided | correct | not_run | — | not_run | — |
| `cuda/f32/topk/large/strided_axis1_r8_c257_k256_smallest` | indexing | 8 × 257 | strided | correct | not_run | — | not_run | — |
| `cuda/f32/topk/large/strided_axis1_r8_c31_k30_smallest` | indexing | 8 × 31 | strided | correct | not_run | — | not_run | — |
| `cuda/f32/topk/large/strided_axis1_r8_c65_k64_smallest` | indexing | 8 × 65 | strided | correct | not_run | — | not_run | — |
| `cuda/f32/topk/large/strided_axis1_r8_c7_k6_smallest` | indexing | 8 × 7 | strided | correct | not_run | — | not_run | — |
| `cuda/f32/topk/small` | indexing | 3 × 3 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/small/dense_axis0_r16_c129_k9` | indexing | 16 × 129 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/small/dense_axis0_r16_c17_k9` | indexing | 16 × 17 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/small/dense_axis0_r16_c3_k9` | indexing | 16 × 3 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/small/dense_axis0_r16_c64_k9` | indexing | 16 × 64 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/small/dense_axis0_r33_c129_k17` | indexing | 33 × 129 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/small/dense_axis0_r33_c17_k17` | indexing | 33 × 17 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/small/dense_axis0_r33_c3_k17` | indexing | 33 × 3 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/small/dense_axis0_r33_c64_k17` | indexing | 33 × 64 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/small/dense_axis0_r3_c129_k2` | indexing | 3 × 129 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/small/dense_axis0_r3_c17_k2` | indexing | 3 × 17 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/small/dense_axis0_r3_c3_k2` | indexing | 3 × 3 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/small/dense_axis0_r3_c64_k2` | indexing | 3 × 64 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/small/dense_axis0_r7_c129_k4` | indexing | 7 × 129 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/small/dense_axis0_r7_c17_k4` | indexing | 7 × 17 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/small/dense_axis0_r7_c3_k4` | indexing | 7 × 3 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/small/dense_axis0_r7_c64_k4` | indexing | 7 × 64 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/small/dense_axis1_r16_c129_k65` | indexing | 16 × 129 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/small/dense_axis1_r16_c17_k9` | indexing | 16 × 17 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/small/dense_axis1_r16_c3_k2` | indexing | 16 × 3 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/small/dense_axis1_r16_c64_k33` | indexing | 16 × 64 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/small/dense_axis1_r33_c129_k65` | indexing | 33 × 129 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/small/dense_axis1_r33_c17_k9` | indexing | 33 × 17 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/small/dense_axis1_r33_c3_k2` | indexing | 33 × 3 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/small/dense_axis1_r33_c64_k33` | indexing | 33 × 64 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/small/dense_axis1_r3_c129_k65` | indexing | 3 × 129 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/small/dense_axis1_r3_c17_k9` | indexing | 3 × 17 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/small/dense_axis1_r3_c3_k2` | indexing | 3 × 3 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/small/dense_axis1_r3_c64_k33` | indexing | 3 × 64 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/small/dense_axis1_r7_c129_k65` | indexing | 7 × 129 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/small/dense_axis1_r7_c17_k9` | indexing | 7 × 17 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/small/dense_axis1_r7_c3_k2` | indexing | 7 × 3 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/small/dense_axis1_r7_c64_k33` | indexing | 7 × 64 | dense | correct | not_run | — | not_run | — |
| `cuda/f32/topk/small/offset_axis0_r16_c129_k9` | indexing | 16 × 129 | offset | correct | not_run | — | not_run | — |
| `cuda/f32/topk/small/offset_axis0_r16_c17_k9` | indexing | 16 × 17 | offset | correct | not_run | — | not_run | — |
| `cuda/f32/topk/small/offset_axis0_r16_c3_k9` | indexing | 16 × 3 | offset | correct | not_run | — | not_run | — |
| `cuda/f32/topk/small/offset_axis0_r16_c64_k9` | indexing | 16 × 64 | offset | correct | not_run | — | not_run | — |
| `cuda/f32/topk/small/offset_axis0_r33_c129_k17` | indexing | 33 × 129 | offset | correct | not_run | — | not_run | — |
| `cuda/f32/topk/small/offset_axis0_r33_c17_k17` | indexing | 33 × 17 | offset | correct | not_run | — | not_run | — |
| `cuda/f32/topk/small/offset_axis0_r33_c3_k17` | indexing | 33 × 3 | offset | correct | not_run | — | not_run | — |
| `cuda/f32/topk/small/offset_axis0_r33_c64_k17` | indexing | 33 × 64 | offset | correct | not_run | — | not_run | — |
| `cuda/f32/topk/small/offset_axis0_r3_c129_k2` | indexing | 3 × 129 | offset | correct | not_run | — | not_run | — |
| `cuda/f32/topk/small/offset_axis0_r3_c17_k2` | indexing | 3 × 17 | offset | correct | not_run | — | not_run | — |
| `cuda/f32/topk/small/offset_axis0_r3_c3_k2` | indexing | 3 × 3 | offset | correct | not_run | — | not_run | — |
| `cuda/f32/topk/small/offset_axis0_r3_c64_k2` | indexing | 3 × 64 | offset | correct | not_run | — | not_run | — |
| `cuda/f32/topk/small/offset_axis0_r7_c129_k4` | indexing | 7 × 129 | offset | correct | not_run | — | not_run | — |
| `cuda/f32/topk/small/offset_axis0_r7_c17_k4` | indexing | 7 × 17 | offset | correct | not_run | — | not_run | — |
| `cuda/f32/topk/small/offset_axis0_r7_c3_k4` | indexing | 7 × 3 | offset | correct | not_run | — | not_run | — |
| `cuda/f32/topk/small/offset_axis0_r7_c64_k4` | indexing | 7 × 64 | offset | correct | not_run | — | not_run | — |
| `cuda/f32/topk/small/offset_axis1_r16_c129_k65` | indexing | 16 × 129 | offset | correct | not_run | — | not_run | — |
| `cuda/f32/topk/small/offset_axis1_r16_c17_k9` | indexing | 16 × 17 | offset | correct | not_run | — | not_run | — |
| `cuda/f32/topk/small/offset_axis1_r16_c3_k2` | indexing | 16 × 3 | offset | correct | not_run | — | not_run | — |
| `cuda/f32/topk/small/offset_axis1_r16_c64_k33` | indexing | 16 × 64 | offset | correct | not_run | — | not_run | — |
| `cuda/f32/topk/small/offset_axis1_r33_c129_k65` | indexing | 33 × 129 | offset | correct | not_run | — | not_run | — |
| `cuda/f32/topk/small/offset_axis1_r33_c17_k9` | indexing | 33 × 17 | offset | correct | not_run | — | not_run | — |
| `cuda/f32/topk/small/offset_axis1_r33_c3_k2` | indexing | 33 × 3 | offset | correct | not_run | — | not_run | — |
| `cuda/f32/topk/small/offset_axis1_r33_c64_k33` | indexing | 33 × 64 | offset | correct | not_run | — | not_run | — |
| `cuda/f32/topk/small/offset_axis1_r3_c129_k65` | indexing | 3 × 129 | offset | correct | not_run | — | not_run | — |
| `cuda/f32/topk/small/offset_axis1_r3_c17_k9` | indexing | 3 × 17 | offset | correct | not_run | — | not_run | — |
| `cuda/f32/topk/small/offset_axis1_r3_c3_k2` | indexing | 3 × 3 | offset | correct | not_run | — | not_run | — |
| `cuda/f32/topk/small/offset_axis1_r3_c64_k33` | indexing | 3 × 64 | offset | correct | not_run | — | not_run | — |
| `cuda/f32/topk/small/offset_axis1_r7_c129_k65` | indexing | 7 × 129 | offset | correct | not_run | — | not_run | — |
| `cuda/f32/topk/small/offset_axis1_r7_c17_k9` | indexing | 7 × 17 | offset | correct | not_run | — | not_run | — |
| `cuda/f32/topk/small/offset_axis1_r7_c3_k2` | indexing | 7 × 3 | offset | correct | not_run | — | not_run | — |
| `cuda/f32/topk/small/offset_axis1_r7_c64_k33` | indexing | 7 × 64 | offset | correct | not_run | — | not_run | — |
| `cuda/f32/topk/small/strided_axis0_r16_c129_k9` | indexing | 16 × 129 | strided | correct | not_run | — | not_run | — |
| `cuda/f32/topk/small/strided_axis0_r16_c17_k9` | indexing | 16 × 17 | strided | correct | not_run | — | not_run | — |
| `cuda/f32/topk/small/strided_axis0_r16_c3_k9` | indexing | 16 × 3 | strided | correct | not_run | — | not_run | — |
| `cuda/f32/topk/small/strided_axis0_r16_c64_k9` | indexing | 16 × 64 | strided | correct | not_run | — | not_run | — |
| `cuda/f32/topk/small/strided_axis0_r33_c129_k17` | indexing | 33 × 129 | strided | correct | not_run | — | not_run | — |
| `cuda/f32/topk/small/strided_axis0_r33_c17_k17` | indexing | 33 × 17 | strided | correct | not_run | — | not_run | — |
| `cuda/f32/topk/small/strided_axis0_r33_c3_k17` | indexing | 33 × 3 | strided | correct | not_run | — | not_run | — |
| `cuda/f32/topk/small/strided_axis0_r33_c64_k17` | indexing | 33 × 64 | strided | correct | not_run | — | not_run | — |
| `cuda/f32/topk/small/strided_axis0_r3_c129_k2` | indexing | 3 × 129 | strided | correct | not_run | — | not_run | — |
| `cuda/f32/topk/small/strided_axis0_r3_c17_k2` | indexing | 3 × 17 | strided | correct | not_run | — | not_run | — |
| `cuda/f32/topk/small/strided_axis0_r3_c3_k2` | indexing | 3 × 3 | strided | correct | not_run | — | not_run | — |
| `cuda/f32/topk/small/strided_axis0_r3_c64_k2` | indexing | 3 × 64 | strided | correct | not_run | — | not_run | — |
| `cuda/f32/topk/small/strided_axis0_r7_c129_k4` | indexing | 7 × 129 | strided | correct | not_run | — | not_run | — |
| `cuda/f32/topk/small/strided_axis0_r7_c17_k4` | indexing | 7 × 17 | strided | correct | not_run | — | not_run | — |
| `cuda/f32/topk/small/strided_axis0_r7_c3_k4` | indexing | 7 × 3 | strided | correct | not_run | — | not_run | — |
| `cuda/f32/topk/small/strided_axis0_r7_c64_k4` | indexing | 7 × 64 | strided | correct | not_run | — | not_run | — |
| `cuda/f32/topk/small/strided_axis1_r16_c129_k65` | indexing | 16 × 129 | strided | correct | not_run | — | not_run | — |
| `cuda/f32/topk/small/strided_axis1_r16_c17_k9` | indexing | 16 × 17 | strided | correct | not_run | — | not_run | — |
| `cuda/f32/topk/small/strided_axis1_r16_c3_k2` | indexing | 16 × 3 | strided | correct | not_run | — | not_run | — |
| `cuda/f32/topk/small/strided_axis1_r16_c64_k33` | indexing | 16 × 64 | strided | correct | not_run | — | not_run | — |
| `cuda/f32/topk/small/strided_axis1_r33_c129_k65` | indexing | 33 × 129 | strided | correct | not_run | — | not_run | — |
| `cuda/f32/topk/small/strided_axis1_r33_c17_k9` | indexing | 33 × 17 | strided | correct | not_run | — | not_run | — |
| `cuda/f32/topk/small/strided_axis1_r33_c3_k2` | indexing | 33 × 3 | strided | correct | not_run | — | not_run | — |
| `cuda/f32/topk/small/strided_axis1_r33_c64_k33` | indexing | 33 × 64 | strided | correct | not_run | — | not_run | — |
| `cuda/f32/topk/small/strided_axis1_r3_c129_k65` | indexing | 3 × 129 | strided | correct | not_run | — | not_run | — |
| `cuda/f32/topk/small/strided_axis1_r3_c17_k9` | indexing | 3 × 17 | strided | correct | not_run | — | not_run | — |
| `cuda/f32/topk/small/strided_axis1_r3_c3_k2` | indexing | 3 × 3 | strided | correct | not_run | — | not_run | — |
| `cuda/f32/topk/small/strided_axis1_r3_c64_k33` | indexing | 3 × 64 | strided | correct | not_run | — | not_run | — |
| `cuda/f32/topk/small/strided_axis1_r7_c129_k65` | indexing | 7 × 129 | strided | correct | not_run | — | not_run | — |
| `cuda/f32/topk/small/strided_axis1_r7_c17_k9` | indexing | 7 × 17 | strided | correct | not_run | — | not_run | — |
| `cuda/f32/topk/small/strided_axis1_r7_c3_k2` | indexing | 7 × 3 | strided | correct | not_run | — | not_run | — |
| `cuda/f32/topk/small/strided_axis1_r7_c64_k33` | indexing | 7 × 64 | strided | correct | not_run | — | not_run | — |
| `cuda/f32/topk/strided` | indexing | 33 × 65 | strided | correct | not_run | — | not_run | — |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
