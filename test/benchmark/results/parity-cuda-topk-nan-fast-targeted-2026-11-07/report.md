# Parity batch report

Target: `NVIDIA GeForce RTX 3090/7.0.0-38-generic/x86_64`  
Catalog SHA-256: `d3ab079eb92f0d536780c5261540b79fae21822556266bccfcd17d054121563c`  
Binary SHA-256: `091a70eaf968058c5e4e4a5e5e28dfb877d8a1ec5c30984642d5d3313eebe027`  
Timing: 10 iterations × 5 samples × 3 rounds

## Checks

- Cases run: 3
- Numerical check passed: 3
- Operator check and timing passed: 3
- Program check and timing passed: 3
- Numerical failures: 0
- Operator failures: 0
- Program failures: 0
- Operator checks pending: 0
- Program checks pending: 0
- Checkpoint updated: 2026-11-06T12:17:33.736171+00:00

## Cases

| Case | Family | Shape | Layout | Numerical | Operator | Affon / PyTorch ms | Program | Affon / PyTorch ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `cuda/bf16/topk/batched/dense_axis2_b2_r3_c17_k8` | indexing | 2 × 3 × 17 | dense | correct | [passed](operator_peer/cuda__bf16__topk__batched__dense_axis2_b2_r3_c17_k8.json) | 0.0447 / 0.04059 | [passed](program_peer/cuda__bf16__topk__batched__dense_axis2_b2_r3_c17_k8.json) | 0.04779 / 0.04034 |
| `cuda/bf16/topk/large/dense_axis1_r8_c31_k30_smallest` | indexing | 8 × 31 | dense | correct | [passed](operator_peer/cuda__bf16__topk__large__dense_axis1_r8_c31_k30_smallest.json) | 0.0882 / 0.04633 | [passed](program_peer/cuda__bf16__topk__large__dense_axis1_r8_c31_k30_smallest.json) | 0.0904 / 0.04511 |
| `cuda/f32/topk/large/strided_axis1_r128_c257_k256_smallest` | indexing | 128 × 257 | strided | correct | [passed](operator_peer/cuda__f32__topk__large__strided_axis1_r128_c257_k256_smallest.json) | 1.202 / 0.4918 | [passed](program_peer/cuda__f32__topk__large__strided_axis1_r128_c257_k256_smallest.json) | 1.21 / 0.5109 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
