# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `7fa464b58b1bc541618d9c8bb55335d709dc66bcdc28cfc1e53d8ce4880c6e5f`  
Binary SHA-256: `a1e58d69ce8e239ab9411f22fc29b1af1b70bb0e44383e4384e3e50970d24bc8`  
Timing: 5 iterations × 5 samples × 3 rounds

## Checks

- Cases run: 2
- Numerical check passed: 2
- Operator check and timing passed: 2
- Program check and timing passed: 2
- Numerical failures: 0
- Operator failures: 0
- Program failures: 0

## Cases

| Case | Family | Shape | Layout | Numerical | Operator | Affon / PyTorch ms | Program | Affon / PyTorch ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `metal/bf16/topk/large/dense_axis1_r128_c257_k256_smallest` | indexing | 128 × 257 | dense | correct | [passed](operator_peer/metal__bf16__topk__large__dense_axis1_r128_c257_k256_smallest.json) | 0.9328 / 1.133 | [passed](program_peer/metal__bf16__topk__large__dense_axis1_r128_c257_k256_smallest.json) | 1.019 / 1.168 |
| `metal/bf16/topk/large/offset_axis1_r128_c257_k256_smallest` | indexing | 128 × 257 | offset | correct | [passed](operator_peer/metal__bf16__topk__large__offset_axis1_r128_c257_k256_smallest.json) | 1.022 / 1.132 | [passed](program_peer/metal__bf16__topk__large__offset_axis1_r128_c257_k256_smallest.json) | 0.9756 / 1.15 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
