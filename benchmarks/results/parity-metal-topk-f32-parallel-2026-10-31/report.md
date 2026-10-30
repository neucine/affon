# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `37597634ab614b632eac5e6b026408fc1ecb053f607b86d8cac459cda9c5663d`  
Binary SHA-256: `5e934b3026f06481e7b89dd409dc51caa017fd00e07a31fa6d5bad76a3adc4d2`  
Timing: 5 iterations × 5 samples × 3 rounds

## Checks

- Cases run: 3
- Numerical check passed: 3
- Operator check and timing passed: 3
- Program check and timing passed: 3
- Numerical failures: 0
- Operator failures: 0
- Program failures: 0

## Cases

| Case | Family | Shape | Layout | Numerical | Operator | Affon / PyTorch ms | Program | Affon / PyTorch ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `metal/f32/topk/large/dense_axis0_r32_c65_k1_smallest` | indexing | 32 × 65 | dense | correct | [passed](operator_peer/metal__f32__topk__large__dense_axis0_r32_c65_k1_smallest.json) | 0.2636 / 0.5355 | [passed](program_peer/metal__f32__topk__large__dense_axis0_r32_c65_k1_smallest.json) | 0.2984 / 0.5686 |
| `metal/f32/topk/large/dense_axis1_r128_c257_k256_smallest` | indexing | 128 × 257 | dense | correct | [passed](operator_peer/metal__f32__topk__large__dense_axis1_r128_c257_k256_smallest.json) | 2.929 / 1.118 | [passed](program_peer/metal__f32__topk__large__dense_axis1_r128_c257_k256_smallest.json) | 2.974 / 1.116 |
| `metal/f32/topk/large/strided_axis1_r128_c257_k256_smallest` | indexing | 128 × 257 | strided | correct | [passed](operator_peer/metal__f32__topk__large__strided_axis1_r128_c257_k256_smallest.json) | 2.926 / 1.163 | [passed](program_peer/metal__f32__topk__large__strided_axis1_r128_c257_k256_smallest.json) | 2.911 / 1.333 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
