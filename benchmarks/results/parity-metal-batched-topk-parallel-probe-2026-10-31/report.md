# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `ce38d82c126d130f40663214819cce78b47fe718be08b9a84b8ab07de25639eb`  
Binary SHA-256: `caa71ec6ea98005c433e270979a30a1f097a30c25835d18dc74edbe829c525a9`  
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
| `metal/f32/topk/batched/dense_axis2_b4_r7_c129_k64` | indexing | 4 × 7 × 129 | dense | correct | [passed](operator_peer/metal__f32__topk__batched__dense_axis2_b4_r7_c129_k64.json) | 0.402 / 0.4833 | [passed](program_peer/metal__f32__topk__batched__dense_axis2_b4_r7_c129_k64.json) | 0.4008 / 0.4871 |
| `metal/f32/topk/batched/dense_axis2_b4_r33_c129_k64` | indexing | 4 × 33 × 129 | dense | correct | [passed](operator_peer/metal__f32__topk__batched__dense_axis2_b4_r33_c129_k64.json) | 0.9504 / 0.7763 | [passed](program_peer/metal__f32__topk__batched__dense_axis2_b4_r33_c129_k64.json) | 0.928 / 0.7985 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
