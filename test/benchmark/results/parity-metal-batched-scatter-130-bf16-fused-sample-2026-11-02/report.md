# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `0d32322b1833cb17b6b5adaa05926d2c4f4606d0f1df21f5ea846fae43ebb668`  
Binary SHA-256: `570305d7d67335f0cedd0c79b09ac0c1ac3c5c092a1cca92f426fe12e23eb9ff`  
Timing: 30 iterations × 5 samples × 3 rounds

## Checks

- Cases run: 4
- Numerical check passed: 4
- Operator check and timing passed: 4
- Program check and timing passed: 4
- Numerical failures: 0
- Operator failures: 0
- Program failures: 0

## Cases

| Case | Family | Shape | Layout | Numerical | Operator | Affon / PyTorch ms | Program | Affon / PyTorch ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `metal/bf16/scatter_add/batched` | indexing | 3 × 7 × 11 | dense | correct | [passed](operator_peer/metal__bf16__scatter_add__batched.json) | 0.3642 / 0.2378 | [passed](program_peer/metal__bf16__scatter_add__batched.json) | 0.4001 / 0.2412 |
| `metal/bf16/scatter_add/batched/dense_axis1_b4_r7_c5_n3` | indexing | 4 × 7 × 5 | dense | correct | [passed](operator_peer/metal__bf16__scatter_add__batched__dense_axis1_b4_r7_c5_n3.json) | 0.3514 / 0.2392 | [passed](program_peer/metal__bf16__scatter_add__batched__dense_axis1_b4_r7_c5_n3.json) | 0.3802 / 0.2382 |
| `metal/bf16/scatter_add/batched/dense_axis2_b2_r16_c65_n32` | indexing | 2 × 16 × 65 | dense | correct | [passed](operator_peer/metal__bf16__scatter_add__batched__dense_axis2_b2_r16_c65_n32.json) | 0.4587 / 0.3059 | [passed](program_peer/metal__bf16__scatter_add__batched__dense_axis2_b2_r16_c65_n32.json) | 0.4415 / 0.2446 |
| `metal/bf16/scatter_add/batched/dense_axis2_b4_r33_c129_n64` | indexing | 4 × 33 × 129 | dense | correct | [passed](operator_peer/metal__bf16__scatter_add__batched__dense_axis2_b4_r33_c129_n64.json) | 0.6706 / 0.4777 | [passed](program_peer/metal__bf16__scatter_add__batched__dense_axis2_b4_r33_c129_n64.json) | 0.7032 / 0.4421 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
