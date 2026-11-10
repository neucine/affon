# Parity batch report

Target: `NVIDIA GeForce RTX 3090/7.0.0-38-generic/x86_64`  
Catalog SHA-256: `7d3a1320e0b4a81b88f645194d6e98f9eb38ef561eabf926e811dc81aa29605c`  
Binary SHA-256: `93e0d52c41e6b29c66f08e7915492197c6a086ae1845644e2f85e5f5c5b57a18`  
Timing: 10 iterations × 5 samples × 3 rounds

## Checks

- Cases run: 1
- Numerical check passed: 1
- Operator check and timing passed: 1
- Program check and timing passed: 1
- Numerical failures: 0
- Operator failures: 0
- Program failures: 0
- Operator checks pending: 0
- Program checks pending: 0

## Cases

| Case | Family | Shape | Layout | Numerical | Operator | Affon / PyTorch ms | Program | Affon / PyTorch ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `cuda/f32/scatter_add/batched/dense_axis2_b4_r7_c65_n32` | indexing | 4 × 7 × 65 | dense | correct | [passed](operator_peer/cuda__f32__scatter_add__batched__dense_axis2_b4_r7_c65_n32.json) | 0.06938 / 0.05949 | [passed](program_peer/cuda__f32__scatter_add__batched__dense_axis2_b4_r7_c65_n32.json) | 0.07519 / 0.06625 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
