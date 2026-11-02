# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `9dae26d7158ddf3c159afd92cf82e18e103cd6c546290dc918fadf14a8f075c2`  
Binary SHA-256: `247ede0f4b7fc7ef84b007a4f1db1ac645e9156a5a33a2fe478ee218e2ea144a`  
Timing: 10 iterations × 5 samples × 3 rounds

## Checks

- Cases run: 1
- Numerical check passed: 1
- Operator check and timing passed: 1
- Program check and timing passed: 1
- Numerical failures: 0
- Operator failures: 0
- Program failures: 0

## Cases

| Case | Family | Shape | Layout | Numerical | Operator | Affon / PyTorch ms | Program | Affon / PyTorch ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `metal/bf16/scatter_add/small/axis0_r3_c3` | indexing | 3 × 3 | dense | correct | [passed](operator_peer/metal__bf16__scatter_add__small__axis0_r3_c3.json) | 0.2429 / 0.2867 | [passed](program_peer/metal__bf16__scatter_add__small__axis0_r3_c3.json) | 0.2581 / 0.293 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
