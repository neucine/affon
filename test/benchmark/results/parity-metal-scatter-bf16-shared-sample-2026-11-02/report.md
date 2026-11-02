# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `f204d384ffa5a091c0712d94722742b7a58910a80355d50c92cb88360201defb`  
Binary SHA-256: `fb745109f15ac1faa5b2de4f09c9f58bcbf2a4b2bc9217c128578c34218a9ffa`  
Timing: 5 iterations × 3 samples × 2 rounds

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
| `metal/bf16/scatter_add/small/axis0_r3_c3` | indexing | 3 × 3 | dense | correct | [passed](operator_peer/metal__bf16__scatter_add__small__axis0_r3_c3.json) | 0.0102 / 0.2759 | [passed](program_peer/metal__bf16__scatter_add__small__axis0_r3_c3.json) | 0.2466 / 0.262 |
| `metal/bf16/scatter_add/small/axis1_r33_c129` | indexing | 33 × 129 | dense | correct | [passed](operator_peer/metal__bf16__scatter_add__small__axis1_r33_c129.json) | 0.5415 / 0.2695 | [passed](program_peer/metal__bf16__scatter_add__small__axis1_r33_c129.json) | 0.7272 / 0.2831 |
| `metal/bf16/scatter_add/large/offset_axis0_r33_c129` | indexing | 33 × 129 | offset | correct | [passed](operator_peer/metal__bf16__scatter_add__large__offset_axis0_r33_c129.json) | 0.6213 / 0.2858 | [passed](program_peer/metal__bf16__scatter_add__large__offset_axis0_r33_c129.json) | 0.6337 / 0.2944 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
