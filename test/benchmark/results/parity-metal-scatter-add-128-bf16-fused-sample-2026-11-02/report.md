# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `53526b13dc5cfac15734d964bfd26373dfaec31100175b0be4fbb6327e88ee52`  
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
| `metal/bf16/scatter_add/small/axis0_r3_c3` | indexing | 3 × 3 | dense | correct | [passed](operator_peer/metal__bf16__scatter_add__small__axis0_r3_c3.json) | 0.3516 / 0.2288 | [passed](program_peer/metal__bf16__scatter_add__small__axis0_r3_c3.json) | 0.2951 / 0.2062 |
| `metal/bf16/scatter_add/small/axis1_r7_c17` | indexing | 7 × 17 | dense | correct | [passed](operator_peer/metal__bf16__scatter_add__small__axis1_r7_c17.json) | 0.3594 / 0.2243 | [passed](program_peer/metal__bf16__scatter_add__small__axis1_r7_c17.json) | 0.3658 / 0.2266 |
| `metal/bf16/scatter_add/large/offset_axis0_r16_c64` | indexing | 16 × 64 | offset | correct | [passed](operator_peer/metal__bf16__scatter_add__large__offset_axis0_r16_c64.json) | 0.3775 / 0.2385 | [passed](program_peer/metal__bf16__scatter_add__large__offset_axis0_r16_c64.json) | 0.4014 / 0.2402 |
| `metal/bf16/scatter_add/large/offset_axis1_r33_c129` | indexing | 33 × 129 | offset | correct | [passed](operator_peer/metal__bf16__scatter_add__large__offset_axis1_r33_c129.json) | 0.4385 / 0.2621 | [passed](program_peer/metal__bf16__scatter_add__large__offset_axis1_r33_c129.json) | 0.5085 / 0.2617 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
