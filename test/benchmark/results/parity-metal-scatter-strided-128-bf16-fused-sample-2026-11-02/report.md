# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `7b95a40024e169d97f18ba6d14e167ff4298cfaae3a4a34a147cacac428679b1`  
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
| `metal/bf16/scatter_add/strided/rows_axis0_r3_c3` | indexing | 3 × 3 | strided | correct | [passed](operator_peer/metal__bf16__scatter_add__strided__rows_axis0_r3_c3.json) | 0.3632 / 0.2354 | [passed](program_peer/metal__bf16__scatter_add__strided__rows_axis0_r3_c3.json) | 0.3717 / 0.2397 |
| `metal/bf16/scatter_add/strided/cols_axis0_r7_c17` | indexing | 7 × 17 | strided | correct | [passed](operator_peer/metal__bf16__scatter_add__strided__cols_axis0_r7_c17.json) | 0.3767 / 0.2371 | [passed](program_peer/metal__bf16__scatter_add__strided__cols_axis0_r7_c17.json) | 0.3777 / 0.236 |
| `metal/bf16/scatter_add/strided/rows_axis1_r16_c64` | indexing | 16 × 64 | strided | correct | [passed](operator_peer/metal__bf16__scatter_add__strided__rows_axis1_r16_c64.json) | 0.3994 / 0.2362 | [passed](program_peer/metal__bf16__scatter_add__strided__rows_axis1_r16_c64.json) | 0.4228 / 0.2447 |
| `metal/bf16/scatter_add/strided/cols_axis1_r33_c129` | indexing | 33 × 129 | strided | correct | [passed](operator_peer/metal__bf16__scatter_add__strided__cols_axis1_r33_c129.json) | 0.4521 / 0.2672 | [passed](program_peer/metal__bf16__scatter_add__strided__cols_axis1_r33_c129.json) | 0.4634 / 0.2772 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
