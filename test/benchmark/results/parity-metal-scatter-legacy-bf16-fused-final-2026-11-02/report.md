# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `3ba0c6e92051f77cd22c4ba5f0a42dfeb06680e7834d5f946ddfd7a3f5775a84`  
Binary SHA-256: `570305d7d67335f0cedd0c79b09ac0c1ac3c5c092a1cca92f426fe12e23eb9ff`  
Timing: 30 iterations × 5 samples × 3 rounds

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
| `metal/bf16/scatter_add/small` | indexing | 3 × 3 | dense | correct | [passed](operator_peer/metal__bf16__scatter_add__small.json) | 0.3548 / 0.2274 | [passed](program_peer/metal__bf16__scatter_add__small.json) | 0.3639 / 0.2382 |
| `metal/bf16/scatter_add/large` | indexing | 3 × 3 | dense | correct | [passed](operator_peer/metal__bf16__scatter_add__large.json) | 0.3472 / 0.2316 | [passed](program_peer/metal__bf16__scatter_add__large.json) | 0.3507 / 0.2353 |
| `metal/bf16/scatter_add/strided` | indexing | 3 × 3 | strided | correct | [passed](operator_peer/metal__bf16__scatter_add__strided.json) | 0.3603 / 0.2306 | [passed](program_peer/metal__bf16__scatter_add__strided.json) | 0.3658 / 0.2369 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
