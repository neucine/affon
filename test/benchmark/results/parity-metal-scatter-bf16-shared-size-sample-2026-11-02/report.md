# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `2dab6e1f0d45405afddb6a925ce6fba0463907355d930fc41c24eeb32109ec27`  
Binary SHA-256: `fb745109f15ac1faa5b2de4f09c9f58bcbf2a4b2bc9217c128578c34218a9ffa`  
Timing: 10 iterations × 5 samples × 3 rounds

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
| `metal/bf16/scatter_add/small/axis0_r16_c64` | indexing | 16 × 64 | dense | correct | [passed](operator_peer/metal__bf16__scatter_add__small__axis0_r16_c64.json) | 0.0274 / 0.2467 | [passed](program_peer/metal__bf16__scatter_add__small__axis0_r16_c64.json) | 0.2646 / 0.2364 |
| `metal/bf16/scatter_add/small/axis0_r33_c64` | indexing | 33 × 64 | dense | correct | [passed](operator_peer/metal__bf16__scatter_add__small__axis0_r33_c64.json) | 0.0467 / 0.2947 | [passed](program_peer/metal__bf16__scatter_add__small__axis0_r33_c64.json) | 0.2935 / 0.268 |
| `metal/bf16/scatter_add/small/axis1_r16_c64` | indexing | 16 × 64 | dense | correct | [passed](operator_peer/metal__bf16__scatter_add__small__axis1_r16_c64.json) | 0.0268 / 0.2361 | [passed](program_peer/metal__bf16__scatter_add__small__axis1_r16_c64.json) | 0.292 / 0.2395 |
| `metal/bf16/scatter_add/small/axis1_r33_c64` | indexing | 33 × 64 | dense | correct | [passed](operator_peer/metal__bf16__scatter_add__small__axis1_r33_c64.json) | 0.0453 / 0.3254 | [passed](program_peer/metal__bf16__scatter_add__small__axis1_r33_c64.json) | 0.2984 / 0.25 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
