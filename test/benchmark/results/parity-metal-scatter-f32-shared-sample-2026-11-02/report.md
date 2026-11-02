# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `a87c28ec6b883e7a0aca8b80806781bfa03c2430a06f1986e5b4211153e8acf2`  
Binary SHA-256: `14c344bbddf697b39a01cc115c45f11f131a0e9384b540118bf06c10a5754b94`  
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
| `metal/f32/scatter_add/small/axis0_r3_c3` | indexing | 3 × 3 | dense | correct | [passed](operator_peer/metal__f32__scatter_add__small__axis0_r3_c3.json) | 0.0102 / 0.3112 | [passed](program_peer/metal__f32__scatter_add__small__axis0_r3_c3.json) | 0.2222 / 0.2786 |
| `metal/f32/scatter_add/small/axis0_r3_c17` | indexing | 3 × 17 | dense | correct | [passed](operator_peer/metal__f32__scatter_add__small__axis0_r3_c17.json) | 0.0112 / 0.2446 | [passed](program_peer/metal__f32__scatter_add__small__axis0_r3_c17.json) | 0.2014 / 0.2205 |
| `metal/f32/scatter_add/small/axis1_r33_c64` | indexing | 33 × 64 | dense | correct | [passed](operator_peer/metal__f32__scatter_add__small__axis1_r33_c64.json) | 0.0473 / 0.2751 | [passed](program_peer/metal__f32__scatter_add__small__axis1_r33_c64.json) | 0.293 / 0.2441 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
