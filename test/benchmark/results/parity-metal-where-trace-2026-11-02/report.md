# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `f943f8d4a34dbce2f3c3bbfb295e3a38d4995c092581e9e8724251b29765176e`  
Binary SHA-256: `d453dae33ce53f0a3a2099ae33b469ee9239b73696fcd0a02dbae903c3d9f881`  
Timing: 1 iterations × 3 samples × 2 rounds

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
| `metal/f32/where/small_dense/grid_r16_c129` | comparison | 16 × 129 | dense | correct | [passed](operator_peer/metal__f32__where__small_dense__grid_r16_c129.json) | 0.4095 / 0.3857 | [passed](program_peer/metal__f32__where__small_dense__grid_r16_c129.json) | 0.3975 / 0.2519 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
