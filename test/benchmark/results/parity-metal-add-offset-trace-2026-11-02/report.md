# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `24f67951b8007489d353833b7fe830bc4bc3a20a1301ff18eb16e44527dabea5`  
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
| `metal/f32/add/offset/offset_r16_c129` | arithmetic | 16 × 129 | offset | correct | [passed](operator_peer/metal__f32__add__offset__offset_r16_c129.json) | 0.423 / 0.2421 | [passed](program_peer/metal__f32__add__offset__offset_r16_c129.json) | 0.405 / 0.2124 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
