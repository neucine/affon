# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `d639b2a3291fb1ab74bf19934e5b10150570b124032a9392350f9ef829c40e7a`  
Binary SHA-256: `d453dae33ce53f0a3a2099ae33b469ee9239b73696fcd0a02dbae903c3d9f881`  
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
| `metal/f32/where/scalar` | comparison | — | dense | correct | [passed](operator_peer/metal__f32__where__scalar.json) | 0.2202 / 0.2096 | [passed](program_peer/metal__f32__where__scalar.json) | 0.2331 / 0.2162 |
| `metal/f32/where/ragged` | comparison | 17 × 65 | dense | correct | [passed](operator_peer/metal__f32__where__ragged.json) | 0.2589 / 0.2222 | [passed](program_peer/metal__f32__where__ragged.json) | 0.2761 / 0.2306 |
| `metal/f32/where/broadcast/r32_c32` | comparison | 32 × 32 | dense | correct | [passed](operator_peer/metal__f32__where__broadcast__r32_c32.json) | 0.2625 / 0.2232 | [passed](program_peer/metal__f32__where__broadcast__r32_c32.json) | 0.2856 / 0.2284 |
| `metal/f32/where/broadcast/r64_c128` | comparison | 64 × 128 | dense | correct | [passed](operator_peer/metal__f32__where__broadcast__r64_c128.json) | 0.3213 / 0.2978 | [passed](program_peer/metal__f32__where__broadcast__r64_c128.json) | 0.3536 / 0.2935 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
