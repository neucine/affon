# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `7d43680ec76a861cd707634737921cc2abc52f103ec809d2f19021da25fd7180`  
Binary SHA-256: `5be7468dbd2419582c4ea8118a0c9de0adb843bdf1d69efd17968bf64c02152c`  
Timing: 10 iterations × 5 samples × 3 rounds

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
| `metal/f32/gather/small/r16_w31_i8x31_a0_offset` | indexing | 16 × 31 | offset | correct | [passed](operator_peer/metal__f32__gather__small__r16_w31_i8x31_a0_offset.json) | 0.0142 / 0.2617 | [passed](program_peer/metal__f32__gather__small__r16_w31_i8x31_a0_offset.json) | 0.1902 / 0.211 |
| `metal/f32/gather/large/r128_c127_i64` | indexing | 128 × 127 | dense | correct | [passed](operator_peer/metal__f32__gather__large__r128_c127_i64.json) | 0.4676 / 0.3032 | [passed](program_peer/metal__f32__gather__large__r128_c127_i64.json) | 0.4653 / 0.2997 |
| `metal/bf16/gather/large/r128_c127_i64` | indexing | 128 × 127 | dense | correct | [passed](operator_peer/metal__bf16__gather__large__r128_c127_i64.json) | 0.4717 / 0.3029 | [passed](program_peer/metal__bf16__gather__large__r128_c127_i64.json) | 0.4828 / 0.2993 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
