# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `563fbdde4c20083b48261c9821a65b214b5e778010dd0484835779d19c31bb53`  
Binary SHA-256: `5be7468dbd2419582c4ea8118a0c9de0adb843bdf1d69efd17968bf64c02152c`  
Timing: 10 iterations × 5 samples × 3 rounds

## Checks

- Cases run: 8
- Numerical check passed: 8
- Operator check and timing passed: 8
- Program check and timing passed: 8
- Numerical failures: 0
- Operator failures: 0
- Program failures: 0

## Cases

| Case | Family | Shape | Layout | Numerical | Operator | Affon / PyTorch ms | Program | Affon / PyTorch ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `metal/f32/gather/strided/axis0_r16_c129` | indexing | 16 × 129 | strided | correct | [passed](operator_peer/metal__f32__gather__strided__axis0_r16_c129.json) | 0.0244 / 0.2821 | [passed](program_peer/metal__f32__gather__strided__axis0_r16_c129.json) | 0.2202 / 0.2359 |
| `metal/f32/gather/strided/axis1_r16_c129` | indexing | 16 × 129 | strided | correct | [passed](operator_peer/metal__f32__gather__strided__axis1_r16_c129.json) | 0.0271 / 0.2641 | [passed](program_peer/metal__f32__gather__strided__axis1_r16_c129.json) | 0.2245 / 0.2382 |
| `metal/f32/gather/large/offset_axis0_r16_c129` | indexing | 16 × 129 | offset | correct | [passed](operator_peer/metal__f32__gather__large__offset_axis0_r16_c129.json) | 0.0246 / 0.2578 | [passed](program_peer/metal__f32__gather__large__offset_axis0_r16_c129.json) | 0.219 / 0.2344 |
| `metal/f32/gather/large/offset_axis1_r16_c129` | indexing | 16 × 129 | offset | correct | [passed](operator_peer/metal__f32__gather__large__offset_axis1_r16_c129.json) | 0.0276 / 0.2411 | [passed](program_peer/metal__f32__gather__large__offset_axis1_r16_c129.json) | 0.2228 / 0.2337 |
| `metal/bf16/gather/strided/axis0_r16_c129` | indexing | 16 × 129 | strided | correct | [passed](operator_peer/metal__bf16__gather__strided__axis0_r16_c129.json) | 0.0248 / 0.2346 | [passed](program_peer/metal__bf16__gather__strided__axis0_r16_c129.json) | 0.2436 / 0.2359 |
| `metal/bf16/gather/strided/axis1_r16_c129` | indexing | 16 × 129 | strided | correct | [passed](operator_peer/metal__bf16__gather__strided__axis1_r16_c129.json) | 0.0275 / 0.2484 | [passed](program_peer/metal__bf16__gather__strided__axis1_r16_c129.json) | 0.2749 / 0.2856 |
| `metal/bf16/gather/large/offset_axis0_r16_c129` | indexing | 16 × 129 | offset | correct | [passed](operator_peer/metal__bf16__gather__large__offset_axis0_r16_c129.json) | 0.0251 / 0.2686 | [passed](program_peer/metal__bf16__gather__large__offset_axis0_r16_c129.json) | 0.2312 / 0.2815 |
| `metal/bf16/gather/large/offset_axis1_r16_c129` | indexing | 16 × 129 | offset | correct | [passed](operator_peer/metal__bf16__gather__large__offset_axis1_r16_c129.json) | 0.0267 / 0.2701 | [passed](program_peer/metal__bf16__gather__large__offset_axis1_r16_c129.json) | 0.223 / 0.2723 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
