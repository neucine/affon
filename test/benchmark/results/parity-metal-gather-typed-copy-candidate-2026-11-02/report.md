# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `563fbdde4c20083b48261c9821a65b214b5e778010dd0484835779d19c31bb53`  
Binary SHA-256: `f56fdb3d2cf6a96452fac4cb790067c5177d64382cd042bccad8e2f76fee9b1b`  
Timing: 5 iterations × 5 samples × 3 rounds

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
| `metal/f32/gather/strided/axis0_r16_c129` | indexing | 16 × 129 | strided | correct | [passed](operator_peer/metal__f32__gather__strided__axis0_r16_c129.json) | 0.4004 / 0.2657 | [passed](program_peer/metal__f32__gather__strided__axis0_r16_c129.json) | 0.4222 / 0.2742 |
| `metal/f32/gather/strided/axis1_r16_c129` | indexing | 16 × 129 | strided | correct | [passed](operator_peer/metal__f32__gather__strided__axis1_r16_c129.json) | 0.496 / 0.232 | [passed](program_peer/metal__f32__gather__strided__axis1_r16_c129.json) | 0.383 / 0.232 |
| `metal/f32/gather/large/offset_axis0_r16_c129` | indexing | 16 × 129 | offset | correct | [passed](operator_peer/metal__f32__gather__large__offset_axis0_r16_c129.json) | 0.4836 / 0.2255 | [passed](program_peer/metal__f32__gather__large__offset_axis0_r16_c129.json) | 0.4304 / 0.2367 |
| `metal/f32/gather/large/offset_axis1_r16_c129` | indexing | 16 × 129 | offset | correct | [passed](operator_peer/metal__f32__gather__large__offset_axis1_r16_c129.json) | 0.4996 / 0.2329 | [passed](program_peer/metal__f32__gather__large__offset_axis1_r16_c129.json) | 0.3974 / 0.226 |
| `metal/bf16/gather/strided/axis0_r16_c129` | indexing | 16 × 129 | strided | correct | [passed](operator_peer/metal__bf16__gather__strided__axis0_r16_c129.json) | 0.379 / 0.225 | [passed](program_peer/metal__bf16__gather__strided__axis0_r16_c129.json) | 0.4828 / 0.2326 |
| `metal/bf16/gather/strided/axis1_r16_c129` | indexing | 16 × 129 | strided | correct | [passed](operator_peer/metal__bf16__gather__strided__axis1_r16_c129.json) | 0.507 / 0.229 | [passed](program_peer/metal__bf16__gather__strided__axis1_r16_c129.json) | 0.427 / 0.2341 |
| `metal/bf16/gather/large/offset_axis0_r16_c129` | indexing | 16 × 129 | offset | correct | [passed](operator_peer/metal__bf16__gather__large__offset_axis0_r16_c129.json) | 0.5028 / 0.233 | [passed](program_peer/metal__bf16__gather__large__offset_axis0_r16_c129.json) | 0.3924 / 0.2377 |
| `metal/bf16/gather/large/offset_axis1_r16_c129` | indexing | 16 × 129 | offset | correct | [passed](operator_peer/metal__bf16__gather__large__offset_axis1_r16_c129.json) | 0.3768 / 0.2181 | [passed](program_peer/metal__bf16__gather__large__offset_axis1_r16_c129.json) | 0.3968 / 0.2333 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
