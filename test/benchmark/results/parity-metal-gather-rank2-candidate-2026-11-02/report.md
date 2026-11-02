# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `563fbdde4c20083b48261c9821a65b214b5e778010dd0484835779d19c31bb53`  
Binary SHA-256: `bb96ef8d6b04b6f0ddf148c1beeddc71796b309a6753bb59d34ebc6b8022109e`  
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
| `metal/f32/gather/strided/axis0_r16_c129` | indexing | 16 × 129 | strided | correct | [passed](operator_peer/metal__f32__gather__strided__axis0_r16_c129.json) | 0.4962 / 0.238 | [passed](program_peer/metal__f32__gather__strided__axis0_r16_c129.json) | 0.4976 / 0.2282 |
| `metal/f32/gather/strided/axis1_r16_c129` | indexing | 16 × 129 | strided | correct | [passed](operator_peer/metal__f32__gather__strided__axis1_r16_c129.json) | 0.3852 / 0.2253 | [passed](program_peer/metal__f32__gather__strided__axis1_r16_c129.json) | 0.5108 / 0.2339 |
| `metal/f32/gather/large/offset_axis0_r16_c129` | indexing | 16 × 129 | offset | correct | [passed](operator_peer/metal__f32__gather__large__offset_axis0_r16_c129.json) | 0.3968 / 0.2349 | [passed](program_peer/metal__f32__gather__large__offset_axis0_r16_c129.json) | 0.4234 / 0.2354 |
| `metal/f32/gather/large/offset_axis1_r16_c129` | indexing | 16 × 129 | offset | correct | [passed](operator_peer/metal__f32__gather__large__offset_axis1_r16_c129.json) | 0.4906 / 0.2289 | [passed](program_peer/metal__f32__gather__large__offset_axis1_r16_c129.json) | 0.3984 / 0.235 |
| `metal/bf16/gather/strided/axis0_r16_c129` | indexing | 16 × 129 | strided | correct | [passed](operator_peer/metal__bf16__gather__strided__axis0_r16_c129.json) | 0.4956 / 0.2333 | [passed](program_peer/metal__bf16__gather__strided__axis0_r16_c129.json) | 0.4296 / 0.2335 |
| `metal/bf16/gather/strided/axis1_r16_c129` | indexing | 16 × 129 | strided | correct | [passed](operator_peer/metal__bf16__gather__strided__axis1_r16_c129.json) | 0.4418 / 0.2269 | [passed](program_peer/metal__bf16__gather__strided__axis1_r16_c129.json) | 0.507 / 0.23 |
| `metal/bf16/gather/large/offset_axis0_r16_c129` | indexing | 16 × 129 | offset | correct | [passed](operator_peer/metal__bf16__gather__large__offset_axis0_r16_c129.json) | 0.497 / 0.2342 | [passed](program_peer/metal__bf16__gather__large__offset_axis0_r16_c129.json) | 0.5114 / 0.2322 |
| `metal/bf16/gather/large/offset_axis1_r16_c129` | indexing | 16 × 129 | offset | correct | [passed](operator_peer/metal__bf16__gather__large__offset_axis1_r16_c129.json) | 0.3908 / 0.2266 | [passed](program_peer/metal__bf16__gather__large__offset_axis1_r16_c129.json) | 0.3924 / 0.2323 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
