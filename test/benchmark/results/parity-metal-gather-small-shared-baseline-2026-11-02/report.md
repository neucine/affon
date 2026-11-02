# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `563fbdde4c20083b48261c9821a65b214b5e778010dd0484835779d19c31bb53`  
Binary SHA-256: `05fcf9eb53831e8074b17d9e9205add5d013a4c4dcd00dc0514f36f0e1eebfa1`  
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
| `metal/f32/gather/strided/axis0_r16_c129` | indexing | 16 × 129 | strided | correct | [passed](operator_peer/metal__f32__gather__strided__axis0_r16_c129.json) | 0.3745 / 0.2258 | [passed](program_peer/metal__f32__gather__strided__axis0_r16_c129.json) | 0.4252 / 0.2974 |
| `metal/f32/gather/strided/axis1_r16_c129` | indexing | 16 × 129 | strided | correct | [passed](operator_peer/metal__f32__gather__strided__axis1_r16_c129.json) | 0.4036 / 0.2241 | [passed](program_peer/metal__f32__gather__strided__axis1_r16_c129.json) | 0.3972 / 0.2312 |
| `metal/f32/gather/large/offset_axis0_r16_c129` | indexing | 16 × 129 | offset | correct | [passed](operator_peer/metal__f32__gather__large__offset_axis0_r16_c129.json) | 0.4471 / 0.2493 | [passed](program_peer/metal__f32__gather__large__offset_axis0_r16_c129.json) | 0.4791 / 0.2796 |
| `metal/f32/gather/large/offset_axis1_r16_c129` | indexing | 16 × 129 | offset | correct | [passed](operator_peer/metal__f32__gather__large__offset_axis1_r16_c129.json) | 0.4936 / 0.2403 | [passed](program_peer/metal__f32__gather__large__offset_axis1_r16_c129.json) | 0.4951 / 0.2859 |
| `metal/bf16/gather/strided/axis0_r16_c129` | indexing | 16 × 129 | strided | correct | [passed](operator_peer/metal__bf16__gather__strided__axis0_r16_c129.json) | 0.4679 / 0.2409 | [passed](program_peer/metal__bf16__gather__strided__axis0_r16_c129.json) | 0.4083 / 0.2329 |
| `metal/bf16/gather/strided/axis1_r16_c129` | indexing | 16 × 129 | strided | correct | [passed](operator_peer/metal__bf16__gather__strided__axis1_r16_c129.json) | 0.4524 / 0.2452 | [passed](program_peer/metal__bf16__gather__strided__axis1_r16_c129.json) | 0.5014 / 0.2534 |
| `metal/bf16/gather/large/offset_axis0_r16_c129` | indexing | 16 × 129 | offset | correct | [passed](operator_peer/metal__bf16__gather__large__offset_axis0_r16_c129.json) | 0.4448 / 0.2402 | [passed](program_peer/metal__bf16__gather__large__offset_axis0_r16_c129.json) | 0.4897 / 0.2676 |
| `metal/bf16/gather/large/offset_axis1_r16_c129` | indexing | 16 × 129 | offset | correct | [passed](operator_peer/metal__bf16__gather__large__offset_axis1_r16_c129.json) | 0.4345 / 0.2411 | [passed](program_peer/metal__bf16__gather__large__offset_axis1_r16_c129.json) | 0.3952 / 0.2202 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
