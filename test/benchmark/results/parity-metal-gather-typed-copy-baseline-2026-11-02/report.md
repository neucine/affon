# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `563fbdde4c20083b48261c9821a65b214b5e778010dd0484835779d19c31bb53`  
Binary SHA-256: `cffe62b8e4546a85edf7b842a6d443b2953546d0d6de9192bafca166f70513f1`  
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
| `metal/f32/gather/strided/axis0_r16_c129` | indexing | 16 × 129 | strided | correct | [passed](operator_peer/metal__f32__gather__strided__axis0_r16_c129.json) | 0.4872 / 0.238 | [passed](program_peer/metal__f32__gather__strided__axis0_r16_c129.json) | 0.4294 / 0.2236 |
| `metal/f32/gather/strided/axis1_r16_c129` | indexing | 16 × 129 | strided | correct | [passed](operator_peer/metal__f32__gather__strided__axis1_r16_c129.json) | 0.38 / 0.2265 | [passed](program_peer/metal__f32__gather__strided__axis1_r16_c129.json) | 0.516 / 0.2324 |
| `metal/f32/gather/large/offset_axis0_r16_c129` | indexing | 16 × 129 | offset | correct | [passed](operator_peer/metal__f32__gather__large__offset_axis0_r16_c129.json) | 0.38 / 0.2332 | [passed](program_peer/metal__f32__gather__large__offset_axis0_r16_c129.json) | 0.5002 / 0.2289 |
| `metal/f32/gather/large/offset_axis1_r16_c129` | indexing | 16 × 129 | offset | correct | [passed](operator_peer/metal__f32__gather__large__offset_axis1_r16_c129.json) | 0.3838 / 0.228 | [passed](program_peer/metal__f32__gather__large__offset_axis1_r16_c129.json) | 0.4226 / 0.2329 |
| `metal/bf16/gather/strided/axis0_r16_c129` | indexing | 16 × 129 | strided | correct | [passed](operator_peer/metal__bf16__gather__strided__axis0_r16_c129.json) | 0.5028 / 0.2314 | [passed](program_peer/metal__bf16__gather__strided__axis0_r16_c129.json) | 0.454 / 0.2379 |
| `metal/bf16/gather/strided/axis1_r16_c129` | indexing | 16 × 129 | strided | correct | [passed](operator_peer/metal__bf16__gather__strided__axis1_r16_c129.json) | 0.509 / 0.2317 | [passed](program_peer/metal__bf16__gather__strided__axis1_r16_c129.json) | 0.4056 / 0.2333 |
| `metal/bf16/gather/large/offset_axis0_r16_c129` | indexing | 16 × 129 | offset | correct | [passed](operator_peer/metal__bf16__gather__large__offset_axis0_r16_c129.json) | 0.5056 / 0.2338 | [passed](program_peer/metal__bf16__gather__large__offset_axis0_r16_c129.json) | 0.5222 / 0.2345 |
| `metal/bf16/gather/large/offset_axis1_r16_c129` | indexing | 16 × 129 | offset | correct | [passed](operator_peer/metal__bf16__gather__large__offset_axis1_r16_c129.json) | 0.4644 / 0.2271 | [passed](program_peer/metal__bf16__gather__large__offset_axis1_r16_c129.json) | 0.3956 / 0.2316 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
