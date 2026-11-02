# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `2f9b7ae53acd0e9262eb911cc41aef8bffba1bf27479febc2a2fed4a9666475b`  
Binary SHA-256: `800452284fd19dac2259d669f92b921538f8fd26a3d6714bcd73f4fd8aae8934`  
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
| `metal/f32/gather/large/axis0_32x64` | indexing | 32 × 64 | dense | correct | [passed](operator_peer/metal__f32__gather__large__axis0_32x64.json) | 0.4205 / 0.2433 | [passed](program_peer/metal__f32__gather__large__axis0_32x64.json) | 0.4139 / 0.2409 |
| `metal/bf16/gather/large/axis0_32x64` | indexing | 32 × 64 | dense | correct | [passed](operator_peer/metal__bf16__gather__large__axis0_32x64.json) | 0.4202 / 0.239 | [passed](program_peer/metal__bf16__gather__large__axis0_32x64.json) | 0.4354 / 0.2435 |
| `metal/f32/gather/strided/r64_c127_i32` | indexing | 64 × 127 | strided | correct | [passed](operator_peer/metal__f32__gather__strided__r64_c127_i32.json) | 0.4176 / 0.2614 | [passed](program_peer/metal__f32__gather__strided__r64_c127_i32.json) | 0.422 / 0.2662 |
| `metal/bf16/gather/strided/r64_c127_i32` | indexing | 64 × 127 | strided | correct | [passed](operator_peer/metal__bf16__gather__strided__r64_c127_i32.json) | 0.4183 / 0.265 | [passed](program_peer/metal__bf16__gather__strided__r64_c127_i32.json) | 0.4492 / 0.2614 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
