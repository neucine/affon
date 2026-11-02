# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `f48c86ff8b00f0cd826eabc34ff67cda451db726f1db2ee7f2888142bab85787`  
Binary SHA-256: `800452284fd19dac2259d669f92b921538f8fd26a3d6714bcd73f4fd8aae8934`  
Timing: 5 iterations × 3 samples × 2 rounds

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
| `metal/f32/gather/strided/axis0_r2_c7` | indexing | 2 × 7 | strided | correct | [passed](operator_peer/metal__f32__gather__strided__axis0_r2_c7.json) | 0.487 / 0.247 | [passed](program_peer/metal__f32__gather__strided__axis0_r2_c7.json) | 0.4339 / 0.2563 |
| `metal/f32/gather/large/offset_axis0_r2_c7` | indexing | 2 × 7 | offset | correct | [passed](operator_peer/metal__f32__gather__large__offset_axis0_r2_c7.json) | 0.4794 / 0.2087 | [passed](program_peer/metal__f32__gather__large__offset_axis0_r2_c7.json) | 0.4827 / 0.2174 |
| `metal/bf16/gather/strided/axis0_r2_c7` | indexing | 2 × 7 | strided | correct | [passed](operator_peer/metal__bf16__gather__strided__axis0_r2_c7.json) | 0.4701 / 0.2113 | [passed](program_peer/metal__bf16__gather__strided__axis0_r2_c7.json) | 0.5029 / 0.2201 |
| `metal/bf16/gather/large/offset_axis0_r2_c7` | indexing | 2 × 7 | offset | correct | [passed](operator_peer/metal__bf16__gather__large__offset_axis0_r2_c7.json) | 0.4808 / 0.2181 | [passed](program_peer/metal__bf16__gather__large__offset_axis0_r2_c7.json) | 0.4912 / 0.2165 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
