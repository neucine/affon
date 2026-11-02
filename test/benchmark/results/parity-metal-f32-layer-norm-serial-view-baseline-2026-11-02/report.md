# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `2026e6efb8d84bef025441769b252c39b6aac7ffa06e8718a040a3fe106c8108`  
Binary SHA-256: `07d2686a6fde2c671da3d04668ac145d8ca91e78bd6eb55675701480f9735945`  
Timing: 5 iterations × 5 samples × 3 rounds

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
| `metal/f32/layer_norm/strided/inner_strided_r3_c5` | normalization | 3 × 5 | strided | correct | [passed](operator_peer/metal__f32__layer_norm__strided__inner_strided_r3_c5.json) | 0.259 / 0.1987 | [passed](program_peer/metal__f32__layer_norm__strided__inner_strided_r3_c5.json) | 0.308 / 0.1936 |
| `metal/f32/layer_norm/strided/inner_strided_r16_c32` | normalization | 16 × 32 | strided | correct | [passed](operator_peer/metal__f32__layer_norm__strided__inner_strided_r16_c32.json) | 0.5044 / 0.1993 | [passed](program_peer/metal__f32__layer_norm__strided__inner_strided_r16_c32.json) | 0.2946 / 0.2019 |
| `metal/f32/layer_norm/strided/inner_offset_r3_c5` | normalization | 3 × 5 | offset | correct | [passed](operator_peer/metal__f32__layer_norm__strided__inner_offset_r3_c5.json) | 0.2302 / 0.1889 | [passed](program_peer/metal__f32__layer_norm__strided__inner_offset_r3_c5.json) | 0.31 / 0.1944 |
| `metal/f32/layer_norm/strided/inner_offset_r16_c32` | normalization | 16 × 32 | offset | correct | [passed](operator_peer/metal__f32__layer_norm__strided__inner_offset_r16_c32.json) | 0.303 / 0.1956 | [passed](program_peer/metal__f32__layer_norm__strided__inner_offset_r16_c32.json) | 0.4178 / 0.2095 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
