# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `2026e6efb8d84bef025441769b252c39b6aac7ffa06e8718a040a3fe106c8108`  
Binary SHA-256: `1263f72409ea8b8e7c7c5ea44ec4bc78639d602290b1885391b28c52332c724e`  
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
| `metal/f32/layer_norm/strided/inner_strided_r3_c5` | normalization | 3 × 5 | strided | correct | [passed](operator_peer/metal__f32__layer_norm__strided__inner_strided_r3_c5.json) | 0.2502 / 0.1986 | [passed](program_peer/metal__f32__layer_norm__strided__inner_strided_r3_c5.json) | 0.2518 / 0.1946 |
| `metal/f32/layer_norm/strided/inner_strided_r16_c32` | normalization | 16 × 32 | strided | correct | [passed](operator_peer/metal__f32__layer_norm__strided__inner_strided_r16_c32.json) | 0.3044 / 0.2014 | [passed](program_peer/metal__f32__layer_norm__strided__inner_strided_r16_c32.json) | 0.3034 / 0.204 |
| `metal/f32/layer_norm/strided/inner_offset_r3_c5` | normalization | 3 × 5 | offset | correct | [passed](operator_peer/metal__f32__layer_norm__strided__inner_offset_r3_c5.json) | 0.2486 / 0.2338 | [passed](program_peer/metal__f32__layer_norm__strided__inner_offset_r3_c5.json) | 0.2504 / 0.1801 |
| `metal/f32/layer_norm/strided/inner_offset_r16_c32` | normalization | 16 × 32 | offset | correct | [passed](operator_peer/metal__f32__layer_norm__strided__inner_offset_r16_c32.json) | 0.315 / 0.2034 | [passed](program_peer/metal__f32__layer_norm__strided__inner_offset_r16_c32.json) | 0.3182 / 0.2005 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
