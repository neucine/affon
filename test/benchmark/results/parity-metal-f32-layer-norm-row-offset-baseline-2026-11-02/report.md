# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `906bb65250699f4fa2ec296b809f441413d89319398a398b1ade2959322c4cae`  
Binary SHA-256: `a9a10dd4a4dd6ab577715a114b1261d475aeddc676e7750489ffff1b83de8cef`  
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
| `metal/f32/layer_norm/strided/inner_offset_r3_c5` | normalization | 3 × 5 | offset | correct | [passed](operator_peer/metal__f32__layer_norm__strided__inner_offset_r3_c5.json) | 0.2877 / 0.1847 | [passed](program_peer/metal__f32__layer_norm__strided__inner_offset_r3_c5.json) | 0.3056 / 0.1977 |
| `metal/f32/layer_norm/strided/inner_offset_r3_c65` | normalization | 3 × 65 | offset | correct | [passed](operator_peer/metal__f32__layer_norm__strided__inner_offset_r3_c65.json) | 0.3418 / 0.1882 | [passed](program_peer/metal__f32__layer_norm__strided__inner_offset_r3_c65.json) | 0.3378 / 0.1879 |
| `metal/f32/layer_norm/strided/inner_offset_r16_c5` | normalization | 16 × 5 | offset | correct | [passed](operator_peer/metal__f32__layer_norm__strided__inner_offset_r16_c5.json) | 0.3176 / 0.1894 | [passed](program_peer/metal__f32__layer_norm__strided__inner_offset_r16_c5.json) | 0.3249 / 0.1892 |
| `metal/f32/layer_norm/strided/inner_offset_r16_c32` | normalization | 16 × 32 | offset | correct | [passed](operator_peer/metal__f32__layer_norm__strided__inner_offset_r16_c32.json) | 0.3122 / 0.2035 | [passed](program_peer/metal__f32__layer_norm__strided__inner_offset_r16_c32.json) | 0.3095 / 0.1996 |
| `metal/f32/layer_norm/strided/inner_offset_r16_c129` | normalization | 16 × 129 | offset | correct | [passed](operator_peer/metal__f32__layer_norm__strided__inner_offset_r16_c129.json) | 0.2651 / 0.2166 | [passed](program_peer/metal__f32__layer_norm__strided__inner_offset_r16_c129.json) | 0.3065 / 0.216 |
| `metal/f32/layer_norm/strided/inner_offset_r33_c32` | normalization | 33 × 32 | offset | correct | [passed](operator_peer/metal__f32__layer_norm__strided__inner_offset_r33_c32.json) | 0.3286 / 0.2026 | [passed](program_peer/metal__f32__layer_norm__strided__inner_offset_r33_c32.json) | 0.3302 / 0.2059 |
| `metal/f32/layer_norm/strided/inner_offset_r33_c129` | normalization | 33 × 129 | offset | correct | [passed](operator_peer/metal__f32__layer_norm__strided__inner_offset_r33_c129.json) | 0.342 / 0.2401 | [passed](program_peer/metal__f32__layer_norm__strided__inner_offset_r33_c129.json) | 0.2792 / 0.2257 |
| `metal/f32/layer_norm/strided/inner_offset_r128_c129` | normalization | 128 × 129 | offset | correct | [passed](operator_peer/metal__f32__layer_norm__strided__inner_offset_r128_c129.json) | 0.4354 / 0.4121 | [passed](program_peer/metal__f32__layer_norm__strided__inner_offset_r128_c129.json) | 0.4284 / 0.4256 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
