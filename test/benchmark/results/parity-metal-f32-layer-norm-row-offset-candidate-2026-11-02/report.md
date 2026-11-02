# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `906bb65250699f4fa2ec296b809f441413d89319398a398b1ade2959322c4cae`  
Binary SHA-256: `6d2defde8cf9b10d0782c369e31cd181ec6552402e509df549c93bc6df8402ff`  
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
| `metal/f32/layer_norm/strided/inner_offset_r3_c5` | normalization | 3 × 5 | offset | correct | [passed](operator_peer/metal__f32__layer_norm__strided__inner_offset_r3_c5.json) | 0.3035 / 0.1877 | [passed](program_peer/metal__f32__layer_norm__strided__inner_offset_r3_c5.json) | 0.3119 / 0.1929 |
| `metal/f32/layer_norm/strided/inner_offset_r3_c65` | normalization | 3 × 65 | offset | correct | [passed](operator_peer/metal__f32__layer_norm__strided__inner_offset_r3_c65.json) | 0.3468 / 0.1994 | [passed](program_peer/metal__f32__layer_norm__strided__inner_offset_r3_c65.json) | 0.3471 / 0.1959 |
| `metal/f32/layer_norm/strided/inner_offset_r16_c5` | normalization | 16 × 5 | offset | correct | [passed](operator_peer/metal__f32__layer_norm__strided__inner_offset_r16_c5.json) | 0.3205 / 0.1949 | [passed](program_peer/metal__f32__layer_norm__strided__inner_offset_r16_c5.json) | 0.3307 / 0.1929 |
| `metal/f32/layer_norm/strided/inner_offset_r16_c32` | normalization | 16 × 32 | offset | correct | [passed](operator_peer/metal__f32__layer_norm__strided__inner_offset_r16_c32.json) | 0.3085 / 0.1952 | [passed](program_peer/metal__f32__layer_norm__strided__inner_offset_r16_c32.json) | 0.3085 / 0.2008 |
| `metal/f32/layer_norm/strided/inner_offset_r16_c129` | normalization | 16 × 129 | offset | correct | [passed](operator_peer/metal__f32__layer_norm__strided__inner_offset_r16_c129.json) | 0.2782 / 0.2208 | [passed](program_peer/metal__f32__layer_norm__strided__inner_offset_r16_c129.json) | 0.28 / 0.2242 |
| `metal/f32/layer_norm/strided/inner_offset_r33_c32` | normalization | 33 × 32 | offset | correct | [passed](operator_peer/metal__f32__layer_norm__strided__inner_offset_r33_c32.json) | 0.3287 / 0.2036 | [passed](program_peer/metal__f32__layer_norm__strided__inner_offset_r33_c32.json) | 0.3323 / 0.2091 |
| `metal/f32/layer_norm/strided/inner_offset_r33_c129` | normalization | 33 × 129 | offset | correct | [passed](operator_peer/metal__f32__layer_norm__strided__inner_offset_r33_c129.json) | 0.3054 / 0.24 | [passed](program_peer/metal__f32__layer_norm__strided__inner_offset_r33_c129.json) | 0.2964 / 0.2434 |
| `metal/f32/layer_norm/strided/inner_offset_r128_c129` | normalization | 128 × 129 | offset | correct | [passed](operator_peer/metal__f32__layer_norm__strided__inner_offset_r128_c129.json) | 0.3862 / 0.3959 | [passed](program_peer/metal__f32__layer_norm__strided__inner_offset_r128_c129.json) | 0.4468 / 0.4281 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
