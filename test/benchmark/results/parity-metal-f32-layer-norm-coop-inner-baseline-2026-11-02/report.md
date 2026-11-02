# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `ecafc8daf6cf3d7d28f8cc7d658b6709825a1a5759321dc07f3ebf9c7abf7ba1`  
Binary SHA-256: `3ed8d74ef1f81a050918c3e3ff64fe120738f504c5426976e5a1ab291f3a390b`  
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
| `metal/f32/layer_norm/strided/inner_strided_r3_c129` | normalization | 3 × 129 | strided | correct | [passed](operator_peer/metal__f32__layer_norm__strided__inner_strided_r3_c129.json) | 0.2956 / 0.2402 | [passed](program_peer/metal__f32__layer_norm__strided__inner_strided_r3_c129.json) | 0.3012 / 0.1982 |
| `metal/f32/layer_norm/strided/inner_strided_r16_c129` | normalization | 16 × 129 | strided | correct | [passed](operator_peer/metal__f32__layer_norm__strided__inner_strided_r16_c129.json) | 0.3072 / 0.2219 | [passed](program_peer/metal__f32__layer_norm__strided__inner_strided_r16_c129.json) | 0.2652 / 0.2223 |
| `metal/f32/layer_norm/strided/inner_strided_r33_c129` | normalization | 33 × 129 | strided | correct | [passed](operator_peer/metal__f32__layer_norm__strided__inner_strided_r33_c129.json) | 0.3372 / 0.2405 | [passed](program_peer/metal__f32__layer_norm__strided__inner_strided_r33_c129.json) | 0.351 / 0.2423 |
| `metal/f32/layer_norm/strided/inner_strided_r128_c129` | normalization | 128 × 129 | strided | correct | [passed](operator_peer/metal__f32__layer_norm__strided__inner_strided_r128_c129.json) | 0.4744 / 0.4209 | [passed](program_peer/metal__f32__layer_norm__strided__inner_strided_r128_c129.json) | 0.551 / 0.414 |
| `metal/f32/layer_norm/strided/inner_offset_r3_c129` | normalization | 3 × 129 | offset | correct | [passed](operator_peer/metal__f32__layer_norm__strided__inner_offset_r3_c129.json) | 0.2958 / 0.1986 | [passed](program_peer/metal__f32__layer_norm__strided__inner_offset_r3_c129.json) | 0.2992 / 0.2016 |
| `metal/f32/layer_norm/strided/inner_offset_r16_c129` | normalization | 16 × 129 | offset | correct | [passed](operator_peer/metal__f32__layer_norm__strided__inner_offset_r16_c129.json) | 0.2672 / 0.2065 | [passed](program_peer/metal__f32__layer_norm__strided__inner_offset_r16_c129.json) | 0.2674 / 0.224 |
| `metal/f32/layer_norm/strided/inner_offset_r33_c129` | normalization | 33 × 129 | offset | correct | [passed](operator_peer/metal__f32__layer_norm__strided__inner_offset_r33_c129.json) | 0.328 / 0.239 | [passed](program_peer/metal__f32__layer_norm__strided__inner_offset_r33_c129.json) | 0.3268 / 0.2417 |
| `metal/f32/layer_norm/strided/inner_offset_r128_c129` | normalization | 128 × 129 | offset | correct | [passed](operator_peer/metal__f32__layer_norm__strided__inner_offset_r128_c129.json) | 0.5184 / 0.4179 | [passed](program_peer/metal__f32__layer_norm__strided__inner_offset_r128_c129.json) | 0.5346 / 0.4275 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
