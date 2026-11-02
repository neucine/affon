# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `ecafc8daf6cf3d7d28f8cc7d658b6709825a1a5759321dc07f3ebf9c7abf7ba1`  
Binary SHA-256: `921ae28ce07860990ec9e503afd1d4472af2c309c647eb11d084cd7d329cdb62`  
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
| `metal/f32/layer_norm/strided/inner_strided_r3_c129` | normalization | 3 × 129 | strided | correct | [passed](operator_peer/metal__f32__layer_norm__strided__inner_strided_r3_c129.json) | 0.2618 / 0.2392 | [passed](program_peer/metal__f32__layer_norm__strided__inner_strided_r3_c129.json) | 0.2704 / 0.1974 |
| `metal/f32/layer_norm/strided/inner_strided_r16_c129` | normalization | 16 × 129 | strided | correct | [passed](operator_peer/metal__f32__layer_norm__strided__inner_strided_r16_c129.json) | 0.2858 / 0.2219 | [passed](program_peer/metal__f32__layer_norm__strided__inner_strided_r16_c129.json) | 0.2918 / 0.2228 |
| `metal/f32/layer_norm/strided/inner_strided_r33_c129` | normalization | 33 × 129 | strided | correct | [passed](operator_peer/metal__f32__layer_norm__strided__inner_strided_r33_c129.json) | 0.2736 / 0.2435 | [passed](program_peer/metal__f32__layer_norm__strided__inner_strided_r33_c129.json) | 0.314 / 0.2412 |
| `metal/f32/layer_norm/strided/inner_strided_r128_c129` | normalization | 128 × 129 | strided | correct | [passed](operator_peer/metal__f32__layer_norm__strided__inner_strided_r128_c129.json) | 0.428 / 0.4088 | [passed](program_peer/metal__f32__layer_norm__strided__inner_strided_r128_c129.json) | 0.453 / 0.4295 |
| `metal/f32/layer_norm/strided/inner_offset_r3_c129` | normalization | 3 × 129 | offset | correct | [passed](operator_peer/metal__f32__layer_norm__strided__inner_offset_r3_c129.json) | 0.2638 / 0.1923 | [passed](program_peer/metal__f32__layer_norm__strided__inner_offset_r3_c129.json) | 0.2622 / 0.211 |
| `metal/f32/layer_norm/strided/inner_offset_r16_c129` | normalization | 16 × 129 | offset | correct | [passed](operator_peer/metal__f32__layer_norm__strided__inner_offset_r16_c129.json) | 0.2856 / 0.2228 | [passed](program_peer/metal__f32__layer_norm__strided__inner_offset_r16_c129.json) | 0.2904 / 0.2244 |
| `metal/f32/layer_norm/strided/inner_offset_r33_c129` | normalization | 33 × 129 | offset | correct | [passed](operator_peer/metal__f32__layer_norm__strided__inner_offset_r33_c129.json) | 0.3078 / 0.2398 | [passed](program_peer/metal__f32__layer_norm__strided__inner_offset_r33_c129.json) | 0.3126 / 0.2316 |
| `metal/f32/layer_norm/strided/inner_offset_r128_c129` | normalization | 128 × 129 | offset | correct | [passed](operator_peer/metal__f32__layer_norm__strided__inner_offset_r128_c129.json) | 0.4434 / 0.4185 | [passed](program_peer/metal__f32__layer_norm__strided__inner_offset_r128_c129.json) | 0.45 / 0.4329 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
