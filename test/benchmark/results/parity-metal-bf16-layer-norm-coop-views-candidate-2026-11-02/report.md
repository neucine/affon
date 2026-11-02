# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `7be5828d2bb090de9e0b41ec5cb36870d9b1baec907338103819930b0a9ae7c1`  
Binary SHA-256: `07d2686a6fde2c671da3d04668ac145d8ca91e78bd6eb55675701480f9735945`  
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
| `metal/bf16/layer_norm/strided/inner_strided_r3_c65` | normalization | 3 × 65 | strided | correct | [passed](operator_peer/metal__bf16__layer_norm__strided__inner_strided_r3_c65.json) | 0.2756 / 0.243 | [passed](program_peer/metal__bf16__layer_norm__strided__inner_strided_r3_c65.json) | 0.2602 / 0.1984 |
| `metal/bf16/layer_norm/strided/inner_strided_r3_c129` | normalization | 3 × 129 | strided | correct | [passed](operator_peer/metal__bf16__layer_norm__strided__inner_strided_r3_c129.json) | 0.25 / 0.1906 | [passed](program_peer/metal__bf16__layer_norm__strided__inner_strided_r3_c129.json) | 0.234 / 0.2054 |
| `metal/bf16/layer_norm/strided/inner_strided_r16_c65` | normalization | 16 × 65 | strided | correct | [passed](operator_peer/metal__bf16__layer_norm__strided__inner_strided_r16_c65.json) | 0.2678 / 0.1904 | [passed](program_peer/metal__bf16__layer_norm__strided__inner_strided_r16_c65.json) | 0.2854 / 0.2128 |
| `metal/bf16/layer_norm/strided/inner_strided_r16_c129` | normalization | 16 × 129 | strided | correct | [passed](operator_peer/metal__bf16__layer_norm__strided__inner_strided_r16_c129.json) | 0.2752 / 0.2204 | [passed](program_peer/metal__bf16__layer_norm__strided__inner_strided_r16_c129.json) | 0.3012 / 0.2246 |
| `metal/bf16/layer_norm/strided/inner_offset_r3_c65` | normalization | 3 × 65 | offset | correct | [passed](operator_peer/metal__bf16__layer_norm__strided__inner_offset_r3_c65.json) | 0.249 / 0.1951 | [passed](program_peer/metal__bf16__layer_norm__strided__inner_offset_r3_c65.json) | 0.2614 / 0.1863 |
| `metal/bf16/layer_norm/strided/inner_offset_r3_c129` | normalization | 3 × 129 | offset | correct | [passed](operator_peer/metal__bf16__layer_norm__strided__inner_offset_r3_c129.json) | 0.2518 / 0.1961 | [passed](program_peer/metal__bf16__layer_norm__strided__inner_offset_r3_c129.json) | 0.2834 / 0.1951 |
| `metal/bf16/layer_norm/strided/inner_offset_r16_c65` | normalization | 16 × 65 | offset | correct | [passed](operator_peer/metal__bf16__layer_norm__strided__inner_offset_r16_c65.json) | 0.2688 / 0.2148 | [passed](program_peer/metal__bf16__layer_norm__strided__inner_offset_r16_c65.json) | 0.2674 / 0.2034 |
| `metal/bf16/layer_norm/strided/inner_offset_r16_c129` | normalization | 16 × 129 | offset | correct | [passed](operator_peer/metal__bf16__layer_norm__strided__inner_offset_r16_c129.json) | 0.2728 / 0.2205 | [passed](program_peer/metal__bf16__layer_norm__strided__inner_offset_r16_c129.json) | 0.3 / 0.2096 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
