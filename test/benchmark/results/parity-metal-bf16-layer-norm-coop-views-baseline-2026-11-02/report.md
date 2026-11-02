# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `7be5828d2bb090de9e0b41ec5cb36870d9b1baec907338103819930b0a9ae7c1`  
Binary SHA-256: `f91be9ab6798b9792f1863db3f9dd08ee1085303718a253112a1e0f54f7d36b2`  
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
| `metal/bf16/layer_norm/strided/inner_strided_r3_c65` | normalization | 3 × 65 | strided | correct | [passed](operator_peer/metal__bf16__layer_norm__strided__inner_strided_r3_c65.json) | 0.2224 / 0.2066 | [passed](program_peer/metal__bf16__layer_norm__strided__inner_strided_r3_c65.json) | 0.257 / 0.2063 |
| `metal/bf16/layer_norm/strided/inner_strided_r3_c129` | normalization | 3 × 129 | strided | correct | [passed](operator_peer/metal__bf16__layer_norm__strided__inner_strided_r3_c129.json) | 0.2474 / 0.2047 | [passed](program_peer/metal__bf16__layer_norm__strided__inner_strided_r3_c129.json) | 0.2888 / 0.2066 |
| `metal/bf16/layer_norm/strided/inner_strided_r16_c65` | normalization | 16 × 65 | strided | correct | [passed](operator_peer/metal__bf16__layer_norm__strided__inner_strided_r16_c65.json) | 0.2768 / 0.2134 | [passed](program_peer/metal__bf16__layer_norm__strided__inner_strided_r16_c65.json) | 0.3098 / 0.2088 |
| `metal/bf16/layer_norm/strided/inner_strided_r16_c129` | normalization | 16 × 129 | strided | correct | [passed](operator_peer/metal__bf16__layer_norm__strided__inner_strided_r16_c129.json) | 0.2936 / 0.2214 | [passed](program_peer/metal__bf16__layer_norm__strided__inner_strided_r16_c129.json) | 0.3146 / 0.2215 |
| `metal/bf16/layer_norm/strided/inner_offset_r3_c65` | normalization | 3 × 65 | offset | correct | [passed](operator_peer/metal__bf16__layer_norm__strided__inner_offset_r3_c65.json) | 0.2594 / 0.1946 | [passed](program_peer/metal__bf16__layer_norm__strided__inner_offset_r3_c65.json) | 0.2862 / 0.2022 |
| `metal/bf16/layer_norm/strided/inner_offset_r3_c129` | normalization | 3 × 129 | offset | correct | [passed](operator_peer/metal__bf16__layer_norm__strided__inner_offset_r3_c129.json) | 0.2616 / 0.204 | [passed](program_peer/metal__bf16__layer_norm__strided__inner_offset_r3_c129.json) | 0.2868 / 0.2039 |
| `metal/bf16/layer_norm/strided/inner_offset_r16_c65` | normalization | 16 × 65 | offset | correct | [passed](operator_peer/metal__bf16__layer_norm__strided__inner_offset_r16_c65.json) | 0.271 / 0.2074 | [passed](program_peer/metal__bf16__layer_norm__strided__inner_offset_r16_c65.json) | 0.262 / 0.2056 |
| `metal/bf16/layer_norm/strided/inner_offset_r16_c129` | normalization | 16 × 129 | offset | correct | [passed](operator_peer/metal__bf16__layer_norm__strided__inner_offset_r16_c129.json) | 0.2846 / 0.2226 | [passed](program_peer/metal__bf16__layer_norm__strided__inner_offset_r16_c129.json) | 0.2658 / 0.2029 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
