# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `04f1bf9e174d26ef056eae0ed7f7a2c8df2436a5cb67dd171696ba824a2afe4a`  
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
| `metal/bf16/layer_norm/strided/inner_strided_r3_c5` | normalization | 3 × 5 | strided | correct | [passed](operator_peer/metal__bf16__layer_norm__strided__inner_strided_r3_c5.json) | 0.279 / 0.2223 | [passed](program_peer/metal__bf16__layer_norm__strided__inner_strided_r3_c5.json) | 0.229 / 0.176 |
| `metal/bf16/layer_norm/strided/inner_strided_r3_c32` | normalization | 3 × 32 | strided | correct | [passed](operator_peer/metal__bf16__layer_norm__strided__inner_strided_r3_c32.json) | 0.2476 / 0.1751 | [passed](program_peer/metal__bf16__layer_norm__strided__inner_strided_r3_c32.json) | 0.2598 / 0.1789 |
| `metal/bf16/layer_norm/strided/inner_strided_r16_c5` | normalization | 16 × 5 | strided | correct | [passed](operator_peer/metal__bf16__layer_norm__strided__inner_strided_r16_c5.json) | 0.2226 / 0.1748 | [passed](program_peer/metal__bf16__layer_norm__strided__inner_strided_r16_c5.json) | 0.2356 / 0.1763 |
| `metal/bf16/layer_norm/strided/inner_strided_r16_c32` | normalization | 16 × 32 | strided | correct | [passed](operator_peer/metal__bf16__layer_norm__strided__inner_strided_r16_c32.json) | 0.256 / 0.1829 | [passed](program_peer/metal__bf16__layer_norm__strided__inner_strided_r16_c32.json) | 0.2974 / 0.1836 |
| `metal/bf16/layer_norm/strided/inner_offset_r3_c5` | normalization | 3 × 5 | offset | correct | [passed](operator_peer/metal__bf16__layer_norm__strided__inner_offset_r3_c5.json) | 0.2186 / 0.1631 | [passed](program_peer/metal__bf16__layer_norm__strided__inner_offset_r3_c5.json) | 0.2306 / 0.1642 |
| `metal/bf16/layer_norm/strided/inner_offset_r3_c32` | normalization | 3 × 32 | offset | correct | [passed](operator_peer/metal__bf16__layer_norm__strided__inner_offset_r3_c32.json) | 0.2502 / 0.1643 | [passed](program_peer/metal__bf16__layer_norm__strided__inner_offset_r3_c32.json) | 0.2898 / 0.2156 |
| `metal/bf16/layer_norm/strided/inner_offset_r16_c5` | normalization | 16 × 5 | offset | correct | [passed](operator_peer/metal__bf16__layer_norm__strided__inner_offset_r16_c5.json) | 0.2226 / 0.2231 | [passed](program_peer/metal__bf16__layer_norm__strided__inner_offset_r16_c5.json) | 0.2356 / 0.1749 |
| `metal/bf16/layer_norm/strided/inner_offset_r16_c32` | normalization | 16 × 32 | offset | correct | [passed](operator_peer/metal__bf16__layer_norm__strided__inner_offset_r16_c32.json) | 0.2532 / 0.1789 | [passed](program_peer/metal__bf16__layer_norm__strided__inner_offset_r16_c32.json) | 0.2952 / 0.1887 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
