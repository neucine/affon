# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `04f1bf9e174d26ef056eae0ed7f7a2c8df2436a5cb67dd171696ba824a2afe4a`  
Binary SHA-256: `6d2defde8cf9b10d0782c369e31cd181ec6552402e509df549c93bc6df8402ff`  
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
| `metal/bf16/layer_norm/strided/inner_strided_r3_c5` | normalization | 3 × 5 | strided | correct | [passed](operator_peer/metal__bf16__layer_norm__strided__inner_strided_r3_c5.json) | 0.301 / 0.2013 | [passed](program_peer/metal__bf16__layer_norm__strided__inner_strided_r3_c5.json) | 0.3158 / 0.1969 |
| `metal/bf16/layer_norm/strided/inner_strided_r3_c32` | normalization | 3 × 32 | strided | correct | [passed](operator_peer/metal__bf16__layer_norm__strided__inner_strided_r3_c32.json) | 0.284 / 0.1785 | [passed](program_peer/metal__bf16__layer_norm__strided__inner_strided_r3_c32.json) | 0.2862 / 0.1985 |
| `metal/bf16/layer_norm/strided/inner_strided_r16_c5` | normalization | 16 × 5 | strided | correct | [passed](operator_peer/metal__bf16__layer_norm__strided__inner_strided_r16_c5.json) | 0.317 / 0.1962 | [passed](program_peer/metal__bf16__layer_norm__strided__inner_strided_r16_c5.json) | 0.3338 / 0.1943 |
| `metal/bf16/layer_norm/strided/inner_strided_r16_c32` | normalization | 16 × 32 | strided | correct | [passed](operator_peer/metal__bf16__layer_norm__strided__inner_strided_r16_c32.json) | 0.2904 / 0.1935 | [passed](program_peer/metal__bf16__layer_norm__strided__inner_strided_r16_c32.json) | 0.3786 / 0.2083 |
| `metal/bf16/layer_norm/strided/inner_offset_r3_c5` | normalization | 3 × 5 | offset | correct | [passed](operator_peer/metal__bf16__layer_norm__strided__inner_offset_r3_c5.json) | 0.3072 / 0.1939 | [passed](program_peer/metal__bf16__layer_norm__strided__inner_offset_r3_c5.json) | 0.2532 / 0.1968 |
| `metal/bf16/layer_norm/strided/inner_offset_r3_c32` | normalization | 3 × 32 | offset | correct | [passed](operator_peer/metal__bf16__layer_norm__strided__inner_offset_r3_c32.json) | 0.278 / 0.2001 | [passed](program_peer/metal__bf16__layer_norm__strided__inner_offset_r3_c32.json) | 0.2832 / 0.1907 |
| `metal/bf16/layer_norm/strided/inner_offset_r16_c5` | normalization | 16 × 5 | offset | correct | [passed](operator_peer/metal__bf16__layer_norm__strided__inner_offset_r16_c5.json) | 0.3168 / 0.1983 | [passed](program_peer/metal__bf16__layer_norm__strided__inner_offset_r16_c5.json) | 0.331 / 0.1924 |
| `metal/bf16/layer_norm/strided/inner_offset_r16_c32` | normalization | 16 × 32 | offset | correct | [passed](operator_peer/metal__bf16__layer_norm__strided__inner_offset_r16_c32.json) | 0.366 / 0.1971 | [passed](program_peer/metal__bf16__layer_norm__strided__inner_offset_r16_c32.json) | 0.3054 / 0.1994 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
