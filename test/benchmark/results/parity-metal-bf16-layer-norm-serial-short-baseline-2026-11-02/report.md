# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `045b41b28bc1b8aa8f7aad018a30ec3582fa1a0cf3e458142d56fb97a8309ca0`  
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
| `metal/bf16/layer_norm/short_row/r32_c31` | normalization | 32 × 31 | dense | correct | [passed](operator_peer/metal__bf16__layer_norm__short_row__r32_c31.json) | 0.3237 / 0.1904 | [passed](program_peer/metal__bf16__layer_norm__short_row__r32_c31.json) | 0.3244 / 0.2069 |
| `metal/bf16/layer_norm/short_row/r32_c32` | normalization | 32 × 32 | dense | correct | [passed](operator_peer/metal__bf16__layer_norm__short_row__r32_c32.json) | 0.3078 / 0.1883 | [passed](program_peer/metal__bf16__layer_norm__short_row__r32_c32.json) | 0.3293 / 0.2034 |
| `metal/bf16/layer_norm/short_row/r32_c33` | normalization | 32 × 33 | dense | correct | [passed](operator_peer/metal__bf16__layer_norm__short_row__r32_c33.json) | 0.3287 / 0.1999 | [passed](program_peer/metal__bf16__layer_norm__short_row__r32_c33.json) | 0.3399 / 0.2041 |
| `metal/bf16/layer_norm/short_row/r8_c7` | normalization | 8 × 7 | dense | correct | [passed](operator_peer/metal__bf16__layer_norm__short_row__r8_c7.json) | 0.2416 / 0.1871 | [passed](program_peer/metal__bf16__layer_norm__short_row__r8_c7.json) | 0.2587 / 0.1932 |
| `metal/bf16/layer_norm/short_row/r8_c15` | normalization | 8 × 15 | dense | correct | [passed](operator_peer/metal__bf16__layer_norm__short_row__r8_c15.json) | 0.2538 / 0.1943 | [passed](program_peer/metal__bf16__layer_norm__short_row__r8_c15.json) | 0.3002 / 0.1898 |
| `metal/bf16/layer_norm/short_row/r8_c17` | normalization | 8 × 17 | dense | correct | [passed](operator_peer/metal__bf16__layer_norm__short_row__r8_c17.json) | 0.2585 / 0.1911 | [passed](program_peer/metal__bf16__layer_norm__short_row__r8_c17.json) | 0.2818 / 0.1856 |
| `metal/bf16/layer_norm/short_row/r16_c32` | normalization | 16 × 32 | dense | correct | [passed](operator_peer/metal__bf16__layer_norm__short_row__r16_c32.json) | 0.3042 / 0.1916 | [passed](program_peer/metal__bf16__layer_norm__short_row__r16_c32.json) | 0.337 / 0.1959 |
| `metal/bf16/layer_norm/short_row/r16_c33` | normalization | 16 × 33 | dense | correct | [passed](operator_peer/metal__bf16__layer_norm__short_row__r16_c33.json) | 0.3687 / 0.1913 | [passed](program_peer/metal__bf16__layer_norm__short_row__r16_c33.json) | 0.314 / 0.2001 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
