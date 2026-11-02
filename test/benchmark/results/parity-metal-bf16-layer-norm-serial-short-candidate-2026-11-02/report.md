# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `045b41b28bc1b8aa8f7aad018a30ec3582fa1a0cf3e458142d56fb97a8309ca0`  
Binary SHA-256: `f91be9ab6798b9792f1863db3f9dd08ee1085303718a253112a1e0f54f7d36b2`  
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
| `metal/bf16/layer_norm/short_row/r32_c31` | normalization | 32 × 31 | dense | correct | [passed](operator_peer/metal__bf16__layer_norm__short_row__r32_c31.json) | 0.2729 / 0.2021 | [passed](program_peer/metal__bf16__layer_norm__short_row__r32_c31.json) | 0.3136 / 0.1848 |
| `metal/bf16/layer_norm/short_row/r32_c32` | normalization | 32 × 32 | dense | correct | [passed](operator_peer/metal__bf16__layer_norm__short_row__r32_c32.json) | 0.2678 / 0.1845 | [passed](program_peer/metal__bf16__layer_norm__short_row__r32_c32.json) | 0.2634 / 0.1858 |
| `metal/bf16/layer_norm/short_row/r32_c33` | normalization | 32 × 33 | dense | correct | [passed](operator_peer/metal__bf16__layer_norm__short_row__r32_c33.json) | 0.2393 / 0.185 | [passed](program_peer/metal__bf16__layer_norm__short_row__r32_c33.json) | 0.3137 / 0.1847 |
| `metal/bf16/layer_norm/short_row/r8_c7` | normalization | 8 × 7 | dense | correct | [passed](operator_peer/metal__bf16__layer_norm__short_row__r8_c7.json) | 0.22 / 0.1709 | [passed](program_peer/metal__bf16__layer_norm__short_row__r8_c7.json) | 0.21 / 0.1704 |
| `metal/bf16/layer_norm/short_row/r8_c15` | normalization | 8 × 15 | dense | correct | [passed](operator_peer/metal__bf16__layer_norm__short_row__r8_c15.json) | 0.2267 / 0.1695 | [passed](program_peer/metal__bf16__layer_norm__short_row__r8_c15.json) | 0.2343 / 0.1714 |
| `metal/bf16/layer_norm/short_row/r8_c17` | normalization | 8 × 17 | dense | correct | [passed](operator_peer/metal__bf16__layer_norm__short_row__r8_c17.json) | 0.2303 / 0.1717 | [passed](program_peer/metal__bf16__layer_norm__short_row__r8_c17.json) | 0.2116 / 0.1706 |
| `metal/bf16/layer_norm/short_row/r16_c32` | normalization | 16 × 32 | dense | correct | [passed](operator_peer/metal__bf16__layer_norm__short_row__r16_c32.json) | 0.2592 / 0.1807 | [passed](program_peer/metal__bf16__layer_norm__short_row__r16_c32.json) | 0.2983 / 0.182 |
| `metal/bf16/layer_norm/short_row/r16_c33` | normalization | 16 × 33 | dense | correct | [passed](operator_peer/metal__bf16__layer_norm__short_row__r16_c33.json) | 0.2566 / 0.1784 | [passed](program_peer/metal__bf16__layer_norm__short_row__r16_c33.json) | 0.2644 / 0.1814 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
