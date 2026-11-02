# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `d6a292006bd8a18b7a71047c02f44aea47ac92eb6c087035b9e84772077c9c8a`  
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
| `metal/bf16/layer_norm/long_row/r32_c64` | normalization | 32 × 64 | dense | correct | [passed](operator_peer/metal__bf16__layer_norm__long_row__r32_c64.json) | 0.2669 / 0.2206 | [passed](program_peer/metal__bf16__layer_norm__long_row__r32_c64.json) | 0.3228 / 0.2111 |
| `metal/bf16/layer_norm/long_row/r32_c65` | normalization | 32 × 65 | dense | correct | [passed](operator_peer/metal__bf16__layer_norm__long_row__r32_c65.json) | 0.2821 / 0.204 | [passed](program_peer/metal__bf16__layer_norm__long_row__r32_c65.json) | 0.3253 / 0.2202 |
| `metal/bf16/layer_norm/long_row/r32_c127` | normalization | 32 × 127 | dense | correct | [passed](operator_peer/metal__bf16__layer_norm__long_row__r32_c127.json) | 0.3114 / 0.237 | [passed](program_peer/metal__bf16__layer_norm__long_row__r32_c127.json) | 0.3425 / 0.2385 |
| `metal/bf16/layer_norm/long_row/r32_c128` | normalization | 32 × 128 | dense | correct | [passed](operator_peer/metal__bf16__layer_norm__long_row__r32_c128.json) | 0.3086 / 0.2333 | [passed](program_peer/metal__bf16__layer_norm__long_row__r32_c128.json) | 0.313 / 0.2368 |
| `metal/bf16/layer_norm/long_row/r32_c129` | normalization | 32 × 129 | dense | correct | [passed](operator_peer/metal__bf16__layer_norm__long_row__r32_c129.json) | 0.3111 / 0.2356 | [passed](program_peer/metal__bf16__layer_norm__long_row__r32_c129.json) | 0.2859 / 0.2369 |
| `metal/bf16/layer_norm/long_row/r128_c129` | normalization | 128 × 129 | dense | correct | [passed](operator_peer/metal__bf16__layer_norm__long_row__r128_c129.json) | 0.4152 / 0.3852 | [passed](program_peer/metal__bf16__layer_norm__long_row__r128_c129.json) | 0.4213 / 0.4053 |
| `metal/bf16/layer_norm/long_row/r33_c255` | normalization | 33 × 255 | dense | correct | [passed](operator_peer/metal__bf16__layer_norm__long_row__r33_c255.json) | 0.3405 / 0.2699 | [passed](program_peer/metal__bf16__layer_norm__long_row__r33_c255.json) | 0.3049 / 0.2743 |
| `metal/bf16/layer_norm/long_row/r32_c257` | normalization | 32 × 257 | dense | correct | [passed](operator_peer/metal__bf16__layer_norm__long_row__r32_c257.json) | 0.3543 / 0.2709 | [passed](program_peer/metal__bf16__layer_norm__long_row__r32_c257.json) | 0.3679 / 0.267 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
