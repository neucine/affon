# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `d6a292006bd8a18b7a71047c02f44aea47ac92eb6c087035b9e84772077c9c8a`  
Binary SHA-256: `07d2686a6fde2c671da3d04668ac145d8ca91e78bd6eb55675701480f9735945`  
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
| `metal/bf16/layer_norm/long_row/r32_c64` | normalization | 32 × 64 | dense | correct | [passed](operator_peer/metal__bf16__layer_norm__long_row__r32_c64.json) | 0.2557 / 0.2175 | [passed](program_peer/metal__bf16__layer_norm__long_row__r32_c64.json) | 0.3122 / 0.2171 |
| `metal/bf16/layer_norm/long_row/r32_c65` | normalization | 32 × 65 | dense | correct | [passed](operator_peer/metal__bf16__layer_norm__long_row__r32_c65.json) | 0.2794 / 0.2125 | [passed](program_peer/metal__bf16__layer_norm__long_row__r32_c65.json) | 0.305 / 0.22 |
| `metal/bf16/layer_norm/long_row/r32_c127` | normalization | 32 × 127 | dense | correct | [passed](operator_peer/metal__bf16__layer_norm__long_row__r32_c127.json) | 0.2941 / 0.2353 | [passed](program_peer/metal__bf16__layer_norm__long_row__r32_c127.json) | 0.3192 / 0.237 |
| `metal/bf16/layer_norm/long_row/r32_c128` | normalization | 32 × 128 | dense | correct | [passed](operator_peer/metal__bf16__layer_norm__long_row__r32_c128.json) | 0.2958 / 0.2334 | [passed](program_peer/metal__bf16__layer_norm__long_row__r32_c128.json) | 0.2821 / 0.2326 |
| `metal/bf16/layer_norm/long_row/r32_c129` | normalization | 32 × 129 | dense | correct | [passed](operator_peer/metal__bf16__layer_norm__long_row__r32_c129.json) | 0.2926 / 0.2359 | [passed](program_peer/metal__bf16__layer_norm__long_row__r32_c129.json) | 0.317 / 0.2382 |
| `metal/bf16/layer_norm/long_row/r128_c129` | normalization | 128 × 129 | dense | correct | [passed](operator_peer/metal__bf16__layer_norm__long_row__r128_c129.json) | 0.3927 / 0.3859 | [passed](program_peer/metal__bf16__layer_norm__long_row__r128_c129.json) | 0.4196 / 0.4052 |
| `metal/bf16/layer_norm/long_row/r33_c255` | normalization | 33 × 255 | dense | correct | [passed](operator_peer/metal__bf16__layer_norm__long_row__r33_c255.json) | 0.2852 / 0.2736 | [passed](program_peer/metal__bf16__layer_norm__long_row__r33_c255.json) | 0.349 / 0.2789 |
| `metal/bf16/layer_norm/long_row/r32_c257` | normalization | 32 × 257 | dense | correct | [passed](operator_peer/metal__bf16__layer_norm__long_row__r32_c257.json) | 0.3241 / 0.2662 | [passed](program_peer/metal__bf16__layer_norm__long_row__r32_c257.json) | 0.3481 / 0.2745 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
