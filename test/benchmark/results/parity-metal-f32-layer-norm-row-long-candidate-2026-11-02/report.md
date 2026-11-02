# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `46d78b41f21937fc9dd12c3e1d9d8dd8b1854a5d85c733328844363e3ac8ab06`  
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
| `metal/f32/layer_norm/long_row/r32_c63` | normalization | 32 × 63 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__long_row__r32_c63.json) | 0.2805 / 0.199 | [passed](program_peer/metal__f32__layer_norm__long_row__r32_c63.json) | 0.2864 / 0.2121 |
| `metal/f32/layer_norm/long_row/r32_c65` | normalization | 32 × 65 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__long_row__r32_c65.json) | 0.4298 / 0.2075 | [passed](program_peer/metal__f32__layer_norm__long_row__r32_c65.json) | 0.4327 / 0.2175 |
| `metal/f32/layer_norm/long_row/r32_c128` | normalization | 32 × 128 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__long_row__r32_c128.json) | 0.2774 / 0.2347 | [passed](program_peer/metal__f32__layer_norm__long_row__r32_c128.json) | 0.3041 / 0.2373 |
| `metal/f32/layer_norm/long_row/r32_c129` | normalization | 32 × 129 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__long_row__r32_c129.json) | 0.2995 / 0.2364 | [passed](program_peer/metal__f32__layer_norm__long_row__r32_c129.json) | 0.2987 / 0.2367 |
| `metal/f32/layer_norm/long_row/r256_c31` | normalization | 256 × 31 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__long_row__r256_c31.json) | 0.4623 / 0.2718 | [passed](program_peer/metal__f32__layer_norm__long_row__r256_c31.json) | 0.4703 / 0.2736 |
| `metal/f32/layer_norm/long_row/r128_c129` | normalization | 128 × 129 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__long_row__r128_c129.json) | 0.4393 / 0.3815 | [passed](program_peer/metal__f32__layer_norm__long_row__r128_c129.json) | 0.4428 / 0.4166 |
| `metal/f32/layer_norm/long_row/r33_c255` | normalization | 33 × 255 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__long_row__r33_c255.json) | 0.3302 / 0.2739 | [passed](program_peer/metal__f32__layer_norm__long_row__r33_c255.json) | 0.3351 / 0.2769 |
| `metal/f32/layer_norm/long_row/r32_c257` | normalization | 32 × 257 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__long_row__r32_c257.json) | 0.3288 / 0.2739 | [passed](program_peer/metal__f32__layer_norm__long_row__r32_c257.json) | 0.3343 / 0.2757 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
