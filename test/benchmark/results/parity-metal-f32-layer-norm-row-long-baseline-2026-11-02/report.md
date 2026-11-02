# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `46d78b41f21937fc9dd12c3e1d9d8dd8b1854a5d85c733328844363e3ac8ab06`  
Binary SHA-256: `a9a10dd4a4dd6ab577715a114b1261d475aeddc676e7750489ffff1b83de8cef`  
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
| `metal/f32/layer_norm/long_row/r32_c63` | normalization | 32 × 63 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__long_row__r32_c63.json) | 0.2755 / 0.2238 | [passed](program_peer/metal__f32__layer_norm__long_row__r32_c63.json) | 0.273 / 0.2117 |
| `metal/f32/layer_norm/long_row/r32_c65` | normalization | 32 × 65 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__long_row__r32_c65.json) | 0.437 / 0.2125 | [passed](program_peer/metal__f32__layer_norm__long_row__r32_c65.json) | 0.4402 / 0.2187 |
| `metal/f32/layer_norm/long_row/r32_c128` | normalization | 32 × 128 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__long_row__r32_c128.json) | 0.3197 / 0.2363 | [passed](program_peer/metal__f32__layer_norm__long_row__r32_c128.json) | 0.3254 / 0.2371 |
| `metal/f32/layer_norm/long_row/r32_c129` | normalization | 32 × 129 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__long_row__r32_c129.json) | 0.3051 / 0.2254 | [passed](program_peer/metal__f32__layer_norm__long_row__r32_c129.json) | 0.3317 / 0.2342 |
| `metal/f32/layer_norm/long_row/r256_c31` | normalization | 256 × 31 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__long_row__r256_c31.json) | 0.463 / 0.2717 | [passed](program_peer/metal__f32__layer_norm__long_row__r256_c31.json) | 0.4691 / 0.2757 |
| `metal/f32/layer_norm/long_row/r128_c129` | normalization | 128 × 129 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__long_row__r128_c129.json) | 0.4195 / 0.3829 | [passed](program_peer/metal__f32__layer_norm__long_row__r128_c129.json) | 0.422 / 0.4087 |
| `metal/f32/layer_norm/long_row/r33_c255` | normalization | 33 × 255 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__long_row__r33_c255.json) | 0.3151 / 0.2754 | [passed](program_peer/metal__f32__layer_norm__long_row__r33_c255.json) | 0.3454 / 0.2798 |
| `metal/f32/layer_norm/long_row/r32_c257` | normalization | 32 × 257 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__long_row__r32_c257.json) | 0.3392 / 0.2713 | [passed](program_peer/metal__f32__layer_norm__long_row__r32_c257.json) | 0.3064 / 0.2751 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
