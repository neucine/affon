# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `b7861a42636ee7504424ce5d74b7b9e6faefe62e9e911df3e9f8cd7d2056004b`  
Binary SHA-256: `07d2686a6fde2c671da3d04668ac145d8ca91e78bd6eb55675701480f9735945`  
Timing: 10 iterations × 5 samples × 3 rounds

## Checks

- Cases run: 4
- Numerical check passed: 4
- Operator check and timing passed: 4
- Program check and timing passed: 4
- Numerical failures: 0
- Operator failures: 0
- Program failures: 0

## Cases

| Case | Family | Shape | Layout | Numerical | Operator | Affon / PyTorch ms | Program | Affon / PyTorch ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `metal/f32/layer_norm/short_row/r32_c31` | normalization | 32 × 31 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__short_row__r32_c31.json) | 0.3334 / 0.2093 | [passed](program_peer/metal__f32__layer_norm__short_row__r32_c31.json) | 0.3329 / 0.2044 |
| `metal/f32/layer_norm/short_row/r32_c32` | normalization | 32 × 32 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__short_row__r32_c32.json) | 0.3261 / 0.2051 | [passed](program_peer/metal__f32__layer_norm__short_row__r32_c32.json) | 0.3221 / 0.202 |
| `metal/f32/layer_norm/short_row/r8_c7` | normalization | 8 × 7 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__short_row__r8_c7.json) | 0.3283 / 0.1897 | [passed](program_peer/metal__f32__layer_norm__short_row__r8_c7.json) | 0.3292 / 0.1925 |
| `metal/f32/layer_norm/short_row/r16_c33` | normalization | 16 × 33 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__short_row__r16_c33.json) | 0.3114 / 0.1969 | [passed](program_peer/metal__f32__layer_norm__short_row__r16_c33.json) | 0.3145 / 0.1987 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
