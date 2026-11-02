# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `b7861a42636ee7504424ce5d74b7b9e6faefe62e9e911df3e9f8cd7d2056004b`  
Binary SHA-256: `1263f72409ea8b8e7c7c5ea44ec4bc78639d602290b1885391b28c52332c724e`  
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
| `metal/f32/layer_norm/short_row/r32_c31` | normalization | 32 × 31 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__short_row__r32_c31.json) | 0.2969 / 0.2094 | [passed](program_peer/metal__f32__layer_norm__short_row__r32_c31.json) | 0.2619 / 0.1997 |
| `metal/f32/layer_norm/short_row/r32_c32` | normalization | 32 × 32 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__short_row__r32_c32.json) | 0.2342 / 0.2034 | [passed](program_peer/metal__f32__layer_norm__short_row__r32_c32.json) | 0.308 / 0.2022 |
| `metal/f32/layer_norm/short_row/r8_c7` | normalization | 8 × 7 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__short_row__r8_c7.json) | 0.2499 / 0.192 | [passed](program_peer/metal__f32__layer_norm__short_row__r8_c7.json) | 0.2536 / 0.1829 |
| `metal/f32/layer_norm/short_row/r16_c33` | normalization | 16 × 33 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__short_row__r16_c33.json) | 0.3144 / 0.2004 | [passed](program_peer/metal__f32__layer_norm__short_row__r16_c33.json) | 0.2534 / 0.1957 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
