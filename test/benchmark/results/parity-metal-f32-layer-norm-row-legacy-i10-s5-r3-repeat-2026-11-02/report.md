# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `8d3f4800f36c2de996a6bf6fdb312f746256cf0af8a7b77f2b280f423f2aa9ae`  
Binary SHA-256: `1263f72409ea8b8e7c7c5ea44ec4bc78639d602290b1885391b28c52332c724e`  
Timing: 10 iterations × 5 samples × 3 rounds

## Checks

- Cases run: 3
- Numerical check passed: 3
- Operator check and timing passed: 3
- Program check and timing passed: 3
- Numerical failures: 0
- Operator failures: 0
- Program failures: 0

## Cases

| Case | Family | Shape | Layout | Numerical | Operator | Affon / PyTorch ms | Program | Affon / PyTorch ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `metal/f32/layer_norm/long_row/r32_c127` | normalization | 32 × 127 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__long_row__r32_c127.json) | 0.3304 / 0.221 | [passed](program_peer/metal__f32__layer_norm__long_row__r32_c127.json) | 0.3361 / 0.2366 |
| `metal/f32/layer_norm/long_row/r33_c255` | normalization | 33 × 255 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__long_row__r33_c255.json) | 0.3606 / 0.2715 | [passed](program_peer/metal__f32__layer_norm__long_row__r33_c255.json) | 0.3622 / 0.2725 |
| `metal/f32/layer_norm/long_row/r64_c255` | normalization | 64 × 255 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__long_row__r64_c255.json) | 0.4351 / 0.3951 | [passed](program_peer/metal__f32__layer_norm__long_row__r64_c255.json) | 0.3853 / 0.4066 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
