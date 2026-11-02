# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `8ec03dd1b33e3c093dbc1b2425115e85c803b8ae08f4b2d46ccc887af99b6bb5`  
Binary SHA-256: `1263f72409ea8b8e7c7c5ea44ec4bc78639d602290b1885391b28c52332c724e`  
Timing: 10 iterations × 5 samples × 3 rounds

## Checks

- Cases run: 1
- Numerical check passed: 1
- Operator check and timing passed: 1
- Program check and timing passed: 1
- Numerical failures: 0
- Operator failures: 0
- Program failures: 0

## Cases

| Case | Family | Shape | Layout | Numerical | Operator | Affon / PyTorch ms | Program | Affon / PyTorch ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `metal/f32/layer_norm/long_row/r33_c255` | normalization | 33 × 255 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__long_row__r33_c255.json) | 0.3688 / 0.268 | [passed](program_peer/metal__f32__layer_norm__long_row__r33_c255.json) | 0.3694 / 0.2764 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
