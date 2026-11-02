# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `1966d95190fca67d060651fd6243bc6381207db5115a60b9235306ecd890ef4c`  
Binary SHA-256: `07d2686a6fde2c671da3d04668ac145d8ca91e78bd6eb55675701480f9735945`  
Timing: 5 iterations × 3 samples × 2 rounds

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
| `metal/bf16/layer_norm/outer_axis/outer_offset_r2_c17` | normalization | 2 × 17 | offset | correct | [passed](operator_peer/metal__bf16__layer_norm__outer_axis__outer_offset_r2_c17.json) | 0.4621 / 0.2502 | [passed](program_peer/metal__bf16__layer_norm__outer_axis__outer_offset_r2_c17.json) | 0.4801 / 0.201 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
