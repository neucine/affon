# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `ecec24c3cee55ecea6a70ab48b1a75864e532bd50acb28f29a2897dffa2d930e`  
Binary SHA-256: `1263f72409ea8b8e7c7c5ea44ec4bc78639d602290b1885391b28c52332c724e`  
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
| `metal/f32/layer_norm/strided/inner_offset_r2_c257` | normalization | 2 × 257 | offset | correct | [passed](operator_peer/metal__f32__layer_norm__strided__inner_offset_r2_c257.json) | 0.2444 / 0.2329 | [passed](program_peer/metal__f32__layer_norm__strided__inner_offset_r2_c257.json) | 0.3801 / 0.1947 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
