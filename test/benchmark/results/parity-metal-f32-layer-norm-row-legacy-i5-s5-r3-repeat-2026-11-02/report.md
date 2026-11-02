# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `e0cb8897782f8e3850357c36fea09d14081a00b62dec05eadfa8be478339a801`  
Binary SHA-256: `1263f72409ea8b8e7c7c5ea44ec4bc78639d602290b1885391b28c52332c724e`  
Timing: 5 iterations × 5 samples × 3 rounds

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
| `metal/f32/layer_norm/outer_axis/outer_strided_r129_c17` | normalization | 129 × 17 | strided | correct | [passed](operator_peer/metal__f32__layer_norm__outer_axis__outer_strided_r129_c17.json) | 0.5326 / 0.2243 | [passed](program_peer/metal__f32__layer_norm__outer_axis__outer_strided_r129_c17.json) | 0.5218 / 0.2162 |
| `metal/f32/layer_norm/outer_axis/outer_offset_r64_c63` | normalization | 64 × 63 | offset | correct | [passed](operator_peer/metal__f32__layer_norm__outer_axis__outer_offset_r64_c63.json) | 0.339 / 0.2285 | [passed](program_peer/metal__f32__layer_norm__outer_axis__outer_offset_r64_c63.json) | 0.5014 / 0.2443 |
| `metal/f32/layer_norm/outer_axis/outer_offset_r64_c127` | normalization | 64 × 127 | offset | correct | [passed](operator_peer/metal__f32__layer_norm__outer_axis__outer_offset_r64_c127.json) | 0.54 / 0.2786 | [passed](program_peer/metal__f32__layer_norm__outer_axis__outer_offset_r64_c127.json) | 0.428 / 0.2797 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
