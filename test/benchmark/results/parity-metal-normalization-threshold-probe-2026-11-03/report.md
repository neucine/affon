# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `b981b1201d98dba46157d367018e2c9b27746ad03f4c91096182024038481784`  
Binary SHA-256: `3e84e3a9cbcf982c6469bc4edfd12c6a72d8a729be77498988b7e5503fc3efec`  
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
| `metal/f32/layer_norm/outer_axis/outer_offset_r33_c65` | normalization | 33 × 65 | offset | correct | [passed](operator_peer/metal__f32__layer_norm__outer_axis__outer_offset_r33_c65.json) | 0.3486 / 0.2383 | [passed](program_peer/metal__f32__layer_norm__outer_axis__outer_offset_r33_c65.json) | 0.3359 / 0.2248 |
| `metal/f32/layer_norm/outer_axis/outer_strided_r129_c17` | normalization | 129 × 17 | strided | correct | [passed](operator_peer/metal__f32__layer_norm__outer_axis__outer_strided_r129_c17.json) | 0.3643 / 0.2122 | [passed](program_peer/metal__f32__layer_norm__outer_axis__outer_strided_r129_c17.json) | 0.4021 / 0.229 |
| `metal/f32/layer_norm/outer_axis/outer_offset_r64_c63` | normalization | 64 × 63 | offset | correct | [passed](operator_peer/metal__f32__layer_norm__outer_axis__outer_offset_r64_c63.json) | 0.368 / 0.2291 | [passed](program_peer/metal__f32__layer_norm__outer_axis__outer_offset_r64_c63.json) | 0.4036 / 0.2265 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
