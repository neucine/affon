# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `342c51cee84e3df0453cdc0a44448227f7efec103d5d887c9e44f812d0113684`  
Binary SHA-256: `337cb1e10a0cddf87d95edbcb2d3472c55fdd1afe2564afb1770053223905380`  
Timing: 10 iterations × 5 samples × 3 rounds

## Checks

- Cases run: 2
- Numerical check passed: 2
- Operator check and timing passed: 2
- Program check and timing passed: 2
- Numerical failures: 0
- Operator failures: 0
- Program failures: 0

## Cases

| Case | Family | Shape | Layout | Numerical | Operator | Affon / PyTorch ms | Program | Affon / PyTorch ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `metal/f32/layer_norm/outer_axis/outer_strided_r129_c257` | normalization | 129 × 257 | strided | correct | [passed](operator_peer/metal__f32__layer_norm__outer_axis__outer_strided_r129_c257.json) | 0.7605 / 0.6047 | [passed](program_peer/metal__f32__layer_norm__outer_axis__outer_strided_r129_c257.json) | 0.7739 / 0.6006 |
| `metal/f32/rms_norm/outer_axis/strided_r129_c257` | normalization | 129 × 257 | strided | correct | [passed](operator_peer/metal__f32__rms_norm__outer_axis__strided_r129_c257.json) | 0.7556 / 0.6737 | [passed](program_peer/metal__f32__rms_norm__outer_axis__strided_r129_c257.json) | 0.7774 / 0.6908 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
