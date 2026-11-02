# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `2ba0b3b51820d16adddf937e94fdd378d43a00b2e2002567855f156877cb5f55`  
Binary SHA-256: `164de0a2a31d0861b68c9ed0b40373fa5509390338cc05ee342db6d317df4f82`  
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
| `metal/f32/rms_norm/outer_axis/strided_r33_c65` | normalization | 33 × 65 | strided | correct | [passed](operator_peer/metal__f32__rms_norm__outer_axis__strided_r33_c65.json) | 0.5491 / 0.9334 | [passed](program_peer/metal__f32__rms_norm__outer_axis__strided_r33_c65.json) | 0.5727 / 1.052 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
