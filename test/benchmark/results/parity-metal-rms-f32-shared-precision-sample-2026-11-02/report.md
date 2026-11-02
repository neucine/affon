# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `2ba0b3b51820d16adddf937e94fdd378d43a00b2e2002567855f156877cb5f55`  
Binary SHA-256: `ba046a556386008db27593da022c231a5e805c8c24dc9bed1c17109c445305f8`  
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
| `metal/f32/rms_norm/outer_axis/strided_r33_c65` | normalization | 33 × 65 | strided | correct | [passed](operator_peer/metal__f32__rms_norm__outer_axis__strided_r33_c65.json) | 0.0496 / 1.123 | [passed](program_peer/metal__f32__rms_norm__outer_axis__strided_r33_c65.json) | 0.39 / 0.8785 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
