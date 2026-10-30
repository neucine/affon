# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `202a1ea732a0414cef2706c6befc4ad0d852b29a0ca53d4558302259bb1909af`  
Binary SHA-256: `057e9fb9bc293dc9eb30a5874e868990448947a4d4863271901d4fd681711bf0`  
Timing: 5 iterations × 5 samples × 3 rounds

## Coverage

- Candidate cells: 1
- Numerical capability passed: 1
- Operator parity and timing passed: 1
- Program parity and timing passed: 1
- Probe failures or incorrect: 0
- Operator incorrect or error: 0
- Program incorrect or error: 0

## Cells

| Cell | Probe | Operator parity | Affon / peer ms | Program parity | Affon / peer ms |
| --- | --- | --- | --- | --- | --- |
| `metal/f32/rms_norm/short_row/r8_c7` | correct | [passed](operator_peer/metal__f32__rms_norm__short_row__r8_c7.json) | 0.3304 / 0.2432 | [passed](program_peer/metal__f32__rms_norm__short_row__r8_c7.json) | 0.3238 / 0.237 |

Each probe result is retained in [capability.json](capability.json).
Timing ratios are diagnostic; only reports with passing numerical parity are eligible for coverage links.
