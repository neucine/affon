# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `1dc7eec93310882600b771747976afd4e6f4d438d258f9c61ae162b10c004205`  
Binary SHA-256: `3fa744765b035d3c4a6b32e9e6968cda341a29dd8092e42a6a44d3e0ecf0287f`  
Timing: 10 iterations × 5 samples × 3 rounds

## Checks

- Cases run: 2
- Numerical check passed: 1
- Operator check and timing passed: 1
- Program check and timing passed: 1
- Numerical failures: 1
- Operator failures: 0
- Program failures: 0

## Cases

| Case | Family | Shape | Layout | Numerical | Operator | Affon / PyTorch ms | Program | Affon / PyTorch ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `metal/f32/rms_norm/outer_axis/strided_r3_c5` | normalization | 3 × 5 | strided | correct | [passed](operator_peer/metal__f32__rms_norm__outer_axis__strided_r3_c5.json) | 0.0095 / 0.7207 | [passed](program_peer/metal__f32__rms_norm__outer_axis__strided_r3_c5.json) | 0.2443 / 0.5059 |
| `metal/f32/rms_norm/outer_axis/strided_r33_c65` | normalization | 33 × 65 | strided | incorrect | probe_not_correct | — | probe_not_correct | — |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
