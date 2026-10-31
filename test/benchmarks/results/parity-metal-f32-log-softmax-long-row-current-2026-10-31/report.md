# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `1eeb18ef3ee998fceed0aaecff1f5c6ef5c2a3676012e39f8c92f2fa487ec76f`  
Binary SHA-256: `94e3c1795a7cbcae6e6202b022ce83d5f4d5a69f03d81ab1e9de230af9c7bc4e`  
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
| `metal/f32/log_softmax/long_row` | normalization | 128 × 512 | dense | correct | [passed](operator_peer/metal__f32__log_softmax__long_row.json) | 0.7479 / 0.9002 | [passed](program_peer/metal__f32__log_softmax__long_row.json) | 0.7611 / 0.9195 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
