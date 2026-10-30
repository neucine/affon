# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `1c710893344992705dc5b7317514f7ef69982db28747d05c88574005c9a31fc0`  
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
| `metal/f32/rms_norm/long_row` | normalization | 128 × 512 | dense | correct | [passed](operator_peer/metal__f32__rms_norm__long_row.json) | 0.7627 / 1.014 | [passed](program_peer/metal__f32__rms_norm__long_row.json) | 0.7647 / 1.001 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
