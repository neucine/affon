# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `eb91914c217177c3b2c00c5c171dbce7476e1c66aaefb490d01aaf24da18b993`  
Binary SHA-256: `f25caea82a00bafd045213e187e5aa8233a760c352d2f4df4d3079e60d2aaadd`  
Timing: 30 iterations × 5 samples × 3 rounds

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
| `metal/f32/dot/ragged` | matrix | 257 | dense | correct | [passed](operator_peer/metal__f32__dot__ragged.json) | 0.1912 / 0.208 | [passed](program_peer/metal__f32__dot__ragged.json) | 0.1785 / 0.2103 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
