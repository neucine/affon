# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `39008773ba163f10df3adfc51746f8d54651b3e8f18e7959a3f34da5232fc77a`  
Binary SHA-256: `94e3c1795a7cbcae6e6202b022ce83d5f4d5a69f03d81ab1e9de230af9c7bc4e`  
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
| `metal/bf16/layer_norm/long_row` | normalization | 128 × 512 | dense | correct | [passed](operator_peer/metal__bf16__layer_norm__long_row.json) | 0.7677 / 0.8806 | [passed](program_peer/metal__bf16__layer_norm__long_row.json) | 0.8008 / 0.8789 |
| `metal/bf16/rms_norm/long_row` | normalization | 128 × 512 | dense | correct | [passed](operator_peer/metal__bf16__rms_norm__long_row.json) | 0.7669 / 1.028 | [passed](program_peer/metal__bf16__rms_norm__long_row.json) | 0.8068 / 1.076 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
