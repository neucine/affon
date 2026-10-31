# Parity batch report

Target: `NVIDIA GeForce RTX 3090/7.0.0-38-generic/x86_64`  
Catalog SHA-256: `8d240d7fbc14fa5c1cff8743e309ab8ccba2b97f7737f39a93fbb4cbabcb6746`  
Binary SHA-256: `95449c1e54d7ab4e63d84e3fdbee6de9b393a4af7615c98f9e9a51f45cf1d4c6`  
Timing: 10 iterations × 5 samples × 3 rounds

## Checks

- Cases run: 6
- Numerical check passed: 6
- Operator check and timing passed: 6
- Program check and timing passed: 6
- Numerical failures: 0
- Operator failures: 0
- Program failures: 0

## Cases

| Case | Family | Shape | Layout | Numerical | Operator | Affon / PyTorch ms | Program | Affon / PyTorch ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `cuda/bf16/argmax_all/short_axis` | reduction | 4 × 8 | dense | correct | [passed](operator_peer/cuda__bf16__argmax_all__short_axis.json) | 0.0319 / 0.01263 | [passed](program_peer/cuda__bf16__argmax_all__short_axis.json) | 0.03661 / 0.01606 |
| `cuda/bf16/argmin_all/short_axis` | reduction | 4 × 8 | dense | correct | [passed](operator_peer/cuda__bf16__argmin_all__short_axis.json) | 0.03164 / 0.01273 | [passed](program_peer/cuda__bf16__argmin_all__short_axis.json) | 0.03659 / 0.01555 |
| `cuda/bf16/argmax_all/long_axis` | reduction | 8 × 512 | dense | correct | [passed](operator_peer/cuda__bf16__argmax_all__long_axis.json) | 0.0321 / 0.01295 | [passed](program_peer/cuda__bf16__argmax_all__long_axis.json) | 0.03675 / 0.01591 |
| `cuda/bf16/argmin_all/long_axis` | reduction | 8 × 512 | dense | correct | [passed](operator_peer/cuda__bf16__argmin_all__long_axis.json) | 0.0323 / 0.01295 | [passed](program_peer/cuda__bf16__argmin_all__long_axis.json) | 0.03705 / 0.01614 |
| `cuda/bf16/argmax_all/outer_axis` | reduction | 128 × 256 | dense | correct | [passed](operator_peer/cuda__bf16__argmax_all__outer_axis.json) | 0.06013 / 0.02158 | [passed](program_peer/cuda__bf16__argmax_all__outer_axis.json) | 0.06474 / 0.02474 |
| `cuda/bf16/argmin_all/outer_axis` | reduction | 128 × 256 | dense | correct | [passed](operator_peer/cuda__bf16__argmin_all__outer_axis.json) | 0.05924 / 0.02151 | [passed](program_peer/cuda__bf16__argmin_all__outer_axis.json) | 0.06398 / 0.02428 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
