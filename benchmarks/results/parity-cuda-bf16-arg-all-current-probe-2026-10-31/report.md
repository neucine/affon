# Parity batch report

Target: `NVIDIA GeForce RTX 3090/7.0.0-38-generic/x86_64`  
Catalog SHA-256: `8d240d7fbc14fa5c1cff8743e309ab8ccba2b97f7737f39a93fbb4cbabcb6746`  
Binary SHA-256: `508a84ce0f83a2fd840b6f71178ef78af228498c129a52c35b7275ef7051b79a`  
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
| `cuda/bf16/argmax_all/short_axis` | reduction | 4 × 8 | dense | correct | [passed](operator_peer/cuda__bf16__argmax_all__short_axis.json) | 0.03128 / 0.01268 | [passed](program_peer/cuda__bf16__argmax_all__short_axis.json) | 0.03658 / 0.01544 |
| `cuda/bf16/argmin_all/short_axis` | reduction | 4 × 8 | dense | correct | [passed](operator_peer/cuda__bf16__argmin_all__short_axis.json) | 0.03151 / 0.01262 | [passed](program_peer/cuda__bf16__argmin_all__short_axis.json) | 0.03654 / 0.0155 |
| `cuda/bf16/argmax_all/long_axis` | reduction | 8 × 512 | dense | correct | [passed](operator_peer/cuda__bf16__argmax_all__long_axis.json) | 0.0319 / 0.01264 | [passed](program_peer/cuda__bf16__argmax_all__long_axis.json) | 0.03666 / 0.01575 |
| `cuda/bf16/argmin_all/long_axis` | reduction | 8 × 512 | dense | correct | [passed](operator_peer/cuda__bf16__argmin_all__long_axis.json) | 0.03233 / 0.01259 | [passed](program_peer/cuda__bf16__argmin_all__long_axis.json) | 0.03707 / 0.01558 |
| `cuda/bf16/argmax_all/outer_axis` | reduction | 128 × 256 | dense | correct | [passed](operator_peer/cuda__bf16__argmax_all__outer_axis.json) | 0.1107 / 0.02147 | [passed](program_peer/cuda__bf16__argmax_all__outer_axis.json) | 0.1157 / 0.02436 |
| `cuda/bf16/argmin_all/outer_axis` | reduction | 128 × 256 | dense | correct | [passed](operator_peer/cuda__bf16__argmin_all__outer_axis.json) | 0.1112 / 0.02108 | [passed](program_peer/cuda__bf16__argmin_all__outer_axis.json) | 0.1161 / 0.024 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
