# Parity batch report

Target: `NVIDIA GeForce RTX 3090/7.0.0-38-generic/x86_64`  
Catalog SHA-256: `8d240d7fbc14fa5c1cff8743e309ab8ccba2b97f7737f39a93fbb4cbabcb6746`  
Binary SHA-256: `dc9721ddc85560d6402c3e666e807aec17725c9261baf4a3b5778df00d2a9958`  
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
| `cuda/bf16/argmax_all/short_axis` | reduction | 4 × 8 | dense | correct | [passed](operator_peer/cuda__bf16__argmax_all__short_axis.json) | 0.03157 / 0.01267 | [passed](program_peer/cuda__bf16__argmax_all__short_axis.json) | 0.03679 / 0.0155 |
| `cuda/bf16/argmin_all/short_axis` | reduction | 4 × 8 | dense | correct | [passed](operator_peer/cuda__bf16__argmin_all__short_axis.json) | 0.03208 / 0.01293 | [passed](program_peer/cuda__bf16__argmin_all__short_axis.json) | 0.03651 / 0.01529 |
| `cuda/bf16/argmax_all/long_axis` | reduction | 8 × 512 | dense | correct | [passed](operator_peer/cuda__bf16__argmax_all__long_axis.json) | 0.03188 / 0.01279 | [passed](program_peer/cuda__bf16__argmax_all__long_axis.json) | 0.03674 / 0.01609 |
| `cuda/bf16/argmin_all/long_axis` | reduction | 8 × 512 | dense | correct | [passed](operator_peer/cuda__bf16__argmin_all__long_axis.json) | 0.03238 / 0.01299 | [passed](program_peer/cuda__bf16__argmin_all__long_axis.json) | 0.03725 / 0.01619 |
| `cuda/bf16/argmax_all/outer_axis` | reduction | 128 × 256 | dense | correct | [passed](operator_peer/cuda__bf16__argmax_all__outer_axis.json) | 0.02325 / 0.02162 | [passed](program_peer/cuda__bf16__argmax_all__outer_axis.json) | 0.03091 / 0.02469 |
| `cuda/bf16/argmin_all/outer_axis` | reduction | 128 × 256 | dense | correct | [passed](operator_peer/cuda__bf16__argmin_all__outer_axis.json) | 0.02318 / 0.02146 | [passed](program_peer/cuda__bf16__argmin_all__outer_axis.json) | 0.03055 / 0.02469 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
