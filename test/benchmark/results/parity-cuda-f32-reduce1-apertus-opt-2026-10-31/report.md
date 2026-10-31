# Parity batch report

Target: `NVIDIA GeForce RTX 3090/7.0.0-38-generic/x86_64`
Catalog SHA-256: `deb5480e0e71baffa9457df61beab75f290ee7dad003171a8619ee70ef2c0961`
Binary SHA-256: `0d97a08a8fc20545876748d34e18370cc3f174df66bf6aa5567e64b87ced6a50`
Timing: 5 iterations × 3 samples × 2 rounds

## Checks

- Cases run: 7
- Numerical check passed: 7
- Operator check and timing passed: 7
- Program check and timing passed: 7
- Numerical failures: 0
- Operator failures: 0
- Program failures: 0

## Cases

| Case | Family | Shape | Layout | Numerical | Operator | Affon / PyTorch ms | Program | Affon / PyTorch ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `cuda/f32/argmax_all/long_axis` | reduction | 8 × 512 | dense | correct | [passed](operator_peer/cuda__f32__argmax_all__long_axis.json) | 0.02767 / 0.01279 | [passed](program_peer/cuda__f32__argmax_all__long_axis.json) | 0.03308 / 0.01602 |
| `cuda/f32/argmin_all/long_axis` | reduction | 8 × 512 | dense | correct | [passed](operator_peer/cuda__f32__argmin_all__long_axis.json) | 0.02841 / 0.01465 | [passed](program_peer/cuda__f32__argmin_all__long_axis.json) | 0.03274 / 0.01613 |
| `cuda/f32/max_all/long_axis` | reduction | 8 × 512 | dense | correct | [passed](operator_peer/cuda__f32__max_all__long_axis.json) | 0.02826 / 0.01363 | [passed](program_peer/cuda__f32__max_all__long_axis.json) | 0.03332 / 0.01615 |
| `cuda/f32/mean_all/long_axis` | reduction | 8 × 512 | dense | correct | [passed](operator_peer/cuda__f32__mean_all__long_axis.json) | 0.02721 / 0.0143 | [passed](program_peer/cuda__f32__mean_all__long_axis.json) | 0.03146 / 0.01589 |
| `cuda/f32/min_all/long_axis` | reduction | 8 × 512 | dense | correct | [passed](operator_peer/cuda__f32__min_all__long_axis.json) | 0.02868 / 0.01346 | [passed](program_peer/cuda__f32__min_all__long_axis.json) | 0.03307 / 0.01667 |
| `cuda/f32/std_all/long_axis` | reduction | 8 × 512 | dense | correct | [passed](operator_peer/cuda__f32__std_all__long_axis.json) | 0.02755 / 0.01375 | [passed](program_peer/cuda__f32__std_all__long_axis.json) | 0.03221 / 0.01723 |
| `cuda/f32/variance_all/long_axis` | reduction | 8 × 512 | dense | correct | [passed](operator_peer/cuda__f32__variance_all__long_axis.json) | 0.02736 / 0.01361 | [passed](program_peer/cuda__f32__variance_all__long_axis.json) | 0.03222 / 0.01679 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
