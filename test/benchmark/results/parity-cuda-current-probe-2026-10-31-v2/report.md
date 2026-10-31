# Parity batch report

Target: `NVIDIA GeForce RTX 3090/7.0.0-38-generic/x86_64`  
Catalog SHA-256: `df7af9e709c60087fd67a278fac51136b75f1734e1032d946fa161623955dff8`  
Binary SHA-256: `8416d7afd6d306a9f5a56e6e4d9646b723f74880d9bf6c817509cf18e967f368`  
Timing: 10 iterations × 5 samples × 3 rounds

## Checks

- Cases run: 3
- Numerical check passed: 3
- Operator check and timing passed: 3
- Program check and timing passed: 3
- Numerical failures: 0
- Operator failures: 0
- Program failures: 0

## Cases

| Case | Family | Shape | Layout | Numerical | Operator | Affon / PyTorch ms | Program | Affon / PyTorch ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `cuda/f32/argmax_all/long_axis` | reduction | 8 × 512 | dense | correct | [passed](operator_peer/cuda__f32__argmax_all__long_axis.json) | 11.08 / 0.01269 | [passed](program_peer/cuda__f32__argmax_all__long_axis.json) | 11.09 / 0.01634 |
| `cuda/f32/mean_all/long_axis` | reduction | 8 × 512 | dense | correct | [passed](operator_peer/cuda__f32__mean_all__long_axis.json) | 11.04 / 0.01294 | [passed](program_peer/cuda__f32__mean_all__long_axis.json) | 11.05 / 0.01622 |
| `cuda/f32/sum_axis/long_axis` | reduction | 128 × 768 | dense | correct | [passed](operator_peer/cuda__f32__sum_axis__long_axis.json) | 3.087 / 0.01333 | [passed](program_peer/cuda__f32__sum_axis__long_axis.json) | 3.092 / 0.01587 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
