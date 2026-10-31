# Parity batch report

Target: `NVIDIA GeForce RTX 3090/7.0.0-38-generic/x86_64`  
Catalog SHA-256: `deb5480e0e71baffa9457df61beab75f290ee7dad003171a8619ee70ef2c0961`  
Binary SHA-256: `5935499eea7d4fd6b5cfbcce670dedcac747c411e5515df9cf3dd7f0d7a956d0`  
Timing: 10 iterations × 5 samples × 3 rounds

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
| `cuda/f32/argmax_all/long_axis` | reduction | 8 × 512 | dense | correct | [passed](operator_peer/cuda__f32__argmax_all__long_axis.json) | 0.02777 / 0.01263 | [passed](program_peer/cuda__f32__argmax_all__long_axis.json) | 0.03266 / 0.01607 |
| `cuda/f32/argmin_all/long_axis` | reduction | 8 × 512 | dense | correct | [passed](operator_peer/cuda__f32__argmin_all__long_axis.json) | 0.02762 / 0.01269 | [passed](program_peer/cuda__f32__argmin_all__long_axis.json) | 0.03252 / 0.01584 |
| `cuda/f32/max_all/long_axis` | reduction | 8 × 512 | dense | correct | [passed](operator_peer/cuda__f32__max_all__long_axis.json) | 0.02802 / 0.01327 | [passed](program_peer/cuda__f32__max_all__long_axis.json) | 0.03266 / 0.01593 |
| `cuda/f32/mean_all/long_axis` | reduction | 8 × 512 | dense | correct | [passed](operator_peer/cuda__f32__mean_all__long_axis.json) | 0.02701 / 0.01306 | [passed](program_peer/cuda__f32__mean_all__long_axis.json) | 0.03166 / 0.01599 |
| `cuda/f32/min_all/long_axis` | reduction | 8 × 512 | dense | correct | [passed](operator_peer/cuda__f32__min_all__long_axis.json) | 0.02824 / 0.01295 | [passed](program_peer/cuda__f32__min_all__long_axis.json) | 0.03299 / 0.01586 |
| `cuda/f32/std_all/long_axis` | reduction | 8 × 512 | dense | correct | [passed](operator_peer/cuda__f32__std_all__long_axis.json) | 0.0272 / 0.01344 | [passed](program_peer/cuda__f32__std_all__long_axis.json) | 0.03176 / 0.01658 |
| `cuda/f32/variance_all/long_axis` | reduction | 8 × 512 | dense | correct | [passed](operator_peer/cuda__f32__variance_all__long_axis.json) | 0.02704 / 0.01348 | [passed](program_peer/cuda__f32__variance_all__long_axis.json) | 0.03175 / 0.01662 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
