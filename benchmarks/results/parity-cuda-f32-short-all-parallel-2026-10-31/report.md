# Parity batch report

Target: `NVIDIA GeForce RTX 3090/7.0.0-38-generic/x86_64`  
Catalog SHA-256: `4cba91e2ddf0ccb01e6e6448561295781b6aa809f2b731ca9d8ea5d1820cca23`  
Binary SHA-256: `508a84ce0f83a2fd840b6f71178ef78af228498c129a52c35b7275ef7051b79a`  
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
| `cuda/f32/argmax_all/short_axis` | reduction | 4 × 8 | dense | correct | [passed](operator_peer/cuda__f32__argmax_all__short_axis.json) | 0.01855 / 0.01258 | [passed](program_peer/cuda__f32__argmax_all__short_axis.json) | 0.02335 / 0.01584 |
| `cuda/f32/argmin_all/short_axis` | reduction | 4 × 8 | dense | correct | [passed](operator_peer/cuda__f32__argmin_all__short_axis.json) | 0.01875 / 0.01284 | [passed](program_peer/cuda__f32__argmin_all__short_axis.json) | 0.02346 / 0.0155 |
| `cuda/f32/max_all/short_axis` | reduction | 4 × 8 | dense | correct | [passed](operator_peer/cuda__f32__max_all__short_axis.json) | 0.01866 / 0.01286 | [passed](program_peer/cuda__f32__max_all__short_axis.json) | 0.02348 / 0.01578 |
| `cuda/f32/mean_all/short_axis` | reduction | 4 × 8 | dense | correct | [passed](operator_peer/cuda__f32__mean_all__short_axis.json) | 0.01852 / 0.01296 | [passed](program_peer/cuda__f32__mean_all__short_axis.json) | 0.02337 / 0.01567 |
| `cuda/f32/min_all/short_axis` | reduction | 4 × 8 | dense | correct | [passed](operator_peer/cuda__f32__min_all__short_axis.json) | 0.0186 / 0.0131 | [passed](program_peer/cuda__f32__min_all__short_axis.json) | 0.02338 / 0.01577 |
| `cuda/f32/std_all/short_axis` | reduction | 4 × 8 | dense | correct | [passed](operator_peer/cuda__f32__std_all__short_axis.json) | 0.01859 / 0.01381 | [passed](program_peer/cuda__f32__std_all__short_axis.json) | 0.02315 / 0.01618 |
| `cuda/f32/variance_all/short_axis` | reduction | 4 × 8 | dense | correct | [passed](operator_peer/cuda__f32__variance_all__short_axis.json) | 0.01838 / 0.01363 | [passed](program_peer/cuda__f32__variance_all__short_axis.json) | 0.02321 / 0.01678 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
