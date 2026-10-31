# Parity batch report

Target: `NVIDIA GeForce RTX 3090/7.0.0-38-generic/x86_64`  
Catalog SHA-256: `4cba91e2ddf0ccb01e6e6448561295781b6aa809f2b731ca9d8ea5d1820cca23`  
Binary SHA-256: `0c165f3b7327de6659b973611275ee22091ae5ad49b75aa24bcea4e9fc1fe67d`  
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
| `cuda/f32/argmax_all/short_axis` | reduction | 4 × 8 | dense | correct | [passed](operator_peer/cuda__f32__argmax_all__short_axis.json) | 0.1107 / 0.01265 | [passed](program_peer/cuda__f32__argmax_all__short_axis.json) | 0.1157 / 0.016 |
| `cuda/f32/argmin_all/short_axis` | reduction | 4 × 8 | dense | correct | [passed](operator_peer/cuda__f32__argmin_all__short_axis.json) | 0.1108 / 0.01276 | [passed](program_peer/cuda__f32__argmin_all__short_axis.json) | 0.1155 / 0.0155 |
| `cuda/f32/max_all/short_axis` | reduction | 4 × 8 | dense | correct | [passed](operator_peer/cuda__f32__max_all__short_axis.json) | 0.1109 / 0.01318 | [passed](program_peer/cuda__f32__max_all__short_axis.json) | 0.1156 / 0.0159 |
| `cuda/f32/mean_all/short_axis` | reduction | 4 × 8 | dense | correct | [passed](operator_peer/cuda__f32__mean_all__short_axis.json) | 0.1108 / 0.01304 | [passed](program_peer/cuda__f32__mean_all__short_axis.json) | 0.114 / 0.01579 |
| `cuda/f32/min_all/short_axis` | reduction | 4 × 8 | dense | correct | [passed](operator_peer/cuda__f32__min_all__short_axis.json) | 0.1107 / 0.01287 | [passed](program_peer/cuda__f32__min_all__short_axis.json) | 0.1141 / 0.01593 |
| `cuda/f32/std_all/short_axis` | reduction | 4 × 8 | dense | correct | [passed](operator_peer/cuda__f32__std_all__short_axis.json) | 0.1099 / 0.01356 | [passed](program_peer/cuda__f32__std_all__short_axis.json) | 0.1157 / 0.01688 |
| `cuda/f32/variance_all/short_axis` | reduction | 4 × 8 | dense | correct | [passed](operator_peer/cuda__f32__variance_all__short_axis.json) | 0.1109 / 0.01351 | [passed](program_peer/cuda__f32__variance_all__short_axis.json) | 0.1155 / 0.01676 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
