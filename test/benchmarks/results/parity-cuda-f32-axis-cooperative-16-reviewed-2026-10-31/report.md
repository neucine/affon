# Parity batch report

Target: `NVIDIA GeForce RTX 3090/7.0.0-38-generic/x86_64`  
Catalog SHA-256: `6a65425a8368ac059b9af6b5134c4b8c2c652955ad6754e2b8ea46e31437c85b`  
Binary SHA-256: `e16b2cff9df06c2433b2aebcc06d11fd2cc9547534ec38eab6ece62e22be829b`  
Timing: 10 iterations × 5 samples × 3 rounds

## Checks

- Cases run: 16
- Numerical check passed: 16
- Operator check and timing passed: 16
- Program check and timing passed: 16
- Numerical failures: 0
- Operator failures: 0
- Program failures: 0

## Cases

| Case | Family | Shape | Layout | Numerical | Operator | Affon / PyTorch ms | Program | Affon / PyTorch ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `cuda/f32/argmax_axis/short_axis` | reduction | 4 × 8 | dense | correct | [passed](operator_peer/cuda__f32__argmax_axis__short_axis.json) | 0.0224 / 0.0119 | [passed](program_peer/cuda__f32__argmax_axis__short_axis.json) | 0.02747 / 0.01477 |
| `cuda/f32/argmin_axis/short_axis` | reduction | 4 × 8 | dense | correct | [passed](operator_peer/cuda__f32__argmin_axis__short_axis.json) | 0.02269 / 0.01185 | [passed](program_peer/cuda__f32__argmin_axis__short_axis.json) | 0.02747 / 0.0151 |
| `cuda/f32/max_axis/short_axis` | reduction | 4 × 32 | dense | correct | [passed](operator_peer/cuda__f32__max_axis__short_axis.json) | 0.02966 / 0.0133 | [passed](program_peer/cuda__f32__max_axis__short_axis.json) | 0.03444 / 0.0161 |
| `cuda/f32/mean_axis/short_axis` | reduction | 4 × 32 | dense | correct | [passed](operator_peer/cuda__f32__mean_axis__short_axis.json) | 0.02904 / 0.01241 | [passed](program_peer/cuda__f32__mean_axis__short_axis.json) | 0.03366 / 0.01537 |
| `cuda/f32/min_axis/short_axis` | reduction | 4 × 32 | dense | correct | [passed](operator_peer/cuda__f32__min_axis__short_axis.json) | 0.02951 / 0.01308 | [passed](program_peer/cuda__f32__min_axis__short_axis.json) | 0.03445 / 0.01607 |
| `cuda/f32/std_axis/short_axis` | reduction | 4 × 32 | dense | correct | [passed](operator_peer/cuda__f32__std_axis__short_axis.json) | 0.02904 / 0.01298 | [passed](program_peer/cuda__f32__std_axis__short_axis.json) | 0.03375 / 0.01713 |
| `cuda/f32/sum_axis/short_axis` | reduction | 4 × 32 | dense | correct | [passed](operator_peer/cuda__f32__sum_axis__short_axis.json) | 0.02598 / 0.01237 | [passed](program_peer/cuda__f32__sum_axis__short_axis.json) | 0.03074 / 0.01499 |
| `cuda/f32/variance_axis/short_axis` | reduction | 4 × 32 | dense | correct | [passed](operator_peer/cuda__f32__variance_axis__short_axis.json) | 0.02892 / 0.01299 | [passed](program_peer/cuda__f32__variance_axis__short_axis.json) | 0.03376 / 0.01586 |
| `cuda/f32/argmax_axis/long_axis` | reduction | 8 × 512 | dense | correct | [passed](operator_peer/cuda__f32__argmax_axis__long_axis.json) | 0.01898 / 0.01889 | [passed](program_peer/cuda__f32__argmax_axis__long_axis.json) | 0.02385 / 0.01495 |
| `cuda/f32/argmin_axis/long_axis` | reduction | 8 × 512 | dense | correct | [passed](operator_peer/cuda__f32__argmin_axis__long_axis.json) | 0.01912 / 0.01807 | [passed](program_peer/cuda__f32__argmin_axis__long_axis.json) | 0.02399 / 0.01498 |
| `cuda/f32/max_axis/long_axis` | reduction | 128 × 768 | dense | correct | [passed](operator_peer/cuda__f32__max_axis__long_axis.json) | 0.02164 / 0.02263 | [passed](program_peer/cuda__f32__max_axis__long_axis.json) | 0.0264 / 0.01979 |
| `cuda/f32/mean_axis/long_axis` | reduction | 128 × 768 | dense | correct | [passed](operator_peer/cuda__f32__mean_axis__long_axis.json) | 0.02102 / 0.01384 | [passed](program_peer/cuda__f32__mean_axis__long_axis.json) | 0.02593 / 0.0162 |
| `cuda/f32/min_axis/long_axis` | reduction | 128 × 768 | dense | correct | [passed](operator_peer/cuda__f32__min_axis__long_axis.json) | 0.02171 / 0.02278 | [passed](program_peer/cuda__f32__min_axis__long_axis.json) | 0.0267 / 0.01994 |
| `cuda/f32/std_axis/long_axis` | reduction | 128 × 768 | dense | correct | [passed](operator_peer/cuda__f32__std_axis__long_axis.json) | 0.02107 / 0.02239 | [passed](program_peer/cuda__f32__std_axis__long_axis.json) | 0.02575 / 0.02552 |
| `cuda/f32/sum_axis/long_axis` | reduction | 128 × 768 | dense | correct | [passed](operator_peer/cuda__f32__sum_axis__long_axis.json) | 0.02085 / 0.01325 | [passed](program_peer/cuda__f32__sum_axis__long_axis.json) | 0.02604 / 0.01657 |
| `cuda/f32/variance_axis/long_axis` | reduction | 128 × 768 | dense | correct | [passed](operator_peer/cuda__f32__variance_axis__long_axis.json) | 0.02103 / 0.02251 | [passed](program_peer/cuda__f32__variance_axis__long_axis.json) | 0.02593 / 0.02516 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
