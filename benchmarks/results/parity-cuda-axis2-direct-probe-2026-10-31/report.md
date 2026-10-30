# Parity batch report

Target: `NVIDIA GeForce RTX 3090/7.0.0-38-generic/x86_64`  
Catalog SHA-256: `6da592b69b2de6d3081f24268332ea84f199527b132c695a3b656f7a0fa504c5`  
Binary SHA-256: `b8761b54238e1ff86cd725fabd5aa974d71642ce8488159ec5b57961bb8b0742`  
Timing: 10 iterations × 5 samples × 3 rounds

## Checks

- Cases run: 4
- Numerical check passed: 4
- Operator check and timing passed: 4
- Program check and timing passed: 4
- Numerical failures: 0
- Operator failures: 0
- Program failures: 0

## Cases

| Case | Family | Shape | Layout | Numerical | Operator | Affon / PyTorch ms | Program | Affon / PyTorch ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `cuda/f32/argmax_axis/long_axis` | reduction | 8 × 512 | dense | correct | [passed](operator_peer/cuda__f32__argmax_axis__long_axis.json) | 0.04498 / 0.01975 | [passed](program_peer/cuda__f32__argmax_axis__long_axis.json) | 0.04956 / 0.01523 |
| `cuda/f32/mean_axis/long_axis` | reduction | 128 × 768 | dense | correct | [passed](operator_peer/cuda__f32__mean_axis__long_axis.json) | 0.1232 / 0.01372 | [passed](program_peer/cuda__f32__mean_axis__long_axis.json) | 0.1279 / 0.01624 |
| `cuda/f32/sum_axis/long_axis` | reduction | 128 × 768 | dense | correct | [passed](operator_peer/cuda__f32__sum_axis__long_axis.json) | 0.1231 / 0.01389 | [passed](program_peer/cuda__f32__sum_axis__long_axis.json) | 0.1279 / 0.01633 |
| `cuda/f32/variance_axis/long_axis` | reduction | 128 × 768 | dense | correct | [passed](operator_peer/cuda__f32__variance_axis__long_axis.json) | 0.1454 / 0.02267 | [passed](program_peer/cuda__f32__variance_axis__long_axis.json) | 0.1501 / 0.02534 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
