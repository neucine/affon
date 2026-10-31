# Parity batch report

Target: `NVIDIA GeForce RTX 3090/7.0.0-38-generic/x86_64`  
Catalog SHA-256: `6da592b69b2de6d3081f24268332ea84f199527b132c695a3b656f7a0fa504c5`  
Binary SHA-256: `d8ff60ca08145d266436804933fa2c2888b125251f4b1b340af29ced2eaeabaf`  
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
| `cuda/f32/argmax_axis/long_axis` | reduction | 8 × 512 | dense | correct | [passed](operator_peer/cuda__f32__argmax_axis__long_axis.json) | 0.01904 / 0.01964 | [passed](program_peer/cuda__f32__argmax_axis__long_axis.json) | 0.0238 / 0.01512 |
| `cuda/f32/mean_axis/long_axis` | reduction | 128 × 768 | dense | correct | [passed](operator_peer/cuda__f32__mean_axis__long_axis.json) | 0.02111 / 0.01447 | [passed](program_peer/cuda__f32__mean_axis__long_axis.json) | 0.02585 / 0.01655 |
| `cuda/f32/sum_axis/long_axis` | reduction | 128 × 768 | dense | correct | [passed](operator_peer/cuda__f32__sum_axis__long_axis.json) | 0.0209 / 0.01479 | [passed](program_peer/cuda__f32__sum_axis__long_axis.json) | 0.02599 / 0.01648 |
| `cuda/f32/variance_axis/long_axis` | reduction | 128 × 768 | dense | correct | [passed](operator_peer/cuda__f32__variance_axis__long_axis.json) | 0.0211 / 0.02253 | [passed](program_peer/cuda__f32__variance_axis__long_axis.json) | 0.02598 / 0.02488 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
