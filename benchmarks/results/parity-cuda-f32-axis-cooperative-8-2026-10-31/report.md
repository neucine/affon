# Parity batch report

Target: `NVIDIA GeForce RTX 3090/7.0.0-38-generic/x86_64`  
Catalog SHA-256: `8d09f7323b440b2e70dd9a02d75ca53911bdf51f8586e5e4f99a80feae77f0cc`  
Binary SHA-256: `d8ff60ca08145d266436804933fa2c2888b125251f4b1b340af29ced2eaeabaf`  
Timing: 10 iterations × 5 samples × 3 rounds

## Checks

- Cases run: 8
- Numerical check passed: 8
- Operator check and timing passed: 8
- Program check and timing passed: 8
- Numerical failures: 0
- Operator failures: 0
- Program failures: 0

## Cases

| Case | Family | Shape | Layout | Numerical | Operator | Affon / PyTorch ms | Program | Affon / PyTorch ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `cuda/f32/argmax_axis/long_axis` | reduction | 8 × 512 | dense | correct | [passed](operator_peer/cuda__f32__argmax_axis__long_axis.json) | 0.01888 / 0.01969 | [passed](program_peer/cuda__f32__argmax_axis__long_axis.json) | 0.02375 / 0.01487 |
| `cuda/f32/argmin_axis/long_axis` | reduction | 8 × 512 | dense | correct | [passed](operator_peer/cuda__f32__argmin_axis__long_axis.json) | 0.01917 / 0.0195 | [passed](program_peer/cuda__f32__argmin_axis__long_axis.json) | 0.02381 / 0.01507 |
| `cuda/f32/max_axis/long_axis` | reduction | 128 × 768 | dense | correct | [passed](operator_peer/cuda__f32__max_axis__long_axis.json) | 0.02148 / 0.02241 | [passed](program_peer/cuda__f32__max_axis__long_axis.json) | 0.02629 / 0.02074 |
| `cuda/f32/mean_axis/long_axis` | reduction | 128 × 768 | dense | correct | [passed](operator_peer/cuda__f32__mean_axis__long_axis.json) | 0.02107 / 0.01364 | [passed](program_peer/cuda__f32__mean_axis__long_axis.json) | 0.02577 / 0.0164 |
| `cuda/f32/min_axis/long_axis` | reduction | 128 × 768 | dense | correct | [passed](operator_peer/cuda__f32__min_axis__long_axis.json) | 0.0219 / 0.02255 | [passed](program_peer/cuda__f32__min_axis__long_axis.json) | 0.02677 / 0.02258 |
| `cuda/f32/std_axis/long_axis` | reduction | 128 × 768 | dense | correct | [passed](operator_peer/cuda__f32__std_axis__long_axis.json) | 0.02112 / 0.02258 | [passed](program_peer/cuda__f32__std_axis__long_axis.json) | 0.02594 / 0.0255 |
| `cuda/f32/sum_axis/long_axis` | reduction | 128 × 768 | dense | correct | [passed](operator_peer/cuda__f32__sum_axis__long_axis.json) | 0.02091 / 0.01411 | [passed](program_peer/cuda__f32__sum_axis__long_axis.json) | 0.02578 / 0.01644 |
| `cuda/f32/variance_axis/long_axis` | reduction | 128 × 768 | dense | correct | [passed](operator_peer/cuda__f32__variance_axis__long_axis.json) | 0.02105 / 0.02282 | [passed](program_peer/cuda__f32__variance_axis__long_axis.json) | 0.02589 / 0.02565 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
