# Parity batch report

Target: `NVIDIA GeForce RTX 3090/7.0.0-38-generic/x86_64`  
Catalog SHA-256: `6a65425a8368ac059b9af6b5134c4b8c2c652955ad6754e2b8ea46e31437c85b`  
Binary SHA-256: `d8ff60ca08145d266436804933fa2c2888b125251f4b1b340af29ced2eaeabaf`  
Timing: 10 iterations × 5 samples × 3 rounds

## Checks

- Cases run: 16
- Numerical check passed: 15
- Operator check and timing passed: 15
- Program check and timing passed: 15
- Numerical failures: 1
- Operator failures: 0
- Program failures: 0

## Cases

| Case | Family | Shape | Layout | Numerical | Operator | Affon / PyTorch ms | Program | Affon / PyTorch ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `cuda/f32/argmax_axis/short_axis` | reduction | 4 × 8 | dense | correct | [passed](operator_peer/cuda__f32__argmax_axis__short_axis.json) | 0.01742 / 0.01181 | [passed](program_peer/cuda__f32__argmax_axis__short_axis.json) | 0.0223 / 0.01491 |
| `cuda/f32/argmin_axis/short_axis` | reduction | 4 × 8 | dense | correct | [passed](operator_peer/cuda__f32__argmin_axis__short_axis.json) | 0.0174 / 0.01181 | [passed](program_peer/cuda__f32__argmin_axis__short_axis.json) | 0.02231 / 0.01512 |
| `cuda/f32/max_axis/short_axis` | reduction | 4 × 32 | dense | correct | [passed](operator_peer/cuda__f32__max_axis__short_axis.json) | 0.0203 / 0.01322 | [passed](program_peer/cuda__f32__max_axis__short_axis.json) | 0.02497 / 0.01649 |
| `cuda/f32/mean_axis/short_axis` | reduction | 4 × 32 | dense | correct | [passed](operator_peer/cuda__f32__mean_axis__short_axis.json) | 0.02045 / 0.01219 | [passed](program_peer/cuda__f32__mean_axis__short_axis.json) | 0.02506 / 0.01512 |
| `cuda/f32/min_axis/short_axis` | reduction | 4 × 32 | dense | incorrect | probe_not_correct | — | probe_not_correct | — |
| `cuda/f32/std_axis/short_axis` | reduction | 4 × 32 | dense | correct | [passed](operator_peer/cuda__f32__std_axis__short_axis.json) | 0.01956 / 0.01279 | [passed](program_peer/cuda__f32__std_axis__short_axis.json) | 0.02431 / 0.01585 |
| `cuda/f32/sum_axis/short_axis` | reduction | 4 × 32 | dense | correct | [passed](operator_peer/cuda__f32__sum_axis__short_axis.json) | 0.0203 / 0.01199 | [passed](program_peer/cuda__f32__sum_axis__short_axis.json) | 0.02512 / 0.0151 |
| `cuda/f32/variance_axis/short_axis` | reduction | 4 × 32 | dense | correct | [passed](operator_peer/cuda__f32__variance_axis__short_axis.json) | 0.01952 / 0.01279 | [passed](program_peer/cuda__f32__variance_axis__short_axis.json) | 0.02432 / 0.0156 |
| `cuda/f32/argmax_axis/long_axis` | reduction | 8 × 512 | dense | correct | [passed](operator_peer/cuda__f32__argmax_axis__long_axis.json) | 0.01885 / 0.01892 | [passed](program_peer/cuda__f32__argmax_axis__long_axis.json) | 0.02387 / 0.0149 |
| `cuda/f32/argmin_axis/long_axis` | reduction | 8 × 512 | dense | correct | [passed](operator_peer/cuda__f32__argmin_axis__long_axis.json) | 0.01911 / 0.01891 | [passed](program_peer/cuda__f32__argmin_axis__long_axis.json) | 0.02383 / 0.01485 |
| `cuda/f32/max_axis/long_axis` | reduction | 128 × 768 | dense | correct | [passed](operator_peer/cuda__f32__max_axis__long_axis.json) | 0.02144 / 0.02292 | [passed](program_peer/cuda__f32__max_axis__long_axis.json) | 0.0263 / 0.02226 |
| `cuda/f32/mean_axis/long_axis` | reduction | 128 × 768 | dense | correct | [passed](operator_peer/cuda__f32__mean_axis__long_axis.json) | 0.02103 / 0.01437 | [passed](program_peer/cuda__f32__mean_axis__long_axis.json) | 0.02581 / 0.01643 |
| `cuda/f32/min_axis/long_axis` | reduction | 128 × 768 | dense | correct | [passed](operator_peer/cuda__f32__min_axis__long_axis.json) | 0.02184 / 0.02268 | [passed](program_peer/cuda__f32__min_axis__long_axis.json) | 0.02668 / 0.02364 |
| `cuda/f32/std_axis/long_axis` | reduction | 128 × 768 | dense | correct | [passed](operator_peer/cuda__f32__std_axis__long_axis.json) | 0.02101 / 0.02185 | [passed](program_peer/cuda__f32__std_axis__long_axis.json) | 0.02593 / 0.02564 |
| `cuda/f32/sum_axis/long_axis` | reduction | 128 × 768 | dense | correct | [passed](operator_peer/cuda__f32__sum_axis__long_axis.json) | 0.02105 / 0.01523 | [passed](program_peer/cuda__f32__sum_axis__long_axis.json) | 0.02578 / 0.01648 |
| `cuda/f32/variance_axis/long_axis` | reduction | 128 × 768 | dense | correct | [passed](operator_peer/cuda__f32__variance_axis__long_axis.json) | 0.02106 / 0.02223 | [passed](program_peer/cuda__f32__variance_axis__long_axis.json) | 0.02587 / 0.0249 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
