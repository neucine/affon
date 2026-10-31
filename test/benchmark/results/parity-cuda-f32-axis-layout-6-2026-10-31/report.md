# Parity batch report

Target: `NVIDIA GeForce RTX 3090/7.0.0-38-generic/x86_64`  
Catalog SHA-256: `87057793c32cb31333926a5a9b1b1571fe7fba955b00ec0c4da733832b1f1ea1`  
Binary SHA-256: `d8ff60ca08145d266436804933fa2c2888b125251f4b1b340af29ced2eaeabaf`  
Timing: 10 iterations × 5 samples × 3 rounds

## Checks

- Cases run: 6
- Numerical check passed: 6
- Operator check and timing passed: 6
- Program check and timing passed: 6
- Numerical failures: 0
- Operator failures: 0
- Program failures: 0

## Cases

| Case | Family | Shape | Layout | Numerical | Operator | Affon / PyTorch ms | Program | Affon / PyTorch ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `cuda/f32/sum_axis/strided/r8_c512` | reduction | 8 × 512 | strided | correct | [passed](operator_peer/cuda__f32__sum_axis__strided__r8_c512.json) | 0.01842 / 0.0154 | [passed](program_peer/cuda__f32__sum_axis__strided__r8_c512.json) | 0.02331 / 0.01861 |
| `cuda/f32/mean_axis/offset/r8_c512` | reduction | 8 × 512 | offset | correct | [passed](operator_peer/cuda__f32__mean_axis__offset__r8_c512.json) | 0.01861 / 0.01697 | [passed](program_peer/cuda__f32__mean_axis__offset__r8_c512.json) | 0.02338 / 0.02008 |
| `cuda/f32/variance_axis/outer_axis/strided_r512_c8` | reduction | 512 × 8 | strided | correct | [passed](operator_peer/cuda__f32__variance_axis__outer_axis__strided_r512_c8.json) | 0.01894 / 0.02463 | [passed](program_peer/cuda__f32__variance_axis__outer_axis__strided_r512_c8.json) | 0.02377 / 0.01927 |
| `cuda/f32/argmin_axis/outer_axis/offset_r512_c8` | reduction | 512 × 8 | offset | correct | [passed](operator_peer/cuda__f32__argmin_axis__outer_axis__offset_r512_c8.json) | 0.01933 / 0.01888 | [passed](program_peer/cuda__f32__argmin_axis__outer_axis__offset_r512_c8.json) | 0.02436 / 0.01974 |
| `cuda/f32/max_axis/strided/r128_c768` | reduction | 128 × 768 | strided | correct | [passed](operator_peer/cuda__f32__max_axis__strided__r128_c768.json) | 0.02189 / 0.02566 | [passed](program_peer/cuda__f32__max_axis__strided__r128_c768.json) | 0.02656 / 0.02099 |
| `cuda/f32/std_axis/offset/r128_c768` | reduction | 128 × 768 | offset | correct | [passed](operator_peer/cuda__f32__std_axis__offset__r128_c768.json) | 0.02121 / 0.02746 | [passed](program_peer/cuda__f32__std_axis__offset__r128_c768.json) | 0.026 / 0.0298 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
