# Parity batch report

Target: `NVIDIA GeForce RTX 3090/7.0.0-42-generic/x86_64`  
Catalog SHA-256: `1cbbac026e1c863adb033b580d58cb26852e7ffa06ecbec59128ebd58d00bab5`  
Binary SHA-256: `7e4c26cb930ec660505be0e0e0452a7b29207727b0582819b3a1ee8fc3537447`  
Timing: 10 iterations × 5 samples × 3 rounds

## Checks

- Cases run: 12
- Numerical check passed: 12
- Operator check and timing passed: 12
- Program check and timing passed: 12
- Numerical failures: 0
- Operator failures: 0
- Program failures: 0
- Operator checks pending: 0
- Program checks pending: 0

## Cases

| Case | Family | Shape | Layout | Numerical | Operator | Affon / PyTorch ms | Program | Affon / PyTorch ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `cuda/bf16/argmax_axis/strided/strided_r64_c63` | reduction | 64 × 63 | strided | correct | [passed](operator_peer/cuda__bf16__argmax_axis__strided__strided_r64_c63.json) | 0.03732 / 0.01561 | [passed](program_peer/cuda__bf16__argmax_axis__strided__strided_r64_c63.json) | 0.04209 / 0.01852 |
| `cuda/bf16/argmin_axis/strided/strided_r64_c63` | reduction | 64 × 63 | strided | correct | [passed](operator_peer/cuda__bf16__argmin_axis__strided__strided_r64_c63.json) | 0.03723 / 0.0163 | [passed](program_peer/cuda__bf16__argmin_axis__strided__strided_r64_c63.json) | 0.04196 / 0.01854 |
| `cuda/bf16/max_axis/strided/inner_strided_r64_c63` | reduction | 64 × 63 | strided | correct | [passed](operator_peer/cuda__bf16__max_axis__strided__inner_strided_r64_c63.json) | 0.03732 / 0.01659 | [passed](program_peer/cuda__bf16__max_axis__strided__inner_strided_r64_c63.json) | 0.03902 / 0.01967 |
| `cuda/bf16/mean_axis/strided/inner_strided_r64_c63` | reduction | 64 × 63 | strided | correct | [passed](operator_peer/cuda__bf16__mean_axis__strided__inner_strided_r64_c63.json) | 0.03769 / 0.01595 | [passed](program_peer/cuda__bf16__mean_axis__strided__inner_strided_r64_c63.json) | 0.03928 / 0.01862 |
| `cuda/bf16/min_axis/strided/inner_strided_r64_c63` | reduction | 64 × 63 | strided | correct | [passed](operator_peer/cuda__bf16__min_axis__strided__inner_strided_r64_c63.json) | 0.03716 / 0.02213 | [passed](program_peer/cuda__bf16__min_axis__strided__inner_strided_r64_c63.json) | 0.03888 / 0.01954 |
| `cuda/bf16/sum_axis/strided/inner_strided_r64_c63` | reduction | 64 × 63 | strided | correct | [passed](operator_peer/cuda__bf16__sum_axis__strided__inner_strided_r64_c63.json) | 0.03724 / 0.01569 | [passed](program_peer/cuda__bf16__sum_axis__strided__inner_strided_r64_c63.json) | 0.03903 / 0.01854 |
| `cuda/f32/argmax_axis/strided/strided_r129_c63` | reduction | 129 × 63 | strided | correct | [passed](operator_peer/cuda__f32__argmax_axis__strided__strided_r129_c63.json) | 0.02138 / 0.01567 | [passed](program_peer/cuda__f32__argmax_axis__strided__strided_r129_c63.json) | 0.0268 / 0.01873 |
| `cuda/f32/argmin_axis/strided/strided_r64_c63` | reduction | 64 × 63 | strided | correct | [passed](operator_peer/cuda__f32__argmin_axis__strided__strided_r64_c63.json) | 0.02056 / 0.01535 | [passed](program_peer/cuda__f32__argmin_axis__strided__strided_r64_c63.json) | 0.02526 / 0.01822 |
| `cuda/f32/max_axis/strided/inner_strided_r64_c63` | reduction | 64 × 63 | strided | correct | [passed](operator_peer/cuda__f32__max_axis__strided__inner_strided_r64_c63.json) | 0.02903 / 0.01694 | [passed](program_peer/cuda__f32__max_axis__strided__inner_strided_r64_c63.json) | 0.03371 / 0.0221 |
| `cuda/f32/mean_axis/strided/inner_strided_r64_c63` | reduction | 64 × 63 | strided | correct | [passed](operator_peer/cuda__f32__mean_axis__strided__inner_strided_r64_c63.json) | 0.02903 / 0.016 | [passed](program_peer/cuda__f32__mean_axis__strided__inner_strided_r64_c63.json) | 0.03369 / 0.01892 |
| `cuda/f32/min_axis/strided/inner_strided_r64_c63` | reduction | 64 × 63 | strided | correct | [passed](operator_peer/cuda__f32__min_axis__strided__inner_strided_r64_c63.json) | 0.02838 / 0.01676 | [passed](program_peer/cuda__f32__min_axis__strided__inner_strided_r64_c63.json) | 0.03316 / 0.01994 |
| `cuda/f32/sum_axis/strided/inner_offset_r64_c63` | reduction | 64 × 63 | offset | correct | [passed](operator_peer/cuda__f32__sum_axis__strided__inner_offset_r64_c63.json) | 0.02441 / 0.0167 | [passed](program_peer/cuda__f32__sum_axis__strided__inner_offset_r64_c63.json) | 0.02931 / 0.02057 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
