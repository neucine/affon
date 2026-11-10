# Parity batch report

Target: `NVIDIA GeForce RTX 3090/7.0.0-42-generic/x86_64`  
Catalog SHA-256: `1cbbac026e1c863adb033b580d58cb26852e7ffa06ecbec59128ebd58d00bab5`  
Binary SHA-256: `bb45d09857a77f7f13db51106d52f12e8cdecf23e655eb553f83e289b09fd651`  
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
| `cuda/bf16/argmax_axis/strided/strided_r64_c63` | reduction | 64 × 63 | strided | correct | [passed](operator_peer/cuda__bf16__argmax_axis__strided__strided_r64_c63.json) | 0.01957 / 0.01522 | [passed](program_peer/cuda__bf16__argmax_axis__strided__strided_r64_c63.json) | 0.02457 / 0.01919 |
| `cuda/bf16/argmin_axis/strided/strided_r64_c63` | reduction | 64 × 63 | strided | correct | [passed](operator_peer/cuda__bf16__argmin_axis__strided__strided_r64_c63.json) | 0.01965 / 0.01536 | [passed](program_peer/cuda__bf16__argmin_axis__strided__strided_r64_c63.json) | 0.02443 / 0.01934 |
| `cuda/bf16/max_axis/strided/inner_strided_r64_c63` | reduction | 64 × 63 | strided | correct | [passed](operator_peer/cuda__bf16__max_axis__strided__inner_strided_r64_c63.json) | 0.01978 / 0.01647 | [passed](program_peer/cuda__bf16__max_axis__strided__inner_strided_r64_c63.json) | 0.02191 / 0.02287 |
| `cuda/bf16/mean_axis/strided/inner_strided_r64_c63` | reduction | 64 × 63 | strided | correct | [passed](operator_peer/cuda__bf16__mean_axis__strided__inner_strided_r64_c63.json) | 0.02022 / 0.01592 | [passed](program_peer/cuda__bf16__mean_axis__strided__inner_strided_r64_c63.json) | 0.02238 / 0.0192 |
| `cuda/bf16/min_axis/strided/inner_strided_r64_c63` | reduction | 64 × 63 | strided | correct | [passed](operator_peer/cuda__bf16__min_axis__strided__inner_strided_r64_c63.json) | 0.01973 / 0.01714 | [passed](program_peer/cuda__bf16__min_axis__strided__inner_strided_r64_c63.json) | 0.02196 / 0.02028 |
| `cuda/bf16/sum_axis/strided/inner_strided_r64_c63` | reduction | 64 × 63 | strided | correct | [passed](operator_peer/cuda__bf16__sum_axis__strided__inner_strided_r64_c63.json) | 0.01973 / 0.01565 | [passed](program_peer/cuda__bf16__sum_axis__strided__inner_strided_r64_c63.json) | 0.0221 / 0.01876 |
| `cuda/f32/argmax_axis/strided/strided_r129_c63` | reduction | 129 × 63 | strided | correct | [passed](operator_peer/cuda__f32__argmax_axis__strided__strided_r129_c63.json) | 0.01858 / 0.01636 | [passed](program_peer/cuda__f32__argmax_axis__strided__strided_r129_c63.json) | 0.02394 / 0.02015 |
| `cuda/f32/argmin_axis/strided/strided_r64_c63` | reduction | 64 × 63 | strided | correct | [passed](operator_peer/cuda__f32__argmin_axis__strided__strided_r64_c63.json) | 0.01789 / 0.0169 | [passed](program_peer/cuda__f32__argmin_axis__strided__strided_r64_c63.json) | 0.0227 / 0.01858 |
| `cuda/f32/max_axis/strided/inner_strided_r64_c63` | reduction | 64 × 63 | strided | correct | [passed](operator_peer/cuda__f32__max_axis__strided__inner_strided_r64_c63.json) | 0.01788 / 0.01725 | [passed](program_peer/cuda__f32__max_axis__strided__inner_strided_r64_c63.json) | 0.02277 / 0.02015 |
| `cuda/f32/mean_axis/strided/inner_strided_r64_c63` | reduction | 64 × 63 | strided | correct | [passed](operator_peer/cuda__f32__mean_axis__strided__inner_strided_r64_c63.json) | 0.01743 / 0.01569 | [passed](program_peer/cuda__f32__mean_axis__strided__inner_strided_r64_c63.json) | 0.02214 / 0.01873 |
| `cuda/f32/min_axis/strided/inner_strided_r64_c63` | reduction | 64 × 63 | strided | correct | [passed](operator_peer/cuda__f32__min_axis__strided__inner_strided_r64_c63.json) | 0.01786 / 0.01708 | [passed](program_peer/cuda__f32__min_axis__strided__inner_strided_r64_c63.json) | 0.02276 / 0.02069 |
| `cuda/f32/sum_axis/strided/inner_offset_r64_c63` | reduction | 64 × 63 | offset | correct | [passed](operator_peer/cuda__f32__sum_axis__strided__inner_offset_r64_c63.json) | 0.01733 / 0.01736 | [passed](program_peer/cuda__f32__sum_axis__strided__inner_offset_r64_c63.json) | 0.02208 / 0.01975 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
