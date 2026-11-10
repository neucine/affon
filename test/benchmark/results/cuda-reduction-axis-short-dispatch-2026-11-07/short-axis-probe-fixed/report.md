# Parity batch report

Target: `NVIDIA GeForce RTX 3090/7.0.0-38-generic/x86_64`  
Catalog SHA-256: `f7cd47e3a57cf3027930e02091ae373545c71f926847be89d74bdc8858a69bdd`  
Binary SHA-256: `4fafaf5b6ff353e6f47b3cff8f7c0e1899bfff0eb16a76fe6f16097df00dce27`  
Timing: 10 iterations × 5 samples × 3 rounds

## Checks

- Cases run: 24
- Numerical check passed: 24
- Operator check and timing passed: 24
- Program check and timing passed: 24
- Numerical failures: 0
- Operator failures: 0
- Program failures: 0
- Operator checks pending: 0
- Program checks pending: 0

## Cases

| Case | Family | Shape | Layout | Numerical | Operator | Affon / PyTorch ms | Program | Affon / PyTorch ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `cuda/f32/argmax_axis/outer_axis/offset_r3_c5` | reduction | 3 × 5 | offset | correct | [passed](operator_peer/cuda__f32__argmax_axis__outer_axis__offset_r3_c5.json) | 0.016 / 0.01597 | [passed](program_peer/cuda__f32__argmax_axis__outer_axis__offset_r3_c5.json) | 0.02082 / 0.01891 |
| `cuda/f32/argmax_axis/outer_axis/offset_r7_c63` | reduction | 7 × 63 | offset | correct | [passed](operator_peer/cuda__f32__argmax_axis__outer_axis__offset_r7_c63.json) | 0.01771 / 0.01665 | [passed](program_peer/cuda__f32__argmax_axis__outer_axis__offset_r7_c63.json) | 0.02254 / 0.02023 |
| `cuda/f32/argmax_axis/strided/strided_r129_c63` | reduction | 129 × 63 | strided | correct | [passed](operator_peer/cuda__f32__argmax_axis__strided__strided_r129_c63.json) | 0.02314 / 0.01612 | [passed](program_peer/cuda__f32__argmax_axis__strided__strided_r129_c63.json) | 0.02833 / 0.01938 |
| `cuda/f32/argmin_axis/outer_axis/offset_r3_c5` | reduction | 3 × 5 | offset | correct | [passed](operator_peer/cuda__f32__argmin_axis__outer_axis__offset_r3_c5.json) | 0.01602 / 0.0159 | [passed](program_peer/cuda__f32__argmin_axis__outer_axis__offset_r3_c5.json) | 0.0209 / 0.01894 |
| `cuda/f32/argmin_axis/outer_axis/offset_r7_c63` | reduction | 7 × 63 | offset | correct | [passed](operator_peer/cuda__f32__argmin_axis__outer_axis__offset_r7_c63.json) | 0.01769 / 0.01648 | [passed](program_peer/cuda__f32__argmin_axis__outer_axis__offset_r7_c63.json) | 0.02259 / 0.01967 |
| `cuda/f32/argmin_axis/strided/strided_r129_c63` | reduction | 129 × 63 | strided | correct | [passed](operator_peer/cuda__f32__argmin_axis__strided__strided_r129_c63.json) | 0.023 / 0.01603 | [passed](program_peer/cuda__f32__argmin_axis__strided__strided_r129_c63.json) | 0.0283 / 0.01929 |
| `cuda/f32/max_axis/outer_axis/offset_r3_c5` | reduction | 3 × 5 | offset | correct | [passed](operator_peer/cuda__f32__max_axis__outer_axis__offset_r3_c5.json) | 0.01635 / 0.01709 | [passed](program_peer/cuda__f32__max_axis__outer_axis__offset_r3_c5.json) | 0.02118 / 0.02024 |
| `cuda/f32/max_axis/outer_axis/offset_r3_c129` | reduction | 3 × 129 | offset | correct | [passed](operator_peer/cuda__f32__max_axis__outer_axis__offset_r3_c129.json) | 0.01896 / 0.0182 | [passed](program_peer/cuda__f32__max_axis__outer_axis__offset_r3_c129.json) | 0.02413 / 0.02151 |
| `cuda/f32/max_axis/strided/inner_strided_r129_c63` | reduction | 129 × 63 | strided | correct | [passed](operator_peer/cuda__f32__max_axis__strided__inner_strided_r129_c63.json) | 0.03125 / 0.01733 | [passed](program_peer/cuda__f32__max_axis__strided__inner_strided_r129_c63.json) | 0.03671 / 0.02047 |
| `cuda/f32/mean_axis/outer_axis/offset_r3_c5` | reduction | 3 × 5 | offset | correct | [passed](operator_peer/cuda__f32__mean_axis__outer_axis__offset_r3_c5.json) | 0.01625 / 0.01622 | [passed](program_peer/cuda__f32__mean_axis__outer_axis__offset_r3_c5.json) | 0.0211 / 0.01977 |
| `cuda/f32/mean_axis/outer_axis/offset_r3_c129` | reduction | 3 × 129 | offset | correct | [passed](operator_peer/cuda__f32__mean_axis__outer_axis__offset_r3_c129.json) | 0.01907 / 0.01745 | [passed](program_peer/cuda__f32__mean_axis__outer_axis__offset_r3_c129.json) | 0.02425 / 0.02036 |
| `cuda/f32/mean_axis/strided/inner_strided_r129_c63` | reduction | 129 × 63 | strided | correct | [passed](operator_peer/cuda__f32__mean_axis__strided__inner_strided_r129_c63.json) | 0.03146 / 0.01648 | [passed](program_peer/cuda__f32__mean_axis__strided__inner_strided_r129_c63.json) | 0.03655 / 0.01939 |
| `cuda/f32/min_axis/outer_axis/offset_r3_c5` | reduction | 3 × 5 | offset | correct | [passed](operator_peer/cuda__f32__min_axis__outer_axis__offset_r3_c5.json) | 0.01622 / 0.01723 | [passed](program_peer/cuda__f32__min_axis__outer_axis__offset_r3_c5.json) | 0.02119 / 0.02008 |
| `cuda/f32/min_axis/outer_axis/offset_r3_c129` | reduction | 3 × 129 | offset | correct | [passed](operator_peer/cuda__f32__min_axis__outer_axis__offset_r3_c129.json) | 0.01892 / 0.01822 | [passed](program_peer/cuda__f32__min_axis__outer_axis__offset_r3_c129.json) | 0.02402 / 0.02141 |
| `cuda/f32/min_axis/strided/inner_strided_r129_c63` | reduction | 129 × 63 | strided | correct | [passed](operator_peer/cuda__f32__min_axis__strided__inner_strided_r129_c63.json) | 0.03072 / 0.01755 | [passed](program_peer/cuda__f32__min_axis__strided__inner_strided_r129_c63.json) | 0.03602 / 0.02054 |
| `cuda/f32/std_axis/outer_axis/offset_r3_c5` | reduction | 3 × 5 | offset | correct | [passed](operator_peer/cuda__f32__std_axis__outer_axis__offset_r3_c5.json) | 0.01639 / 0.0167 | [passed](program_peer/cuda__f32__std_axis__outer_axis__offset_r3_c5.json) | 0.02125 / 0.01963 |
| `cuda/f32/std_axis/outer_axis/offset_r7_c63` | reduction | 7 × 63 | offset | correct | [passed](operator_peer/cuda__f32__std_axis__outer_axis__offset_r7_c63.json) | 0.01799 / 0.01732 | [passed](program_peer/cuda__f32__std_axis__outer_axis__offset_r7_c63.json) | 0.02292 / 0.02043 |
| `cuda/f32/std_axis/strided/strided_r129_c63` | reduction | 129 × 63 | strided | correct | [passed](operator_peer/cuda__f32__std_axis__strided__strided_r129_c63.json) | 0.02529 / 0.01702 | [passed](program_peer/cuda__f32__std_axis__strided__strided_r129_c63.json) | 0.03061 / 0.0201 |
| `cuda/f32/sum_axis/outer_axis/offset_r3_c5` | reduction | 3 × 5 | offset | correct | [passed](operator_peer/cuda__f32__sum_axis__outer_axis__offset_r3_c5.json) | 0.01637 / 0.0161 | [passed](program_peer/cuda__f32__sum_axis__outer_axis__offset_r3_c5.json) | 0.02095 / 0.0196 |
| `cuda/f32/sum_axis/outer_axis/offset_r3_c129` | reduction | 3 × 129 | offset | correct | [passed](operator_peer/cuda__f32__sum_axis__outer_axis__offset_r3_c129.json) | 0.0188 / 0.01746 | [passed](program_peer/cuda__f32__sum_axis__outer_axis__offset_r3_c129.json) | 0.02416 / 0.02102 |
| `cuda/f32/sum_axis/strided/inner_strided_r129_c63` | reduction | 129 × 63 | strided | correct | [passed](operator_peer/cuda__f32__sum_axis__strided__inner_strided_r129_c63.json) | 0.03124 / 0.01637 | [passed](program_peer/cuda__f32__sum_axis__strided__inner_strided_r129_c63.json) | 0.03665 / 0.01988 |
| `cuda/f32/variance_axis/outer_axis/offset_r3_c5` | reduction | 3 × 5 | offset | correct | [passed](operator_peer/cuda__f32__variance_axis__outer_axis__offset_r3_c5.json) | 0.0164 / 0.01678 | [passed](program_peer/cuda__f32__variance_axis__outer_axis__offset_r3_c5.json) | 0.0212 / 0.02037 |
| `cuda/f32/variance_axis/outer_axis/offset_r7_c63` | reduction | 7 × 63 | offset | correct | [passed](operator_peer/cuda__f32__variance_axis__outer_axis__offset_r7_c63.json) | 0.01809 / 0.01735 | [passed](program_peer/cuda__f32__variance_axis__outer_axis__offset_r7_c63.json) | 0.02282 / 0.02111 |
| `cuda/f32/variance_axis/strided/strided_r129_c63` | reduction | 129 × 63 | strided | correct | [passed](operator_peer/cuda__f32__variance_axis__strided__strided_r129_c63.json) | 0.02515 / 0.01684 | [passed](program_peer/cuda__f32__variance_axis__strided__strided_r129_c63.json) | 0.03077 / 0.02044 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
