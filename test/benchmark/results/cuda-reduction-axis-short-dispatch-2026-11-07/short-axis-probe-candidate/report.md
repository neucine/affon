# Parity batch report

Target: `NVIDIA GeForce RTX 3090/7.0.0-38-generic/x86_64`  
Catalog SHA-256: `f7cd47e3a57cf3027930e02091ae373545c71f926847be89d74bdc8858a69bdd`  
Binary SHA-256: `c63919532dcc180c240a339dc2e432b2186248a6adcbc8b16a770275bd81f8ab`  
Timing: 10 iterations × 5 samples × 3 rounds

## Checks

- Cases run: 24
- Numerical check passed: 21
- Operator check and timing passed: 21
- Program check and timing passed: 21
- Numerical failures: 3
- Operator failures: 0
- Program failures: 0
- Operator checks pending: 0
- Program checks pending: 0

## Cases

| Case | Family | Shape | Layout | Numerical | Operator | Affon / PyTorch ms | Program | Affon / PyTorch ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `cuda/f32/argmax_axis/outer_axis/offset_r3_c5` | reduction | 3 × 5 | offset | correct | [passed](operator_peer/cuda__f32__argmax_axis__outer_axis__offset_r3_c5.json) | 0.01607 / 0.01633 | [passed](program_peer/cuda__f32__argmax_axis__outer_axis__offset_r3_c5.json) | 0.0209 / 0.01919 |
| `cuda/f32/argmax_axis/outer_axis/offset_r7_c63` | reduction | 7 × 63 | offset | correct | [passed](operator_peer/cuda__f32__argmax_axis__outer_axis__offset_r7_c63.json) | 0.01747 / 0.0168 | [passed](program_peer/cuda__f32__argmax_axis__outer_axis__offset_r7_c63.json) | 0.02235 / 0.01994 |
| `cuda/f32/argmax_axis/strided/strided_r129_c63` | reduction | 129 × 63 | strided | correct | [passed](operator_peer/cuda__f32__argmax_axis__strided__strided_r129_c63.json) | 0.02297 / 0.01619 | [passed](program_peer/cuda__f32__argmax_axis__strided__strided_r129_c63.json) | 0.02836 / 0.01975 |
| `cuda/f32/argmin_axis/outer_axis/offset_r3_c5` | reduction | 3 × 5 | offset | correct | [passed](operator_peer/cuda__f32__argmin_axis__outer_axis__offset_r3_c5.json) | 0.01632 / 0.0161 | [passed](program_peer/cuda__f32__argmin_axis__outer_axis__offset_r3_c5.json) | 0.02075 / 0.01952 |
| `cuda/f32/argmin_axis/outer_axis/offset_r7_c63` | reduction | 7 × 63 | offset | correct | [passed](operator_peer/cuda__f32__argmin_axis__outer_axis__offset_r7_c63.json) | 0.01768 / 0.01682 | [passed](program_peer/cuda__f32__argmin_axis__outer_axis__offset_r7_c63.json) | 0.02277 / 0.01978 |
| `cuda/f32/argmin_axis/strided/strided_r129_c63` | reduction | 129 × 63 | strided | correct | [passed](operator_peer/cuda__f32__argmin_axis__strided__strided_r129_c63.json) | 0.02294 / 0.01644 | [passed](program_peer/cuda__f32__argmin_axis__strided__strided_r129_c63.json) | 0.02855 / 0.01944 |
| `cuda/f32/max_axis/outer_axis/offset_r3_c5` | reduction | 3 × 5 | offset | correct | [passed](operator_peer/cuda__f32__max_axis__outer_axis__offset_r3_c5.json) | 0.01612 / 0.01774 | [passed](program_peer/cuda__f32__max_axis__outer_axis__offset_r3_c5.json) | 0.02089 / 0.02029 |
| `cuda/f32/max_axis/outer_axis/offset_r3_c129` | reduction | 3 × 129 | offset | correct | [passed](operator_peer/cuda__f32__max_axis__outer_axis__offset_r3_c129.json) | 0.01865 / 0.01862 | [passed](program_peer/cuda__f32__max_axis__outer_axis__offset_r3_c129.json) | 0.02403 / 0.02162 |
| `cuda/f32/max_axis/strided/inner_strided_r129_c63` | reduction | 129 × 63 | strided | correct | [passed](operator_peer/cuda__f32__max_axis__strided__inner_strided_r129_c63.json) | 0.03102 / 0.01733 | [passed](program_peer/cuda__f32__max_axis__strided__inner_strided_r129_c63.json) | 0.03654 / 0.02073 |
| `cuda/f32/mean_axis/outer_axis/offset_r3_c5` | reduction | 3 × 5 | offset | correct | [passed](operator_peer/cuda__f32__mean_axis__outer_axis__offset_r3_c5.json) | 0.01637 / 0.01645 | [passed](program_peer/cuda__f32__mean_axis__outer_axis__offset_r3_c5.json) | 0.02131 / 0.01941 |
| `cuda/f32/mean_axis/outer_axis/offset_r3_c129` | reduction | 3 × 129 | offset | correct | [passed](operator_peer/cuda__f32__mean_axis__outer_axis__offset_r3_c129.json) | 0.01883 / 0.01784 | [passed](program_peer/cuda__f32__mean_axis__outer_axis__offset_r3_c129.json) | 0.02412 / 0.02081 |
| `cuda/f32/mean_axis/strided/inner_strided_r129_c63` | reduction | 129 × 63 | strided | correct | [passed](operator_peer/cuda__f32__mean_axis__strided__inner_strided_r129_c63.json) | 0.03136 / 0.01697 | [passed](program_peer/cuda__f32__mean_axis__strided__inner_strided_r129_c63.json) | 0.03667 / 0.01968 |
| `cuda/f32/min_axis/outer_axis/offset_r3_c5` | reduction | 3 × 5 | offset | incorrect | probe_not_correct | — | probe_not_correct | — |
| `cuda/f32/min_axis/outer_axis/offset_r3_c129` | reduction | 3 × 129 | offset | incorrect | probe_not_correct | — | probe_not_correct | — |
| `cuda/f32/min_axis/strided/inner_strided_r129_c63` | reduction | 129 × 63 | strided | incorrect | probe_not_correct | — | probe_not_correct | — |
| `cuda/f32/std_axis/outer_axis/offset_r3_c5` | reduction | 3 × 5 | offset | correct | [passed](operator_peer/cuda__f32__std_axis__outer_axis__offset_r3_c5.json) | 0.01647 / 0.01723 | [passed](program_peer/cuda__f32__std_axis__outer_axis__offset_r3_c5.json) | 0.02127 / 0.02075 |
| `cuda/f32/std_axis/outer_axis/offset_r7_c63` | reduction | 7 × 63 | offset | correct | [passed](operator_peer/cuda__f32__std_axis__outer_axis__offset_r7_c63.json) | 0.01834 / 0.01823 | [passed](program_peer/cuda__f32__std_axis__outer_axis__offset_r7_c63.json) | 0.02282 / 0.02149 |
| `cuda/f32/std_axis/strided/strided_r129_c63` | reduction | 129 × 63 | strided | correct | [passed](operator_peer/cuda__f32__std_axis__strided__strided_r129_c63.json) | 0.02535 / 0.01727 | [passed](program_peer/cuda__f32__std_axis__strided__strided_r129_c63.json) | 0.03073 / 0.0203 |
| `cuda/f32/sum_axis/outer_axis/offset_r3_c5` | reduction | 3 × 5 | offset | correct | [passed](operator_peer/cuda__f32__sum_axis__outer_axis__offset_r3_c5.json) | 0.01634 / 0.0167 | [passed](program_peer/cuda__f32__sum_axis__outer_axis__offset_r3_c5.json) | 0.02101 / 0.01945 |
| `cuda/f32/sum_axis/outer_axis/offset_r3_c129` | reduction | 3 × 129 | offset | correct | [passed](operator_peer/cuda__f32__sum_axis__outer_axis__offset_r3_c129.json) | 0.01893 / 0.01763 | [passed](program_peer/cuda__f32__sum_axis__outer_axis__offset_r3_c129.json) | 0.02423 / 0.02055 |
| `cuda/f32/sum_axis/strided/inner_strided_r129_c63` | reduction | 129 × 63 | strided | correct | [passed](operator_peer/cuda__f32__sum_axis__strided__inner_strided_r129_c63.json) | 0.03139 / 0.01646 | [passed](program_peer/cuda__f32__sum_axis__strided__inner_strided_r129_c63.json) | 0.03671 / 0.01953 |
| `cuda/f32/variance_axis/outer_axis/offset_r3_c5` | reduction | 3 × 5 | offset | correct | [passed](operator_peer/cuda__f32__variance_axis__outer_axis__offset_r3_c5.json) | 0.01652 / 0.01724 | [passed](program_peer/cuda__f32__variance_axis__outer_axis__offset_r3_c5.json) | 0.02133 / 0.02012 |
| `cuda/f32/variance_axis/outer_axis/offset_r7_c63` | reduction | 7 × 63 | offset | correct | [passed](operator_peer/cuda__f32__variance_axis__outer_axis__offset_r7_c63.json) | 0.01815 / 0.01822 | [passed](program_peer/cuda__f32__variance_axis__outer_axis__offset_r7_c63.json) | 0.02284 / 0.02093 |
| `cuda/f32/variance_axis/strided/strided_r129_c63` | reduction | 129 × 63 | strided | correct | [passed](operator_peer/cuda__f32__variance_axis__strided__strided_r129_c63.json) | 0.02515 / 0.01732 | [passed](program_peer/cuda__f32__variance_axis__strided__strided_r129_c63.json) | 0.0306 / 0.0205 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
