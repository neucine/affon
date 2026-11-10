# Parity batch report

Target: `NVIDIA GeForce RTX 3090/7.0.0-38-generic/x86_64`  
Catalog SHA-256: `f7cd47e3a57cf3027930e02091ae373545c71f926847be89d74bdc8858a69bdd`  
Binary SHA-256: `6019dcacc6ce96d967924bbee7fdcea13cbce8721db3e3f1585d4b7b96c7c48c`  
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
| `cuda/f32/argmax_axis/outer_axis/offset_r3_c5` | reduction | 3 × 5 | offset | correct | [passed](operator_peer/cuda__f32__argmax_axis__outer_axis__offset_r3_c5.json) | 0.02317 / 0.01607 | [passed](program_peer/cuda__f32__argmax_axis__outer_axis__offset_r3_c5.json) | 0.02805 / 0.01909 |
| `cuda/f32/argmax_axis/outer_axis/offset_r7_c63` | reduction | 7 × 63 | offset | correct | [passed](operator_peer/cuda__f32__argmax_axis__outer_axis__offset_r7_c63.json) | 0.1393 / 0.01681 | [passed](program_peer/cuda__f32__argmax_axis__outer_axis__offset_r7_c63.json) | 0.1439 / 0.01977 |
| `cuda/f32/argmax_axis/strided/strided_r129_c63` | reduction | 129 × 63 | strided | correct | [passed](operator_peer/cuda__f32__argmax_axis__strided__strided_r129_c63.json) | 0.644 / 0.01578 | [passed](program_peer/cuda__f32__argmax_axis__strided__strided_r129_c63.json) | 0.6492 / 0.01951 |
| `cuda/f32/argmin_axis/outer_axis/offset_r3_c5` | reduction | 3 × 5 | offset | correct | [passed](operator_peer/cuda__f32__argmin_axis__outer_axis__offset_r3_c5.json) | 0.02294 / 0.01586 | [passed](program_peer/cuda__f32__argmin_axis__outer_axis__offset_r3_c5.json) | 0.02805 / 0.02021 |
| `cuda/f32/argmin_axis/outer_axis/offset_r7_c63` | reduction | 7 × 63 | offset | correct | [passed](operator_peer/cuda__f32__argmin_axis__outer_axis__offset_r7_c63.json) | 0.1382 / 0.01654 | [passed](program_peer/cuda__f32__argmin_axis__outer_axis__offset_r7_c63.json) | 0.1437 / 0.01985 |
| `cuda/f32/argmin_axis/strided/strided_r129_c63` | reduction | 129 × 63 | strided | correct | [passed](operator_peer/cuda__f32__argmin_axis__strided__strided_r129_c63.json) | 0.6443 / 0.01644 | [passed](program_peer/cuda__f32__argmin_axis__strided__strided_r129_c63.json) | 0.6492 / 0.01954 |
| `cuda/f32/max_axis/outer_axis/offset_r3_c5` | reduction | 3 × 5 | offset | correct | [passed](operator_peer/cuda__f32__max_axis__outer_axis__offset_r3_c5.json) | 0.02349 / 0.01728 | [passed](program_peer/cuda__f32__max_axis__outer_axis__offset_r3_c5.json) | 0.02826 / 0.02048 |
| `cuda/f32/max_axis/outer_axis/offset_r3_c129` | reduction | 3 × 129 | offset | correct | [passed](operator_peer/cuda__f32__max_axis__outer_axis__offset_r3_c129.json) | 0.2657 / 0.01851 | [passed](program_peer/cuda__f32__max_axis__outer_axis__offset_r3_c129.json) | 0.2708 / 0.02218 |
| `cuda/f32/max_axis/strided/inner_strided_r129_c63` | reduction | 129 × 63 | strided | correct | [passed](operator_peer/cuda__f32__max_axis__strided__inner_strided_r129_c63.json) | 0.8599 / 0.01731 | [passed](program_peer/cuda__f32__max_axis__strided__inner_strided_r129_c63.json) | 0.8648 / 0.02032 |
| `cuda/f32/mean_axis/outer_axis/offset_r3_c5` | reduction | 3 × 5 | offset | correct | [passed](operator_peer/cuda__f32__mean_axis__outer_axis__offset_r3_c5.json) | 0.02352 / 0.0167 | [passed](program_peer/cuda__f32__mean_axis__outer_axis__offset_r3_c5.json) | 0.02836 / 0.01978 |
| `cuda/f32/mean_axis/outer_axis/offset_r3_c129` | reduction | 3 × 129 | offset | correct | [passed](operator_peer/cuda__f32__mean_axis__outer_axis__offset_r3_c129.json) | 0.2681 / 0.01764 | [passed](program_peer/cuda__f32__mean_axis__outer_axis__offset_r3_c129.json) | 0.2735 / 0.02055 |
| `cuda/f32/mean_axis/strided/inner_strided_r129_c63` | reduction | 129 × 63 | strided | correct | [passed](operator_peer/cuda__f32__mean_axis__strided__inner_strided_r129_c63.json) | 0.425 / 0.01656 | [passed](program_peer/cuda__f32__mean_axis__strided__inner_strided_r129_c63.json) | 0.4313 / 0.01949 |
| `cuda/f32/min_axis/outer_axis/offset_r3_c5` | reduction | 3 × 5 | offset | correct | [passed](operator_peer/cuda__f32__min_axis__outer_axis__offset_r3_c5.json) | 0.02337 / 0.01756 | [passed](program_peer/cuda__f32__min_axis__outer_axis__offset_r3_c5.json) | 0.02808 / 0.0202 |
| `cuda/f32/min_axis/outer_axis/offset_r3_c129` | reduction | 3 × 129 | offset | correct | [passed](operator_peer/cuda__f32__min_axis__outer_axis__offset_r3_c129.json) | 0.2654 / 0.01858 | [passed](program_peer/cuda__f32__min_axis__outer_axis__offset_r3_c129.json) | 0.2707 / 0.02138 |
| `cuda/f32/min_axis/strided/inner_strided_r129_c63` | reduction | 129 × 63 | strided | correct | [passed](operator_peer/cuda__f32__min_axis__strided__inner_strided_r129_c63.json) | 0.8596 / 0.01696 | [passed](program_peer/cuda__f32__min_axis__strided__inner_strided_r129_c63.json) | 0.865 / 0.02074 |
| `cuda/f32/std_axis/outer_axis/offset_r3_c5` | reduction | 3 × 5 | offset | correct | [passed](operator_peer/cuda__f32__std_axis__outer_axis__offset_r3_c5.json) | 0.02521 / 0.01698 | [passed](program_peer/cuda__f32__std_axis__outer_axis__offset_r3_c5.json) | 0.02981 / 0.02033 |
| `cuda/f32/std_axis/outer_axis/offset_r7_c63` | reduction | 7 × 63 | offset | correct | [passed](operator_peer/cuda__f32__std_axis__outer_axis__offset_r7_c63.json) | 0.1723 / 0.01752 | [passed](program_peer/cuda__f32__std_axis__outer_axis__offset_r7_c63.json) | 0.1748 / 0.02122 |
| `cuda/f32/std_axis/strided/strided_r129_c63` | reduction | 129 × 63 | strided | correct | [passed](operator_peer/cuda__f32__std_axis__strided__strided_r129_c63.json) | 0.7817 / 0.01703 | [passed](program_peer/cuda__f32__std_axis__strided__strided_r129_c63.json) | 0.7864 / 0.0205 |
| `cuda/f32/sum_axis/outer_axis/offset_r3_c5` | reduction | 3 × 5 | offset | correct | [passed](operator_peer/cuda__f32__sum_axis__outer_axis__offset_r3_c5.json) | 0.0236 / 0.0162 | [passed](program_peer/cuda__f32__sum_axis__outer_axis__offset_r3_c5.json) | 0.02832 / 0.01918 |
| `cuda/f32/sum_axis/outer_axis/offset_r3_c129` | reduction | 3 × 129 | offset | correct | [passed](operator_peer/cuda__f32__sum_axis__outer_axis__offset_r3_c129.json) | 0.2632 / 0.01745 | [passed](program_peer/cuda__f32__sum_axis__outer_axis__offset_r3_c129.json) | 0.2689 / 0.02046 |
| `cuda/f32/sum_axis/strided/inner_strided_r129_c63` | reduction | 129 × 63 | strided | correct | [passed](operator_peer/cuda__f32__sum_axis__strided__inner_strided_r129_c63.json) | 0.6016 / 0.01613 | [passed](program_peer/cuda__f32__sum_axis__strided__inner_strided_r129_c63.json) | 0.6072 / 0.0192 |
| `cuda/f32/variance_axis/outer_axis/offset_r3_c5` | reduction | 3 × 5 | offset | correct | [passed](operator_peer/cuda__f32__variance_axis__outer_axis__offset_r3_c5.json) | 0.02519 / 0.01696 | [passed](program_peer/cuda__f32__variance_axis__outer_axis__offset_r3_c5.json) | 0.02962 / 0.01997 |
| `cuda/f32/variance_axis/outer_axis/offset_r7_c63` | reduction | 7 × 63 | offset | correct | [passed](operator_peer/cuda__f32__variance_axis__outer_axis__offset_r7_c63.json) | 0.1712 / 0.01747 | [passed](program_peer/cuda__f32__variance_axis__outer_axis__offset_r7_c63.json) | 0.1752 / 0.02052 |
| `cuda/f32/variance_axis/strided/strided_r129_c63` | reduction | 129 × 63 | strided | correct | [passed](operator_peer/cuda__f32__variance_axis__strided__strided_r129_c63.json) | 0.7794 / 0.01756 | [passed](program_peer/cuda__f32__variance_axis__strided__strided_r129_c63.json) | 0.7845 / 0.02045 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
