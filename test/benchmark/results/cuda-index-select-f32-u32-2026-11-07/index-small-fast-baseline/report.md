# Parity batch report

Target: `NVIDIA GeForce RTX 3090/7.0.0-38-generic/x86_64`  
Catalog SHA-256: `7ec7a5a4166ba0887c0927fd922c93490bc3f5499edeacfa5f1eee320b02fd3c`  
Binary SHA-256: `6019dcacc6ce96d967924bbee7fdcea13cbce8721db3e3f1585d4b7b96c7c48c`  
Timing: 10 iterations × 5 samples × 3 rounds

## Checks

- Cases run: 33
- Numerical check passed: 33
- Operator check and timing passed: 33
- Program check and timing passed: 33
- Numerical failures: 0
- Operator failures: 0
- Program failures: 0
- Operator checks pending: 0
- Program checks pending: 0

## Cases

| Case | Family | Shape | Layout | Numerical | Operator | Affon / PyTorch ms | Program | Affon / PyTorch ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `cuda/bf16/embedding/batched` | indexing | 16 × 32 | dense | correct | [passed](operator_peer/cuda__bf16__embedding__batched.json) | 0.03019 / 0.0235 | [passed](program_peer/cuda__bf16__embedding__batched.json) | 0.03594 / 0.01846 |
| `cuda/bf16/embedding/batched/batch3` | indexing | 64 × 32 | dense | correct | [passed](operator_peer/cuda__bf16__embedding__batched__batch3.json) | 0.03716 / 0.02589 | [passed](program_peer/cuda__bf16__embedding__batched__batch3.json) | 0.04302 / 0.02878 |
| `cuda/bf16/embedding/batched/batch8` | indexing | 64 × 32 | dense | correct | [passed](operator_peer/cuda__bf16__embedding__batched__batch8.json) | 0.08185 / 0.0342 | [passed](program_peer/cuda__bf16__embedding__batched__batch8.json) | 0.08518 / 0.03841 |
| `cuda/bf16/embedding/large/r64_w31_i31` | indexing | 64 × 31 | dense | correct | [passed](operator_peer/cuda__bf16__embedding__large__r64_w31_i31.json) | 0.04951 / 0.02801 | [passed](program_peer/cuda__bf16__embedding__large__r64_w31_i31.json) | 0.05488 / 0.02704 |
| `cuda/bf16/embedding/large/rows65` | indexing | 65 × 64 | dense | correct | [passed](operator_peer/cuda__bf16__embedding__large__rows65.json) | 0.04575 / 0.03078 | [passed](program_peer/cuda__bf16__embedding__large__rows65.json) | 0.05083 / 0.0335 |
| `cuda/bf16/embedding/large/r128_c127_i64` | indexing | 128 × 127 | dense | correct | [passed](operator_peer/cuda__bf16__embedding__large__r128_c127_i64.json) | 0.1752 / 0.08227 | [passed](program_peer/cuda__bf16__embedding__large__r128_c127_i64.json) | 0.1805 / 0.08301 |
| `cuda/bf16/embedding/small` | indexing | 8 × 16 | dense | correct | [passed](operator_peer/cuda__bf16__embedding__small.json) | 0.02575 / 0.01288 | [passed](program_peer/cuda__bf16__embedding__small.json) | 0.03117 / 0.01628 |
| `cuda/bf16/embedding/small/r32_w16_i16` | indexing | 32 × 16 | dense | correct | [passed](operator_peer/cuda__bf16__embedding__small__r32_w16_i16.json) | 0.03213 / 0.02355 | [passed](program_peer/cuda__bf16__embedding__small__r32_w16_i16.json) | 0.03714 / 0.02601 |
| `cuda/bf16/embedding/small/r32_w32_i16_strided` | indexing | 32 × 32 | strided | correct | [passed](operator_peer/cuda__bf16__embedding__small__r32_w32_i16_strided.json) | 0.03675 / 0.02697 | [passed](program_peer/cuda__bf16__embedding__small__r32_w32_i16_strided.json) | 0.04209 / 0.03092 |
| `cuda/bf16/embedding/strided` | indexing | 8 × 16 | strided | correct | [passed](operator_peer/cuda__bf16__embedding__strided.json) | 0.02592 / 0.01431 | [passed](program_peer/cuda__bf16__embedding__strided.json) | 0.03122 / 0.01746 |
| `cuda/bf16/embedding/strided/r64_c127_i16` | indexing | 64 × 127 | strided | correct | [passed](operator_peer/cuda__bf16__embedding__strided__r64_c127_i16.json) | 0.06268 / 0.04147 | [passed](program_peer/cuda__bf16__embedding__strided__r64_c127_i16.json) | 0.0679 / 0.04515 |
| `cuda/bf16/gather/batched` | indexing | 16 × 32 | dense | correct | [passed](operator_peer/cuda__bf16__gather__batched.json) | 0.03009 / 0.01449 | [passed](program_peer/cuda__bf16__gather__batched.json) | 0.03525 / 0.0169 |
| `cuda/bf16/gather/large/offset_axis0_r3_c3` | indexing | 3 × 3 | offset | correct | [passed](operator_peer/cuda__bf16__gather__large__offset_axis0_r3_c3.json) | 0.02402 / 0.01423 | [passed](program_peer/cuda__bf16__gather__large__offset_axis0_r3_c3.json) | 0.02916 / 0.0177 |
| `cuda/bf16/gather/large/offset_axis1_r8_c63` | indexing | 8 × 63 | offset | correct | [passed](operator_peer/cuda__bf16__gather__large__offset_axis1_r8_c63.json) | 0.03013 / 0.01688 | [passed](program_peer/cuda__bf16__gather__large__offset_axis1_r8_c63.json) | 0.03544 / 0.01967 |
| `cuda/bf16/gather/large/r128_c127_i64` | indexing | 128 × 127 | dense | correct | [passed](operator_peer/cuda__bf16__gather__large__r128_c127_i64.json) | 0.176 / 0.08529 | [passed](program_peer/cuda__bf16__gather__large__r128_c127_i64.json) | 0.1803 / 0.08129 |
| `cuda/bf16/gather/small` | indexing | 4 × 8 | dense | correct | [passed](operator_peer/cuda__bf16__gather__small.json) | 0.02505 / 0.01217 | [passed](program_peer/cuda__bf16__gather__small.json) | 0.03054 / 0.01507 |
| `cuda/bf16/gather/small/a1_r32_w31_o16` | indexing | 32 × 31 | dense | correct | [passed](operator_peer/cuda__bf16__gather__small__a1_r32_w31_o16.json) | 0.04073 / 0.01729 | [passed](program_peer/cuda__bf16__gather__small__a1_r32_w31_o16.json) | 0.04597 / 0.02003 |
| `cuda/bf16/gather/small/r32_w64_i32x31_a1_strided` | indexing | 32 × 64 | strided | correct | [passed](operator_peer/cuda__bf16__gather__small__r32_w64_i32x31_a1_strided.json) | 0.05003 / 0.0317 | [passed](program_peer/cuda__bf16__gather__small__r32_w64_i32x31_a1_strided.json) | 0.05603 / 0.02824 |
| `cuda/bf16/gather/strided/axis0_r3_c3` | indexing | 3 × 3 | strided | correct | [passed](operator_peer/cuda__bf16__gather__strided__axis0_r3_c3.json) | 0.02374 / 0.01326 | [passed](program_peer/cuda__bf16__gather__strided__axis0_r3_c3.json) | 0.0295 / 0.01624 |
| `cuda/bf16/gather/strided/axis0_r7_c64` | indexing | 7 × 64 | strided | correct | [passed](operator_peer/cuda__bf16__gather__strided__axis0_r7_c64.json) | 0.02916 / 0.01571 | [passed](program_peer/cuda__bf16__gather__strided__axis0_r7_c64.json) | 0.03435 / 0.01879 |
| `cuda/bf16/gather/strided/axis1_r65_c127` | indexing | 65 × 127 | strided | correct | [passed](operator_peer/cuda__bf16__gather__strided__axis1_r65_c127.json) | 0.1162 / 0.05631 | [passed](program_peer/cuda__bf16__gather__strided__axis1_r65_c127.json) | 0.121 / 0.05327 |
| `cuda/bf16/index_select/batched/dense_axis1_b2_r3_c5_n1` | indexing | 2 × 3 × 5 | dense | correct | [passed](operator_peer/cuda__bf16__index_select__batched__dense_axis1_b2_r3_c5_n1.json) | 0.02417 / 0.01209 | [passed](program_peer/cuda__bf16__index_select__batched__dense_axis1_b2_r3_c5_n1.json) | 0.02977 / 0.01505 |
| `cuda/bf16/index_select/batched/dense_axis2_b4_r3_c65_n32` | indexing | 4 × 3 × 65 | dense | correct | [passed](operator_peer/cuda__bf16__index_select__batched__dense_axis2_b4_r3_c65_n32.json) | 0.03513 / 0.0179 | [passed](program_peer/cuda__bf16__index_select__batched__dense_axis2_b4_r3_c65_n32.json) | 0.04053 / 0.02096 |
| `cuda/bf16/index_select/batched/dense_axis2_b4_r33_c129_n64` | indexing | 4 × 33 × 129 | dense | correct | [passed](operator_peer/cuda__bf16__index_select__batched__dense_axis2_b4_r33_c129_n64.json) | 0.2055 / 0.08522 | [passed](program_peer/cuda__bf16__index_select__batched__dense_axis2_b4_r33_c129_n64.json) | 0.2109 / 0.08886 |
| `cuda/bf16/index_select/large/offset_axis0_r2_c7_i2` | indexing | 2 × 7 | offset | correct | [passed](operator_peer/cuda__bf16__index_select__large__offset_axis0_r2_c7_i2.json) | 0.02429 / 0.01466 | [passed](program_peer/cuda__bf16__index_select__large__offset_axis0_r2_c7_i2.json) | 0.02939 / 0.01747 |
| `cuda/bf16/index_select/large/offset_axis0_r7_c129_i4` | indexing | 7 × 129 | offset | correct | [passed](operator_peer/cuda__bf16__index_select__large__offset_axis0_r7_c129_i4.json) | 0.03351 / 0.01869 | [passed](program_peer/cuda__bf16__index_select__large__offset_axis0_r7_c129_i4.json) | 0.03887 / 0.02237 |
| `cuda/bf16/index_select/large/r128_c127_i64` | indexing | 128 × 127 | dense | correct | [passed](operator_peer/cuda__bf16__index_select__large__r128_c127_i64.json) | 0.1756 / 0.0799 | [passed](program_peer/cuda__bf16__index_select__large__r128_c127_i64.json) | 0.1804 / 0.0835 |
| `cuda/bf16/index_select/small` | indexing | 8 × 16 | dense | correct | [passed](operator_peer/cuda__bf16__index_select__small.json) | 0.02553 / 0.01252 | [passed](program_peer/cuda__bf16__index_select__small.json) | 0.03107 / 0.01561 |
| `cuda/bf16/index_select/small/a1_r32_w31_i15` | indexing | 32 × 31 | dense | correct | [passed](operator_peer/cuda__bf16__index_select__small__a1_r32_w31_i15.json) | 0.03986 / 0.02556 | [passed](program_peer/cuda__bf16__index_select__small__a1_r32_w31_i15.json) | 0.04568 / 0.02835 |
| `cuda/bf16/index_select/small/r32_w64_i31_a1_strided` | indexing | 32 × 64 | strided | correct | [passed](operator_peer/cuda__bf16__index_select__small__r32_w64_i31_a1_strided.json) | 0.05012 / 0.02538 | [passed](program_peer/cuda__bf16__index_select__small__r32_w64_i31_a1_strided.json) | 0.05489 / 0.02902 |
| `cuda/bf16/index_select/strided/strided_axis0_r2_c7_i2` | indexing | 2 × 7 | strided | correct | [passed](operator_peer/cuda__bf16__index_select__strided__strided_axis0_r2_c7_i2.json) | 0.0241 / 0.0138 | [passed](program_peer/cuda__bf16__index_select__strided__strided_axis0_r2_c7_i2.json) | 0.02929 / 0.01689 |
| `cuda/bf16/index_select/strided/strided_axis1_r65_c7_i4` | indexing | 65 × 7 | strided | correct | [passed](operator_peer/cuda__bf16__index_select__strided__strided_axis1_r65_c7_i4.json) | 0.04013 / 0.01686 | [passed](program_peer/cuda__bf16__index_select__strided__strided_axis1_r65_c7_i4.json) | 0.04513 / 0.02013 |
| `cuda/bf16/index_select/strided/strided_axis1_r65_c127_i64` | indexing | 65 × 127 | strided | correct | [passed](operator_peer/cuda__bf16__index_select__strided__strided_axis1_r65_c127_i64.json) | 0.1146 / 0.05612 | [passed](program_peer/cuda__bf16__index_select__strided__strided_axis1_r65_c127_i64.json) | 0.1197 / 0.05482 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
