# Parity batch report

Target: `NVIDIA GeForce RTX 3090/7.0.0-38-generic/x86_64`  
Catalog SHA-256: `7ec7a5a4166ba0887c0927fd922c93490bc3f5499edeacfa5f1eee320b02fd3c`  
Binary SHA-256: `8346d573e917ab34d6bd745d4cef1c5dadea07f9f29f1c02b5bf7ab6e962eff6`  
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
| `cuda/bf16/embedding/batched` | indexing | 16 × 32 | dense | correct | [passed](operator_peer/cuda__bf16__embedding__batched.json) | 0.03045 / 0.02319 | [passed](program_peer/cuda__bf16__embedding__batched.json) | 0.03595 / 0.01859 |
| `cuda/bf16/embedding/batched/batch3` | indexing | 64 × 32 | dense | correct | [passed](operator_peer/cuda__bf16__embedding__batched__batch3.json) | 0.03782 / 0.02602 | [passed](program_peer/cuda__bf16__embedding__batched__batch3.json) | 0.04301 / 0.02879 |
| `cuda/bf16/embedding/batched/batch8` | indexing | 64 × 32 | dense | correct | [passed](operator_peer/cuda__bf16__embedding__batched__batch8.json) | 0.08132 / 0.03518 | [passed](program_peer/cuda__bf16__embedding__batched__batch8.json) | 0.08741 / 0.03734 |
| `cuda/bf16/embedding/large/r64_w31_i31` | indexing | 64 × 31 | dense | correct | [passed](operator_peer/cuda__bf16__embedding__large__r64_w31_i31.json) | 0.04985 / 0.02716 | [passed](program_peer/cuda__bf16__embedding__large__r64_w31_i31.json) | 0.05504 / 0.02651 |
| `cuda/bf16/embedding/large/rows65` | indexing | 65 × 64 | dense | correct | [passed](operator_peer/cuda__bf16__embedding__large__rows65.json) | 0.04575 / 0.03104 | [passed](program_peer/cuda__bf16__embedding__large__rows65.json) | 0.05131 / 0.03289 |
| `cuda/bf16/embedding/large/r128_c127_i64` | indexing | 128 × 127 | dense | correct | [passed](operator_peer/cuda__bf16__embedding__large__r128_c127_i64.json) | 0.1765 / 0.08202 | [passed](program_peer/cuda__bf16__embedding__large__r128_c127_i64.json) | 0.1812 / 0.08374 |
| `cuda/bf16/embedding/small` | indexing | 8 × 16 | dense | correct | [passed](operator_peer/cuda__bf16__embedding__small.json) | 0.02591 / 0.01294 | [passed](program_peer/cuda__bf16__embedding__small.json) | 0.03108 / 0.01603 |
| `cuda/bf16/embedding/small/r32_w16_i16` | indexing | 32 × 16 | dense | correct | [passed](operator_peer/cuda__bf16__embedding__small__r32_w16_i16.json) | 0.03218 / 0.02364 | [passed](program_peer/cuda__bf16__embedding__small__r32_w16_i16.json) | 0.0372 / 0.02634 |
| `cuda/bf16/embedding/small/r32_w32_i16_strided` | indexing | 32 × 32 | strided | correct | [passed](operator_peer/cuda__bf16__embedding__small__r32_w32_i16_strided.json) | 0.03718 / 0.02699 | [passed](program_peer/cuda__bf16__embedding__small__r32_w32_i16_strided.json) | 0.04219 / 0.02982 |
| `cuda/bf16/embedding/strided` | indexing | 8 × 16 | strided | correct | [passed](operator_peer/cuda__bf16__embedding__strided.json) | 0.02585 / 0.01434 | [passed](program_peer/cuda__bf16__embedding__strided.json) | 0.03111 / 0.01776 |
| `cuda/bf16/embedding/strided/r64_c127_i16` | indexing | 64 × 127 | strided | correct | [passed](operator_peer/cuda__bf16__embedding__strided__r64_c127_i16.json) | 0.06235 / 0.03873 | [passed](program_peer/cuda__bf16__embedding__strided__r64_c127_i16.json) | 0.06753 / 0.04266 |
| `cuda/bf16/gather/batched` | indexing | 16 × 32 | dense | correct | [passed](operator_peer/cuda__bf16__gather__batched.json) | 0.02997 / 0.0144 | [passed](program_peer/cuda__bf16__gather__batched.json) | 0.03542 / 0.01753 |
| `cuda/bf16/gather/large/offset_axis0_r3_c3` | indexing | 3 × 3 | offset | correct | [passed](operator_peer/cuda__bf16__gather__large__offset_axis0_r3_c3.json) | 0.02401 / 0.01424 | [passed](program_peer/cuda__bf16__gather__large__offset_axis0_r3_c3.json) | 0.02929 / 0.01748 |
| `cuda/bf16/gather/large/offset_axis1_r8_c63` | indexing | 8 × 63 | offset | correct | [passed](operator_peer/cuda__bf16__gather__large__offset_axis1_r8_c63.json) | 0.03052 / 0.01676 | [passed](program_peer/cuda__bf16__gather__large__offset_axis1_r8_c63.json) | 0.03556 / 0.01978 |
| `cuda/bf16/gather/large/r128_c127_i64` | indexing | 128 × 127 | dense | correct | [passed](operator_peer/cuda__bf16__gather__large__r128_c127_i64.json) | 0.1753 / 0.0851 | [passed](program_peer/cuda__bf16__gather__large__r128_c127_i64.json) | 0.1806 / 0.08048 |
| `cuda/bf16/gather/small` | indexing | 4 × 8 | dense | correct | [passed](operator_peer/cuda__bf16__gather__small.json) | 0.02524 / 0.01212 | [passed](program_peer/cuda__bf16__gather__small.json) | 0.03031 / 0.01529 |
| `cuda/bf16/gather/small/a1_r32_w31_o16` | indexing | 32 × 31 | dense | correct | [passed](operator_peer/cuda__bf16__gather__small__a1_r32_w31_o16.json) | 0.04076 / 0.01755 | [passed](program_peer/cuda__bf16__gather__small__a1_r32_w31_o16.json) | 0.04657 / 0.021 |
| `cuda/bf16/gather/small/r32_w64_i32x31_a1_strided` | indexing | 32 × 64 | strided | correct | [passed](operator_peer/cuda__bf16__gather__small__r32_w64_i32x31_a1_strided.json) | 0.05066 / 0.03182 | [passed](program_peer/cuda__bf16__gather__small__r32_w64_i32x31_a1_strided.json) | 0.05607 / 0.02617 |
| `cuda/bf16/gather/strided/axis0_r3_c3` | indexing | 3 × 3 | strided | correct | [passed](operator_peer/cuda__bf16__gather__strided__axis0_r3_c3.json) | 0.02407 / 0.01322 | [passed](program_peer/cuda__bf16__gather__strided__axis0_r3_c3.json) | 0.02925 / 0.01649 |
| `cuda/bf16/gather/strided/axis0_r7_c64` | indexing | 7 × 64 | strided | correct | [passed](operator_peer/cuda__bf16__gather__strided__axis0_r7_c64.json) | 0.02917 / 0.01578 | [passed](program_peer/cuda__bf16__gather__strided__axis0_r7_c64.json) | 0.03466 / 0.01934 |
| `cuda/bf16/gather/strided/axis1_r65_c127` | indexing | 65 × 127 | strided | correct | [passed](operator_peer/cuda__bf16__gather__strided__axis1_r65_c127.json) | 0.1165 / 0.06217 | [passed](program_peer/cuda__bf16__gather__strided__axis1_r65_c127.json) | 0.1204 / 0.05391 |
| `cuda/bf16/index_select/batched/dense_axis1_b2_r3_c5_n1` | indexing | 2 × 3 × 5 | dense | correct | [passed](operator_peer/cuda__bf16__index_select__batched__dense_axis1_b2_r3_c5_n1.json) | 0.02441 / 0.01202 | [passed](program_peer/cuda__bf16__index_select__batched__dense_axis1_b2_r3_c5_n1.json) | 0.03006 / 0.01574 |
| `cuda/bf16/index_select/batched/dense_axis2_b4_r3_c65_n32` | indexing | 4 × 3 × 65 | dense | correct | [passed](operator_peer/cuda__bf16__index_select__batched__dense_axis2_b4_r3_c65_n32.json) | 0.03504 / 0.01754 | [passed](program_peer/cuda__bf16__index_select__batched__dense_axis2_b4_r3_c65_n32.json) | 0.04047 / 0.02108 |
| `cuda/bf16/index_select/batched/dense_axis2_b4_r33_c129_n64` | indexing | 4 × 33 × 129 | dense | correct | [passed](operator_peer/cuda__bf16__index_select__batched__dense_axis2_b4_r33_c129_n64.json) | 0.2066 / 0.08588 | [passed](program_peer/cuda__bf16__index_select__batched__dense_axis2_b4_r33_c129_n64.json) | 0.21 / 0.0896 |
| `cuda/bf16/index_select/large/offset_axis0_r2_c7_i2` | indexing | 2 × 7 | offset | correct | [passed](operator_peer/cuda__bf16__index_select__large__offset_axis0_r2_c7_i2.json) | 0.02413 / 0.01446 | [passed](program_peer/cuda__bf16__index_select__large__offset_axis0_r2_c7_i2.json) | 0.02944 / 0.0181 |
| `cuda/bf16/index_select/large/offset_axis0_r7_c129_i4` | indexing | 7 × 129 | offset | correct | [passed](operator_peer/cuda__bf16__index_select__large__offset_axis0_r7_c129_i4.json) | 0.03374 / 0.01914 | [passed](program_peer/cuda__bf16__index_select__large__offset_axis0_r7_c129_i4.json) | 0.03904 / 0.02307 |
| `cuda/bf16/index_select/large/r128_c127_i64` | indexing | 128 × 127 | dense | correct | [passed](operator_peer/cuda__bf16__index_select__large__r128_c127_i64.json) | 0.176 / 0.07968 | [passed](program_peer/cuda__bf16__index_select__large__r128_c127_i64.json) | 0.1802 / 0.08275 |
| `cuda/bf16/index_select/small` | indexing | 8 × 16 | dense | correct | [passed](operator_peer/cuda__bf16__index_select__small.json) | 0.02569 / 0.01263 | [passed](program_peer/cuda__bf16__index_select__small.json) | 0.03111 / 0.01591 |
| `cuda/bf16/index_select/small/a1_r32_w31_i15` | indexing | 32 × 31 | dense | correct | [passed](operator_peer/cuda__bf16__index_select__small__a1_r32_w31_i15.json) | 0.04006 / 0.02569 | [passed](program_peer/cuda__bf16__index_select__small__a1_r32_w31_i15.json) | 0.04535 / 0.02874 |
| `cuda/bf16/index_select/small/r32_w64_i31_a1_strided` | indexing | 32 × 64 | strided | correct | [passed](operator_peer/cuda__bf16__index_select__small__r32_w64_i31_a1_strided.json) | 0.05031 / 0.02536 | [passed](program_peer/cuda__bf16__index_select__small__r32_w64_i31_a1_strided.json) | 0.05554 / 0.02818 |
| `cuda/bf16/index_select/strided/strided_axis0_r2_c7_i2` | indexing | 2 × 7 | strided | correct | [passed](operator_peer/cuda__bf16__index_select__strided__strided_axis0_r2_c7_i2.json) | 0.02429 / 0.0136 | [passed](program_peer/cuda__bf16__index_select__strided__strided_axis0_r2_c7_i2.json) | 0.02933 / 0.01772 |
| `cuda/bf16/index_select/strided/strided_axis1_r65_c7_i4` | indexing | 65 × 7 | strided | correct | [passed](operator_peer/cuda__bf16__index_select__strided__strided_axis1_r65_c7_i4.json) | 0.04009 / 0.01737 | [passed](program_peer/cuda__bf16__index_select__strided__strided_axis1_r65_c7_i4.json) | 0.04513 / 0.02024 |
| `cuda/bf16/index_select/strided/strided_axis1_r65_c127_i64` | indexing | 65 × 127 | strided | correct | [passed](operator_peer/cuda__bf16__index_select__strided__strided_axis1_r65_c127_i64.json) | 0.116 / 0.05485 | [passed](program_peer/cuda__bf16__index_select__strided__strided_axis1_r65_c127_i64.json) | 0.1202 / 0.05325 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
