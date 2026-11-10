# Parity batch report

Target: `NVIDIA GeForce RTX 3090/7.0.0-42-generic/x86_64`  
Catalog SHA-256: `ad02110fec941a61a5dd0eb93d74b4e72cfecb2ebe545f61898157c86f276bef`  
Binary SHA-256: `42a8c02d83f1ffcb4ddeac1c144f2330ec8a7ce96db672bb8fc89c9321e71b2a`  
Timing: 10 iterations × 5 samples × 3 rounds

## Checks

- Cases run: 30
- Numerical check passed: 30
- Operator check and timing passed: 30
- Program check and timing passed: 30
- Numerical failures: 0
- Operator failures: 0
- Program failures: 0
- Operator checks pending: 0
- Program checks pending: 0

## Cases

| Case | Family | Shape | Layout | Numerical | Operator | Affon / PyTorch ms | Program | Affon / PyTorch ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `cuda/bf16/gather/large/a0_r64_w64_o32` | indexing | 64 × 64 | dense | correct | [passed](operator_peer/cuda__bf16__gather__large__a0_r64_w64_o32.json) | 0.03698 / 0.03775 | [passed](program_peer/cuda__bf16__gather__large__a0_r64_w64_o32.json) | 0.04279 / 0.03446 |
| `cuda/bf16/gather/large/offset_axis0_r2_c7` | indexing | 2 × 7 | offset | correct | [passed](operator_peer/cuda__bf16__gather__large__offset_axis0_r2_c7.json) | 0.0241 / 0.0148 | [passed](program_peer/cuda__bf16__gather__large__offset_axis0_r2_c7.json) | 0.02931 / 0.01768 |
| `cuda/bf16/gather/large/offset_axis0_r32_c63` | indexing | 32 × 63 | offset | correct | [passed](operator_peer/cuda__bf16__gather__large__offset_axis0_r32_c63.json) | 0.03085 / 0.02745 | [passed](program_peer/cuda__bf16__gather__large__offset_axis0_r32_c63.json) | 0.03622 / 0.02798 |
| `cuda/bf16/gather/large/offset_axis0_r8_c31` | indexing | 8 × 31 | offset | correct | [passed](operator_peer/cuda__bf16__gather__large__offset_axis0_r8_c31.json) | 0.0251 / 0.01634 | [passed](program_peer/cuda__bf16__gather__large__offset_axis0_r8_c31.json) | 0.03039 / 0.01889 |
| `cuda/bf16/gather/large/offset_axis1_r16_c129` | indexing | 16 × 129 | offset | correct | [passed](operator_peer/cuda__bf16__gather__large__offset_axis1_r16_c129.json) | 0.03127 / 0.03354 | [passed](program_peer/cuda__bf16__gather__large__offset_axis1_r16_c129.json) | 0.03665 / 0.02773 |
| `cuda/bf16/gather/large/offset_axis1_r3_c3` | indexing | 3 × 3 | offset | correct | [passed](operator_peer/cuda__bf16__gather__large__offset_axis1_r3_c3.json) | 0.0241 / 0.01485 | [passed](program_peer/cuda__bf16__gather__large__offset_axis1_r3_c3.json) | 0.02951 / 0.01741 |
| `cuda/bf16/gather/large/r128_c127_i64` | indexing | 128 × 127 | dense | correct | [passed](operator_peer/cuda__bf16__gather__large__r128_c127_i64.json) | 0.07387 / 0.08706 | [passed](program_peer/cuda__bf16__gather__large__r128_c127_i64.json) | 0.07859 / 0.08302 |
| `cuda/bf16/gather/large/r128_w64_i128x32_a1_offset` | indexing | 128 × 64 | offset | correct | [passed](operator_peer/cuda__bf16__gather__large__r128_w64_i128x32_a1_offset.json) | 0.05842 / 0.06241 | [passed](program_peer/cuda__bf16__gather__large__r128_w64_i128x32_a1_offset.json) | 0.06343 / 0.05908 |
| `cuda/bf16/gather/small/axis0_4x8` | indexing | 4 × 8 | dense | correct | [passed](operator_peer/cuda__bf16__gather__small__axis0_4x8.json) | 0.02427 / 0.01239 | [passed](program_peer/cuda__bf16__gather__small__axis0_4x8.json) | 0.02942 / 0.01554 |
| `cuda/bf16/gather/small/axis0_8x16` | indexing | 8 × 16 | dense | correct | [passed](operator_peer/cuda__bf16__gather__small__axis0_8x16.json) | 0.0251 / 0.01321 | [passed](program_peer/cuda__bf16__gather__small__axis0_8x16.json) | 0.03032 / 0.01651 |
| `cuda/bf16/gather/small/axis1_8x16` | indexing | 8 × 16 | dense | correct | [passed](operator_peer/cuda__bf16__gather__small__axis1_8x16.json) | 0.02508 / 0.01335 | [passed](program_peer/cuda__bf16__gather__small__axis1_8x16.json) | 0.03048 / 0.01634 |
| `cuda/bf16/gather/small/r16_w31_i8x31_a0_strided` | indexing | 16 × 31 | strided | correct | [passed](operator_peer/cuda__bf16__gather__small__r16_w31_i8x31_a0_strided.json) | 0.02578 / 0.01578 | [passed](program_peer/cuda__bf16__gather__small__r16_w31_i8x31_a0_strided.json) | 0.03123 / 0.01901 |
| `cuda/bf16/gather/small/r32_w64_i32x31_a1_strided` | indexing | 32 × 64 | strided | correct | [passed](operator_peer/cuda__bf16__gather__small__r32_w64_i32x31_a1_strided.json) | 0.03215 / 0.03179 | [passed](program_peer/cuda__bf16__gather__small__r32_w64_i32x31_a1_strided.json) | 0.03765 / 0.02665 |
| `cuda/bf16/gather/strided/axis0_r16_c3` | indexing | 16 × 3 | strided | correct | [passed](operator_peer/cuda__bf16__gather__strided__axis0_r16_c3.json) | 0.02462 / 0.01399 | [passed](program_peer/cuda__bf16__gather__strided__axis0_r16_c3.json) | 0.03015 / 0.01697 |
| `cuda/bf16/gather/strided/r64_c127_i32` | indexing | 64 × 127 | strided | correct | [passed](operator_peer/cuda__bf16__gather__strided__r64_c127_i32.json) | 0.04879 / 0.06083 | [passed](program_peer/cuda__bf16__gather__strided__r64_c127_i32.json) | 0.05395 / 0.05326 |
| `cuda/bf16/index_select/large/a0_r64_w64_i31` | indexing | 64 × 64 | dense | correct | [passed](operator_peer/cuda__bf16__index_select__large__a0_r64_w64_i31.json) | 0.03716 / 0.03249 | [passed](program_peer/cuda__bf16__index_select__large__a0_r64_w64_i31.json) | 0.04219 / 0.03719 |
| `cuda/bf16/index_select/large/offset_axis0_r2_c7_i2` | indexing | 2 × 7 | offset | correct | [passed](operator_peer/cuda__bf16__index_select__large__offset_axis0_r2_c7_i2.json) | 0.0239 / 0.01483 | [passed](program_peer/cuda__bf16__index_select__large__offset_axis0_r2_c7_i2.json) | 0.02898 / 0.01822 |
| `cuda/bf16/index_select/large/offset_axis0_r32_c127_i17` | indexing | 32 × 127 | offset | correct | [passed](operator_peer/cuda__bf16__index_select__large__offset_axis0_r32_c127_i17.json) | 0.03675 / 0.04007 | [passed](program_peer/cuda__bf16__index_select__large__offset_axis0_r32_c127_i17.json) | 0.04221 / 0.04098 |
| `cuda/bf16/index_select/large/offset_axis0_r8_c31_i5` | indexing | 8 × 31 | offset | correct | [passed](operator_peer/cuda__bf16__index_select__large__offset_axis0_r8_c31_i5.json) | 0.02495 / 0.0165 | [passed](program_peer/cuda__bf16__index_select__large__offset_axis0_r8_c31_i5.json) | 0.03019 / 0.01909 |
| `cuda/bf16/index_select/large/offset_axis1_r16_c129_i65` | indexing | 16 × 129 | offset | correct | [passed](operator_peer/cuda__bf16__index_select__large__offset_axis1_r16_c129_i65.json) | 0.03105 / 0.0264 | [passed](program_peer/cuda__bf16__index_select__large__offset_axis1_r16_c129_i65.json) | 0.03666 / 0.02872 |
| `cuda/bf16/index_select/large/offset_axis1_r3_c5_i3` | indexing | 3 × 5 | offset | correct | [passed](operator_peer/cuda__bf16__index_select__large__offset_axis1_r3_c5_i3.json) | 0.02415 / 0.01475 | [passed](program_peer/cuda__bf16__index_select__large__offset_axis1_r3_c5_i3.json) | 0.0293 / 0.01813 |
| `cuda/bf16/index_select/large/offset_axis1_r8_c7_i4` | indexing | 8 × 7 | offset | correct | [passed](operator_peer/cuda__bf16__index_select__large__offset_axis1_r8_c7_i4.json) | 0.0249 / 0.01525 | [passed](program_peer/cuda__bf16__index_select__large__offset_axis1_r8_c7_i4.json) | 0.03006 / 0.01853 |
| `cuda/bf16/index_select/large/r128_c127_i64` | indexing | 128 × 127 | dense | correct | [passed](operator_peer/cuda__bf16__index_select__large__r128_c127_i64.json) | 0.07373 / 0.08025 | [passed](program_peer/cuda__bf16__index_select__large__r128_c127_i64.json) | 0.07835 / 0.08401 |
| `cuda/bf16/index_select/small/a0_r33_w32_i17` | indexing | 33 × 32 | dense | correct | [passed](operator_peer/cuda__bf16__index_select__small__a0_r33_w32_i17.json) | 0.02785 / 0.01876 | [passed](program_peer/cuda__bf16__index_select__small__a0_r33_w32_i17.json) | 0.03299 / 0.02177 |
| `cuda/bf16/index_select/small/a1_r32_w32_i16` | indexing | 32 × 32 | dense | correct | [passed](operator_peer/cuda__bf16__index_select__small__a1_r32_w32_i16.json) | 0.02934 / 0.02649 | [passed](program_peer/cuda__bf16__index_select__small__a1_r32_w32_i16.json) | 0.03448 / 0.02923 |
| `cuda/bf16/index_select/small/axis0_len8` | indexing | 16 × 32 | dense | correct | [passed](operator_peer/cuda__bf16__index_select__small__axis0_len8.json) | 0.02516 / 0.02328 | [passed](program_peer/cuda__bf16__index_select__small__axis0_len8.json) | 0.03059 / 0.01784 |
| `cuda/bf16/index_select/small/axis1_len1` | indexing | 16 × 32 | dense | correct | [passed](operator_peer/cuda__bf16__index_select__small__axis1_len1.json) | 0.02525 / 0.01266 | [passed](program_peer/cuda__bf16__index_select__small__axis1_len1.json) | 0.03069 / 0.01569 |
| `cuda/bf16/index_select/small/r32_w32_i16_a0_strided` | indexing | 32 × 32 | strided | correct | [passed](operator_peer/cuda__bf16__index_select__small__r32_w32_i16_a0_strided.json) | 0.027 / 0.02762 | [passed](program_peer/cuda__bf16__index_select__small__r32_w32_i16_a0_strided.json) | 0.03271 / 0.03078 |
| `cuda/bf16/index_select/strided/r64_c127_i32` | indexing | 64 × 127 | strided | correct | [passed](operator_peer/cuda__bf16__index_select__strided__r64_c127_i32.json) | 0.04859 / 0.05345 | [passed](program_peer/cuda__bf16__index_select__strided__r64_c127_i32.json) | 0.0541 / 0.05301 |
| `cuda/bf16/index_select/strided/strided_axis1_r16_c5_i3` | indexing | 16 × 5 | strided | correct | [passed](operator_peer/cuda__bf16__index_select__strided__strided_axis1_r16_c5_i3.json) | 0.02536 / 0.01435 | [passed](program_peer/cuda__bf16__index_select__strided__strided_axis1_r16_c5_i3.json) | 0.03069 / 0.01736 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
