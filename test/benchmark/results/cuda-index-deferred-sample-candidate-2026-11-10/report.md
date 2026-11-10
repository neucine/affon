# Parity batch report

Target: `NVIDIA GeForce RTX 3090/7.0.0-42-generic/x86_64`  
Catalog SHA-256: `ad02110fec941a61a5dd0eb93d74b4e72cfecb2ebe545f61898157c86f276bef`  
Binary SHA-256: `b3db0353778c60e0dbe45469cdc58ffb600e5050608b3c61598a2959470686d4`  
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
| `cuda/bf16/gather/large/a0_r64_w64_o32` | indexing | 64 × 64 | dense | correct | [passed](operator_peer/cuda__bf16__gather__large__a0_r64_w64_o32.json) | 0.03606 / 0.03787 | [passed](program_peer/cuda__bf16__gather__large__a0_r64_w64_o32.json) | 0.03846 / 0.03242 |
| `cuda/bf16/gather/large/offset_axis0_r2_c7` | indexing | 2 × 7 | offset | correct | [passed](operator_peer/cuda__bf16__gather__large__offset_axis0_r2_c7.json) | 0.02246 / 0.01465 | [passed](program_peer/cuda__bf16__gather__large__offset_axis0_r2_c7.json) | 0.02489 / 0.01739 |
| `cuda/bf16/gather/large/offset_axis0_r32_c63` | indexing | 32 × 63 | offset | correct | [passed](operator_peer/cuda__bf16__gather__large__offset_axis0_r32_c63.json) | 0.02917 / 0.0264 | [passed](program_peer/cuda__bf16__gather__large__offset_axis0_r32_c63.json) | 0.0316 / 0.02916 |
| `cuda/bf16/gather/large/offset_axis0_r8_c31` | indexing | 8 × 31 | offset | correct | [passed](operator_peer/cuda__bf16__gather__large__offset_axis0_r8_c31.json) | 0.02338 / 0.01603 | [passed](program_peer/cuda__bf16__gather__large__offset_axis0_r8_c31.json) | 0.02595 / 0.01918 |
| `cuda/bf16/gather/large/offset_axis1_r16_c129` | indexing | 16 × 129 | offset | correct | [passed](operator_peer/cuda__bf16__gather__large__offset_axis1_r16_c129.json) | 0.02969 / 0.032 | [passed](program_peer/cuda__bf16__gather__large__offset_axis1_r16_c129.json) | 0.03216 / 0.02893 |
| `cuda/bf16/gather/large/offset_axis1_r3_c3` | indexing | 3 × 3 | offset | correct | [passed](operator_peer/cuda__bf16__gather__large__offset_axis1_r3_c3.json) | 0.02249 / 0.01424 | [passed](program_peer/cuda__bf16__gather__large__offset_axis1_r3_c3.json) | 0.02498 / 0.01758 |
| `cuda/bf16/gather/large/r128_c127_i64` | indexing | 128 × 127 | dense | correct | [passed](operator_peer/cuda__bf16__gather__large__r128_c127_i64.json) | 0.07228 / 0.08712 | [passed](program_peer/cuda__bf16__gather__large__r128_c127_i64.json) | 0.07467 / 0.08116 |
| `cuda/bf16/gather/large/r128_w64_i128x32_a1_offset` | indexing | 128 × 64 | offset | correct | [passed](operator_peer/cuda__bf16__gather__large__r128_w64_i128x32_a1_offset.json) | 0.05669 / 0.06217 | [passed](program_peer/cuda__bf16__gather__large__r128_w64_i128x32_a1_offset.json) | 0.05901 / 0.05424 |
| `cuda/bf16/gather/small/axis0_4x8` | indexing | 4 × 8 | dense | correct | [passed](operator_peer/cuda__bf16__gather__small__axis0_4x8.json) | 0.0227 / 0.01218 | [passed](program_peer/cuda__bf16__gather__small__axis0_4x8.json) | 0.02511 / 0.01486 |
| `cuda/bf16/gather/small/axis0_8x16` | indexing | 8 × 16 | dense | correct | [passed](operator_peer/cuda__bf16__gather__small__axis0_8x16.json) | 0.02337 / 0.01342 | [passed](program_peer/cuda__bf16__gather__small__axis0_8x16.json) | 0.02601 / 0.01665 |
| `cuda/bf16/gather/small/axis1_8x16` | indexing | 8 × 16 | dense | correct | [passed](operator_peer/cuda__bf16__gather__small__axis1_8x16.json) | 0.02356 / 0.01336 | [passed](program_peer/cuda__bf16__gather__small__axis1_8x16.json) | 0.02587 / 0.0164 |
| `cuda/bf16/gather/small/r16_w31_i8x31_a0_strided` | indexing | 16 × 31 | strided | correct | [passed](operator_peer/cuda__bf16__gather__small__r16_w31_i8x31_a0_strided.json) | 0.02412 / 0.01608 | [passed](program_peer/cuda__bf16__gather__small__r16_w31_i8x31_a0_strided.json) | 0.02665 / 0.01927 |
| `cuda/bf16/gather/small/r32_w64_i32x31_a1_strided` | indexing | 32 × 64 | strided | correct | [passed](operator_peer/cuda__bf16__gather__small__r32_w64_i32x31_a1_strided.json) | 0.03065 / 0.03192 | [passed](program_peer/cuda__bf16__gather__small__r32_w64_i32x31_a1_strided.json) | 0.03286 / 0.02734 |
| `cuda/bf16/gather/strided/axis0_r16_c3` | indexing | 16 × 3 | strided | correct | [passed](operator_peer/cuda__bf16__gather__strided__axis0_r16_c3.json) | 0.02329 / 0.01363 | [passed](program_peer/cuda__bf16__gather__strided__axis0_r16_c3.json) | 0.02546 / 0.01688 |
| `cuda/bf16/gather/strided/r64_c127_i32` | indexing | 64 × 127 | strided | correct | [passed](operator_peer/cuda__bf16__gather__strided__r64_c127_i32.json) | 0.04691 / 0.05944 | [passed](program_peer/cuda__bf16__gather__strided__r64_c127_i32.json) | 0.04981 / 0.0542 |
| `cuda/bf16/index_select/large/a0_r64_w64_i31` | indexing | 64 × 64 | dense | correct | [passed](operator_peer/cuda__bf16__index_select__large__a0_r64_w64_i31.json) | 0.03539 / 0.03355 | [passed](program_peer/cuda__bf16__index_select__large__a0_r64_w64_i31.json) | 0.03789 / 0.03406 |
| `cuda/bf16/index_select/large/offset_axis0_r2_c7_i2` | indexing | 2 × 7 | offset | correct | [passed](operator_peer/cuda__bf16__index_select__large__offset_axis0_r2_c7_i2.json) | 0.02221 / 0.0149 | [passed](program_peer/cuda__bf16__index_select__large__offset_axis0_r2_c7_i2.json) | 0.02469 / 0.01764 |
| `cuda/bf16/index_select/large/offset_axis0_r32_c127_i17` | indexing | 32 × 127 | offset | correct | [passed](operator_peer/cuda__bf16__index_select__large__offset_axis0_r32_c127_i17.json) | 0.0353 / 0.03659 | [passed](program_peer/cuda__bf16__index_select__large__offset_axis0_r32_c127_i17.json) | 0.03803 / 0.04043 |
| `cuda/bf16/index_select/large/offset_axis0_r8_c31_i5` | indexing | 8 × 31 | offset | correct | [passed](operator_peer/cuda__bf16__index_select__large__offset_axis0_r8_c31_i5.json) | 0.02329 / 0.01644 | [passed](program_peer/cuda__bf16__index_select__large__offset_axis0_r8_c31_i5.json) | 0.02573 / 0.0196 |
| `cuda/bf16/index_select/large/offset_axis1_r16_c129_i65` | indexing | 16 × 129 | offset | correct | [passed](operator_peer/cuda__bf16__index_select__large__offset_axis1_r16_c129_i65.json) | 0.02941 / 0.02653 | [passed](program_peer/cuda__bf16__index_select__large__offset_axis1_r16_c129_i65.json) | 0.03206 / 0.0297 |
| `cuda/bf16/index_select/large/offset_axis1_r3_c5_i3` | indexing | 3 × 5 | offset | correct | [passed](operator_peer/cuda__bf16__index_select__large__offset_axis1_r3_c5_i3.json) | 0.02233 / 0.01515 | [passed](program_peer/cuda__bf16__index_select__large__offset_axis1_r3_c5_i3.json) | 0.02489 / 0.01769 |
| `cuda/bf16/index_select/large/offset_axis1_r8_c7_i4` | indexing | 8 × 7 | offset | correct | [passed](operator_peer/cuda__bf16__index_select__large__offset_axis1_r8_c7_i4.json) | 0.02323 / 0.01525 | [passed](program_peer/cuda__bf16__index_select__large__offset_axis1_r8_c7_i4.json) | 0.02552 / 0.01851 |
| `cuda/bf16/index_select/large/r128_c127_i64` | indexing | 128 × 127 | dense | correct | [passed](operator_peer/cuda__bf16__index_select__large__r128_c127_i64.json) | 0.07187 / 0.08257 | [passed](program_peer/cuda__bf16__index_select__large__r128_c127_i64.json) | 0.07422 / 0.08381 |
| `cuda/bf16/index_select/small/a0_r33_w32_i17` | indexing | 33 × 32 | dense | correct | [passed](operator_peer/cuda__bf16__index_select__small__a0_r33_w32_i17.json) | 0.02609 / 0.01926 | [passed](program_peer/cuda__bf16__index_select__small__a0_r33_w32_i17.json) | 0.02871 / 0.02273 |
| `cuda/bf16/index_select/small/a1_r32_w32_i16` | indexing | 32 × 32 | dense | correct | [passed](operator_peer/cuda__bf16__index_select__small__a1_r32_w32_i16.json) | 0.02785 / 0.02678 | [passed](program_peer/cuda__bf16__index_select__small__a1_r32_w32_i16.json) | 0.03045 / 0.02891 |
| `cuda/bf16/index_select/small/axis0_len8` | indexing | 16 × 32 | dense | correct | [passed](operator_peer/cuda__bf16__index_select__small__axis0_len8.json) | 0.02405 / 0.02337 | [passed](program_peer/cuda__bf16__index_select__small__axis0_len8.json) | 0.02643 / 0.01822 |
| `cuda/bf16/index_select/small/axis1_len1` | indexing | 16 × 32 | dense | correct | [passed](operator_peer/cuda__bf16__index_select__small__axis1_len1.json) | 0.02383 / 0.01276 | [passed](program_peer/cuda__bf16__index_select__small__axis1_len1.json) | 0.02618 / 0.01599 |
| `cuda/bf16/index_select/small/r32_w32_i16_a0_strided` | indexing | 32 × 32 | strided | correct | [passed](operator_peer/cuda__bf16__index_select__small__r32_w32_i16_a0_strided.json) | 0.02584 / 0.02779 | [passed](program_peer/cuda__bf16__index_select__small__r32_w32_i16_a0_strided.json) | 0.02852 / 0.0308 |
| `cuda/bf16/index_select/strided/r64_c127_i32` | indexing | 64 × 127 | strided | correct | [passed](operator_peer/cuda__bf16__index_select__strided__r64_c127_i32.json) | 0.0469 / 0.05022 | [passed](program_peer/cuda__bf16__index_select__strided__r64_c127_i32.json) | 0.04954 / 0.05427 |
| `cuda/bf16/index_select/strided/strided_axis1_r16_c5_i3` | indexing | 16 × 5 | strided | correct | [passed](operator_peer/cuda__bf16__index_select__strided__strided_axis1_r16_c5_i3.json) | 0.02376 / 0.0144 | [passed](program_peer/cuda__bf16__index_select__strided__strided_axis1_r16_c5_i3.json) | 0.0263 / 0.01781 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
