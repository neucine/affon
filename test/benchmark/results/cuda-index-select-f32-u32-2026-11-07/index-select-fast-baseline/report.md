# Parity batch report

Target: `NVIDIA GeForce RTX 3090/7.0.0-38-generic/x86_64`  
Catalog SHA-256: `7c778eddf9dbaf466bb383dd417c5d3a70e1df6cdec2b1681c9998fd92f46ef8`  
Binary SHA-256: `f3b481e37eb9d5a16e7a2208f85cf952f644d5b35102d5e989ddacefb345641b`  
Timing: 10 iterations × 5 samples × 3 rounds

## Checks

- Cases run: 18
- Numerical check passed: 18
- Operator check and timing passed: 18
- Program check and timing passed: 18
- Numerical failures: 0
- Operator failures: 0
- Program failures: 0
- Operator checks pending: 0
- Program checks pending: 0

## Cases

| Case | Family | Shape | Layout | Numerical | Operator | Affon / PyTorch ms | Program | Affon / PyTorch ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `cuda/f32/index_select/large/offset_axis0_r2_c7_i2` | indexing | 2 × 7 | offset | correct | [passed](operator_peer/cuda__f32__index_select__large__offset_axis0_r2_c7_i2.json) | 0.01718 / 0.01444 | [passed](program_peer/cuda__f32__index_select__large__offset_axis0_r2_c7_i2.json) | 0.02209 / 0.0173 |
| `cuda/f32/index_select/large/offset_axis0_r8_c127_i5` | indexing | 8 × 127 | offset | correct | [passed](operator_peer/cuda__f32__index_select__large__offset_axis0_r8_c127_i5.json) | 0.02943 / 0.02122 | [passed](program_peer/cuda__f32__index_select__large__offset_axis0_r8_c127_i5.json) | 0.03473 / 0.02493 |
| `cuda/f32/index_select/large/r128_c127_i64` | indexing | 128 × 127 | dense | correct | [passed](operator_peer/cuda__f32__index_select__large__r128_c127_i64.json) | 0.1709 / 0.08206 | [passed](program_peer/cuda__f32__index_select__large__r128_c127_i64.json) | 0.1739 / 0.08668 |
| `cuda/f32/index_select/large/offset_axis1_r3_c5_i3` | indexing | 3 × 5 | offset | correct | [passed](operator_peer/cuda__f32__index_select__large__offset_axis1_r3_c5_i3.json) | 0.01714 / 0.01481 | [passed](program_peer/cuda__f32__index_select__large__offset_axis1_r3_c5_i3.json) | 0.0222 / 0.01746 |
| `cuda/f32/index_select/large/offset_axis1_r2_c127_i64` | indexing | 2 × 127 | offset | correct | [passed](operator_peer/cuda__f32__index_select__large__offset_axis1_r2_c127_i64.json) | 0.01942 / 0.01709 | [passed](program_peer/cuda__f32__index_select__large__offset_axis1_r2_c127_i64.json) | 0.02434 / 0.02004 |
| `cuda/f32/index_select/large/offset_axis1_r65_c127_i64` | indexing | 65 × 127 | offset | correct | [passed](operator_peer/cuda__f32__index_select__large__offset_axis1_r65_c127_i64.json) | 0.1123 / 0.06266 | [passed](program_peer/cuda__f32__index_select__large__offset_axis1_r65_c127_i64.json) | 0.114 / 0.06169 |
| `cuda/f32/index_select/small` | indexing | 8 × 16 | dense | correct | [passed](operator_peer/cuda__f32__index_select__small.json) | 0.01905 / 0.01248 | [passed](program_peer/cuda__f32__index_select__small.json) | 0.02384 / 0.01562 |
| `cuda/f32/index_select/small/a0_r31_w32_i15` | indexing | 31 × 32 | dense | correct | [passed](operator_peer/cuda__f32__index_select__small__a0_r31_w32_i15.json) | 0.02972 / 0.02512 | [passed](program_peer/cuda__f32__index_select__small__a0_r31_w32_i15.json) | 0.03516 / 0.02785 |
| `cuda/f32/index_select/small/a0_r33_w32_i17` | indexing | 33 × 32 | dense | correct | [passed](operator_peer/cuda__f32__index_select__small__a0_r33_w32_i17.json) | 0.03099 / 0.01979 | [passed](program_peer/cuda__f32__index_select__small__a0_r33_w32_i17.json) | 0.03699 / 0.02282 |
| `cuda/f32/index_select/small/axis1_len1` | indexing | 16 × 32 | dense | correct | [passed](operator_peer/cuda__f32__index_select__small__axis1_len1.json) | 0.01849 / 0.01231 | [passed](program_peer/cuda__f32__index_select__small__axis1_len1.json) | 0.02353 / 0.01563 |
| `cuda/f32/index_select/small/a1_r32_w32_i16` | indexing | 32 × 32 | dense | correct | [passed](operator_peer/cuda__f32__index_select__small__a1_r32_w32_i16.json) | 0.03398 / 0.02671 | [passed](program_peer/cuda__f32__index_select__small__a1_r32_w32_i16.json) | 0.03981 / 0.02993 |
| `cuda/f32/index_select/small/r32_w64_i31_a1_strided` | indexing | 32 × 64 | strided | correct | [passed](operator_peer/cuda__f32__index_select__small__r32_w64_i31_a1_strided.json) | 0.04372 / 0.02372 | [passed](program_peer/cuda__f32__index_select__small__r32_w64_i31_a1_strided.json) | 0.04941 / 0.03071 |
| `cuda/f32/index_select/strided/strided_axis0_r2_c7_i2` | indexing | 2 × 7 | strided | correct | [passed](operator_peer/cuda__f32__index_select__strided__strided_axis0_r2_c7_i2.json) | 0.01716 / 0.01349 | [passed](program_peer/cuda__f32__index_select__strided__strided_axis0_r2_c7_i2.json) | 0.02189 / 0.01674 |
| `cuda/f32/index_select/strided/strided_axis0_r7_c129_i4` | indexing | 7 × 129 | strided | correct | [passed](operator_peer/cuda__f32__index_select__strided__strided_axis0_r7_c129_i4.json) | 0.02715 / 0.01927 | [passed](program_peer/cuda__f32__index_select__strided__strided_axis0_r7_c129_i4.json) | 0.03266 / 0.02242 |
| `cuda/f32/index_select/strided/strided_axis0_r65_c127_i33` | indexing | 65 × 127 | strided | correct | [passed](operator_peer/cuda__f32__index_select__strided__strided_axis0_r65_c127_i33.json) | 0.1001 / 0.0596 | [passed](program_peer/cuda__f32__index_select__strided__strided_axis0_r65_c127_i33.json) | 0.1025 / 0.05484 |
| `cuda/f32/index_select/strided/strided_axis1_r3_c5_i3` | indexing | 3 × 5 | strided | correct | [passed](operator_peer/cuda__f32__index_select__strided__strided_axis1_r3_c5_i3.json) | 0.01732 / 0.01349 | [passed](program_peer/cuda__f32__index_select__strided__strided_axis1_r3_c5_i3.json) | 0.02211 / 0.01674 |
| `cuda/f32/index_select/strided/strided_axis1_r32_c31_i16` | indexing | 32 × 31 | strided | correct | [passed](operator_peer/cuda__f32__index_select__strided__strided_axis1_r32_c31_i16.json) | 0.0341 / 0.02842 | [passed](program_peer/cuda__f32__index_select__strided__strided_axis1_r32_c31_i16.json) | 0.03945 / 0.03007 |
| `cuda/f32/index_select/strided/strided_axis1_r65_c127_i64` | indexing | 65 × 127 | strided | correct | [passed](operator_peer/cuda__f32__index_select__strided__strided_axis1_r65_c127_i64.json) | 0.1114 / 0.05233 | [passed](program_peer/cuda__f32__index_select__strided__strided_axis1_r65_c127_i64.json) | 0.1142 / 0.05577 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
