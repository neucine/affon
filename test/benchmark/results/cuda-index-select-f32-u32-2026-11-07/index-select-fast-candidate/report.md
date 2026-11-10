# Parity batch report

Target: `NVIDIA GeForce RTX 3090/7.0.0-38-generic/x86_64`  
Catalog SHA-256: `7c778eddf9dbaf466bb383dd417c5d3a70e1df6cdec2b1681c9998fd92f46ef8`  
Binary SHA-256: `6019dcacc6ce96d967924bbee7fdcea13cbce8721db3e3f1585d4b7b96c7c48c`  
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
| `cuda/f32/index_select/large/offset_axis0_r2_c7_i2` | indexing | 2 × 7 | offset | correct | [passed](operator_peer/cuda__f32__index_select__large__offset_axis0_r2_c7_i2.json) | 0.01676 / 0.01435 | [passed](program_peer/cuda__f32__index_select__large__offset_axis0_r2_c7_i2.json) | 0.02138 / 0.01772 |
| `cuda/f32/index_select/large/offset_axis0_r8_c127_i5` | indexing | 8 × 127 | offset | correct | [passed](operator_peer/cuda__f32__index_select__large__offset_axis0_r8_c127_i5.json) | 0.02891 / 0.02136 | [passed](program_peer/cuda__f32__index_select__large__offset_axis0_r8_c127_i5.json) | 0.03423 / 0.02445 |
| `cuda/f32/index_select/large/r128_c127_i64` | indexing | 128 × 127 | dense | correct | [passed](operator_peer/cuda__f32__index_select__large__r128_c127_i64.json) | 0.1712 / 0.08007 | [passed](program_peer/cuda__f32__index_select__large__r128_c127_i64.json) | 0.1712 / 0.08442 |
| `cuda/f32/index_select/large/offset_axis1_r3_c5_i3` | indexing | 3 × 5 | offset | correct | [passed](operator_peer/cuda__f32__index_select__large__offset_axis1_r3_c5_i3.json) | 0.01688 / 0.01431 | [passed](program_peer/cuda__f32__index_select__large__offset_axis1_r3_c5_i3.json) | 0.02161 / 0.01717 |
| `cuda/f32/index_select/large/offset_axis1_r2_c127_i64` | indexing | 2 × 127 | offset | correct | [passed](operator_peer/cuda__f32__index_select__large__offset_axis1_r2_c127_i64.json) | 0.01885 / 0.01747 | [passed](program_peer/cuda__f32__index_select__large__offset_axis1_r2_c127_i64.json) | 0.02378 / 0.02005 |
| `cuda/f32/index_select/large/offset_axis1_r65_c127_i64` | indexing | 65 × 127 | offset | correct | [passed](operator_peer/cuda__f32__index_select__large__offset_axis1_r65_c127_i64.json) | 0.1112 / 0.05997 | [passed](program_peer/cuda__f32__index_select__large__offset_axis1_r65_c127_i64.json) | 0.1146 / 0.06331 |
| `cuda/f32/index_select/small` | indexing | 8 × 16 | dense | correct | [passed](operator_peer/cuda__f32__index_select__small.json) | 0.01854 / 0.01261 | [passed](program_peer/cuda__f32__index_select__small.json) | 0.02323 / 0.01561 |
| `cuda/f32/index_select/small/a0_r31_w32_i15` | indexing | 31 × 32 | dense | correct | [passed](operator_peer/cuda__f32__index_select__small__a0_r31_w32_i15.json) | 0.02897 / 0.02484 | [passed](program_peer/cuda__f32__index_select__small__a0_r31_w32_i15.json) | 0.03434 / 0.02885 |
| `cuda/f32/index_select/small/a0_r33_w32_i17` | indexing | 33 × 32 | dense | correct | [passed](operator_peer/cuda__f32__index_select__small__a0_r33_w32_i17.json) | 0.03108 / 0.01954 | [passed](program_peer/cuda__f32__index_select__small__a0_r33_w32_i17.json) | 0.03651 / 0.02264 |
| `cuda/f32/index_select/small/axis1_len1` | indexing | 16 × 32 | dense | correct | [passed](operator_peer/cuda__f32__index_select__small__axis1_len1.json) | 0.01808 / 0.0123 | [passed](program_peer/cuda__f32__index_select__small__axis1_len1.json) | 0.02289 / 0.01534 |
| `cuda/f32/index_select/small/a1_r32_w32_i16` | indexing | 32 × 32 | dense | correct | [passed](operator_peer/cuda__f32__index_select__small__a1_r32_w32_i16.json) | 0.03365 / 0.02697 | [passed](program_peer/cuda__f32__index_select__small__a1_r32_w32_i16.json) | 0.03865 / 0.02983 |
| `cuda/f32/index_select/small/r32_w64_i31_a1_strided` | indexing | 32 × 64 | strided | correct | [passed](operator_peer/cuda__f32__index_select__small__r32_w64_i31_a1_strided.json) | 0.04368 / 0.02646 | [passed](program_peer/cuda__f32__index_select__small__r32_w64_i31_a1_strided.json) | 0.04875 / 0.03029 |
| `cuda/f32/index_select/strided/strided_axis0_r2_c7_i2` | indexing | 2 × 7 | strided | correct | [passed](operator_peer/cuda__f32__index_select__strided__strided_axis0_r2_c7_i2.json) | 0.01683 / 0.01339 | [passed](program_peer/cuda__f32__index_select__strided__strided_axis0_r2_c7_i2.json) | 0.02164 / 0.01639 |
| `cuda/f32/index_select/strided/strided_axis0_r7_c129_i4` | indexing | 7 × 129 | strided | correct | [passed](operator_peer/cuda__f32__index_select__strided__strided_axis0_r7_c129_i4.json) | 0.02663 / 0.01928 | [passed](program_peer/cuda__f32__index_select__strided__strided_axis0_r7_c129_i4.json) | 0.03196 / 0.02228 |
| `cuda/f32/index_select/strided/strided_axis0_r65_c127_i33` | indexing | 65 × 127 | strided | correct | [passed](operator_peer/cuda__f32__index_select__strided__strided_axis0_r65_c127_i33.json) | 0.09877 / 0.05545 | [passed](program_peer/cuda__f32__index_select__strided__strided_axis0_r65_c127_i33.json) | 0.1028 / 0.05478 |
| `cuda/f32/index_select/strided/strided_axis1_r3_c5_i3` | indexing | 3 × 5 | strided | correct | [passed](operator_peer/cuda__f32__index_select__strided__strided_axis1_r3_c5_i3.json) | 0.01681 / 0.01358 | [passed](program_peer/cuda__f32__index_select__strided__strided_axis1_r3_c5_i3.json) | 0.02156 / 0.01648 |
| `cuda/f32/index_select/strided/strided_axis1_r32_c31_i16` | indexing | 32 × 31 | strided | correct | [passed](operator_peer/cuda__f32__index_select__strided__strided_axis1_r32_c31_i16.json) | 0.03334 / 0.02795 | [passed](program_peer/cuda__f32__index_select__strided__strided_axis1_r32_c31_i16.json) | 0.03912 / 0.03112 |
| `cuda/f32/index_select/strided/strided_axis1_r65_c127_i64` | indexing | 65 × 127 | strided | correct | [passed](operator_peer/cuda__f32__index_select__strided__strided_axis1_r65_c127_i64.json) | 0.1117 / 0.06047 | [passed](program_peer/cuda__f32__index_select__strided__strided_axis1_r65_c127_i64.json) | 0.1137 / 0.05781 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
