# Parity batch report

Target: `NVIDIA GeForce RTX 3090/7.0.0-42-generic/x86_64`  
Catalog SHA-256: `f6267864b24c303d10d81305e53033c4f62507f3cc432b4119a330c1f72369e2`  
Binary SHA-256: `92b13d04776a56d3c7b9d157d954e47dc5b2cf351a58b240e439541a8e140a3d`  
Timing: 10 iterations × 5 samples × 3 rounds

## Checks

- Cases run: 11
- Numerical check passed: 11
- Operator check and timing passed: 11
- Program check and timing passed: 11
- Numerical failures: 0
- Operator failures: 0
- Program failures: 0
- Operator checks pending: 0
- Program checks pending: 0

## Cases

| Case | Family | Shape | Layout | Numerical | Operator | Affon / PyTorch ms | Program | Affon / PyTorch ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `cuda/bf16/index_select/batched/dense_axis1_b2_r16_c129_n8` | indexing | 2 × 16 × 129 | dense | correct | [passed](operator_peer/cuda__bf16__index_select__batched__dense_axis1_b2_r16_c129_n8.json) | 0.02978 / 0.03694 | [passed](program_peer/cuda__bf16__index_select__batched__dense_axis1_b2_r16_c129_n8.json) | 0.03259 / 0.03632 |
| `cuda/bf16/index_select/batched/dense_axis1_b4_r33_c5_n16` | indexing | 4 × 33 × 5 | dense | correct | [passed](operator_peer/cuda__bf16__index_select__batched__dense_axis1_b4_r33_c5_n16.json) | 0.02695 / 0.02466 | [passed](program_peer/cuda__bf16__index_select__batched__dense_axis1_b4_r33_c5_n16.json) | 0.02937 / 0.02783 |
| `cuda/f32/index_select/batched/dense_axis2_b4_r33_c5_n2` | indexing | 4 × 33 × 5 | dense | correct | [passed](operator_peer/cuda__f32__index_select__batched__dense_axis2_b4_r33_c5_n2.json) | 0.03463 / 0.01708 | [passed](program_peer/cuda__f32__index_select__batched__dense_axis2_b4_r33_c5_n2.json) | 0.04046 / 0.02028 |
| `cuda/bf16/index_select/large` | indexing | 128 × 64 | dense | correct | [passed](operator_peer/cuda__bf16__index_select__large.json) | 0.03012 / 0.03314 | [passed](program_peer/cuda__bf16__index_select__large.json) | 0.03261 / 0.0362 |
| `cuda/bf16/index_select/small` | indexing | 8 × 16 | dense | correct | [passed](operator_peer/cuda__bf16__index_select__small.json) | 0.01736 / 0.01235 | [passed](program_peer/cuda__bf16__index_select__small.json) | 0.01981 / 0.01535 |
| `cuda/bf16/gather/large/a0_r64_w64_o32` | indexing | 64 × 64 | dense | correct | [passed](operator_peer/cuda__bf16__gather__large__a0_r64_w64_o32.json) | 0.03528 / 0.03784 | [passed](program_peer/cuda__bf16__gather__large__a0_r64_w64_o32.json) | 0.03805 / 0.03481 |
| `cuda/bf16/gather/large/a1_r64_w64_o32` | indexing | 64 × 64 | dense | correct | [passed](operator_peer/cuda__bf16__gather__large__a1_r64_w64_o32.json) | 0.03899 / 0.04037 | [passed](program_peer/cuda__bf16__gather__large__a1_r64_w64_o32.json) | 0.04166 / 0.03475 |
| `cuda/bf16/gather/large/offset_axis0_r16_c129` | indexing | 16 × 129 | offset | correct | [passed](operator_peer/cuda__bf16__gather__large__offset_axis0_r16_c129.json) | 0.02892 / 0.03079 | [passed](program_peer/cuda__bf16__gather__large__offset_axis0_r16_c129.json) | 0.03138 / 0.02901 |
| `cuda/bf16/gather/strided/axis1_r65_c7` | indexing | 65 × 7 | strided | correct | [passed](operator_peer/cuda__bf16__gather__strided__axis1_r65_c7.json) | 0.03085 / 0.01675 | [passed](program_peer/cuda__bf16__gather__strided__axis1_r65_c7.json) | 0.03312 / 0.02077 |
| `cuda/bf16/gather/small/a1_r32_w7_o1` | indexing | 32 × 7 | dense | correct | [passed](operator_peer/cuda__bf16__gather__small__a1_r32_w7_o1.json) | 0.0256 / 0.01244 | [passed](program_peer/cuda__bf16__gather__small__a1_r32_w7_o1.json) | 0.02794 / 0.01575 |
| `cuda/f32/gather/strided/axis1_r65_c7` | indexing | 65 × 7 | strided | correct | [passed](operator_peer/cuda__f32__gather__strided__axis1_r65_c7.json) | 0.02569 / 0.01713 | [passed](program_peer/cuda__f32__gather__strided__axis1_r65_c7.json) | 0.03114 / 0.02097 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
