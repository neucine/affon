# Parity batch report

Target: `NVIDIA GeForce RTX 3090/7.0.0-42-generic/x86_64`  
Catalog SHA-256: `f6267864b24c303d10d81305e53033c4f62507f3cc432b4119a330c1f72369e2`  
Binary SHA-256: `dce57e5223002a1e77668a3869779c27ba584e739ae57afd35f16f0af3e25aa5`  
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
| `cuda/bf16/index_select/batched/dense_axis1_b2_r16_c129_n8` | indexing | 2 × 16 × 129 | dense | correct | [passed](operator_peer/cuda__bf16__index_select__batched__dense_axis1_b2_r16_c129_n8.json) | 0.03017 / 0.03687 | [passed](program_peer/cuda__bf16__index_select__batched__dense_axis1_b2_r16_c129_n8.json) | 0.033 / 0.03387 |
| `cuda/bf16/index_select/batched/dense_axis1_b4_r33_c5_n16` | indexing | 4 × 33 × 5 | dense | correct | [passed](operator_peer/cuda__bf16__index_select__batched__dense_axis1_b4_r33_c5_n16.json) | 0.02755 / 0.02441 | [passed](program_peer/cuda__bf16__index_select__batched__dense_axis1_b4_r33_c5_n16.json) | 0.0299 / 0.02769 |
| `cuda/f32/index_select/batched/dense_axis2_b4_r33_c5_n2` | indexing | 4 × 33 × 5 | dense | correct | [passed](operator_peer/cuda__f32__index_select__batched__dense_axis2_b4_r33_c5_n2.json) | 0.0351 / 0.01712 | [passed](program_peer/cuda__f32__index_select__batched__dense_axis2_b4_r33_c5_n2.json) | 0.04071 / 0.02039 |
| `cuda/bf16/index_select/large` | indexing | 128 × 64 | dense | correct | [passed](operator_peer/cuda__bf16__index_select__large.json) | 0.03062 / 0.03288 | [passed](program_peer/cuda__bf16__index_select__large.json) | 0.03316 / 0.03726 |
| `cuda/bf16/index_select/small` | indexing | 8 × 16 | dense | correct | [passed](operator_peer/cuda__bf16__index_select__small.json) | 0.01784 / 0.0124 | [passed](program_peer/cuda__bf16__index_select__small.json) | 0.02032 / 0.0158 |
| `cuda/bf16/gather/large/a0_r64_w64_o32` | indexing | 64 × 64 | dense | correct | [passed](operator_peer/cuda__bf16__gather__large__a0_r64_w64_o32.json) | 0.03595 / 0.04039 | [passed](program_peer/cuda__bf16__gather__large__a0_r64_w64_o32.json) | 0.03838 / 0.0358 |
| `cuda/bf16/gather/large/a1_r64_w64_o32` | indexing | 64 × 64 | dense | correct | [passed](operator_peer/cuda__bf16__gather__large__a1_r64_w64_o32.json) | 0.03967 / 0.04017 | [passed](program_peer/cuda__bf16__gather__large__a1_r64_w64_o32.json) | 0.04228 / 0.03307 |
| `cuda/bf16/gather/large/offset_axis0_r16_c129` | indexing | 16 × 129 | offset | correct | [passed](operator_peer/cuda__bf16__gather__large__offset_axis0_r16_c129.json) | 0.02933 / 0.03111 | [passed](program_peer/cuda__bf16__gather__large__offset_axis0_r16_c129.json) | 0.03189 / 0.02894 |
| `cuda/bf16/gather/strided/axis1_r65_c7` | indexing | 65 × 7 | strided | correct | [passed](operator_peer/cuda__bf16__gather__strided__axis1_r65_c7.json) | 0.03139 / 0.01671 | [passed](program_peer/cuda__bf16__gather__strided__axis1_r65_c7.json) | 0.03398 / 0.02007 |
| `cuda/bf16/gather/small/a1_r32_w7_o1` | indexing | 32 × 7 | dense | correct | [passed](operator_peer/cuda__bf16__gather__small__a1_r32_w7_o1.json) | 0.02604 / 0.01308 | [passed](program_peer/cuda__bf16__gather__small__a1_r32_w7_o1.json) | 0.02838 / 0.01612 |
| `cuda/f32/gather/strided/axis1_r65_c7` | indexing | 65 × 7 | strided | correct | [passed](operator_peer/cuda__f32__gather__strided__axis1_r65_c7.json) | 0.02575 / 0.01796 | [passed](program_peer/cuda__f32__gather__strided__axis1_r65_c7.json) | 0.03125 / 0.02083 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
