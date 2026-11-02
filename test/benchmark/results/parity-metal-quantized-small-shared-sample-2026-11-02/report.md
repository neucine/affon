# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `2ef2233d2164aa1c92a19070c626571facae2d0f3a3528cf4224332bd0b3e0c5`  
Binary SHA-256: `e38e754e644d7e0737245efef89c78babe442e593f52cf94a888993fef21e17f`  
Timing: 10 iterations × 5 samples × 3 rounds

## Checks

- Cases run: 3
- Numerical check passed: 3
- Operator check and timing passed: 3
- Program check and timing passed: 3
- Numerical failures: 0
- Operator failures: 0
- Program failures: 0

## Cases

| Case | Family | Shape | Layout | Numerical | Operator | Affon / PyTorch ms | Program | Affon / PyTorch ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `metal/f32/quantized_matmul/transposed/view_m1_k32_n3_b8_g4` | quantized_matrix | 1 × 32 × 3 | transposed | correct | [passed](operator_peer/metal__f32__quantized_matmul__transposed__view_m1_k32_n3_b8_g4.json) | 0.0108 / 0.3057 | [passed](program_peer/metal__f32__quantized_matmul__transposed__view_m1_k32_n3_b8_g4.json) | 0.1689 / 0.2824 |
| `metal/f32/quantized_matmul/transposed/view_m64_k96_n19_b8_g32` | quantized_matrix | 64 × 96 × 19 | transposed | correct | [passed](operator_peer/metal__f32__quantized_matmul__transposed__view_m64_k96_n19_b8_g32.json) | 0.1366 / 0.3967 | [passed](program_peer/metal__f32__quantized_matmul__transposed__view_m64_k96_n19_b8_g32.json) | 0.3287 / 0.3116 |
| `metal/bf16/quantized_matmul/transposed/view_m17_k128_n3_b2_g4` | quantized_matrix | 17 × 128 × 3 | transposed | correct | [passed](operator_peer/metal__bf16__quantized_matmul__transposed__view_m17_k128_n3_b2_g4.json) | 0.021 / 0.3221 | [passed](program_peer/metal__bf16__quantized_matmul__transposed__view_m17_k128_n3_b2_g4.json) | 0.2091 / 0.3501 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
