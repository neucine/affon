# Parity regression check

Target: `Mac16,1/26.6.2/arm64`

- Cells compared: 128
- Added cells: 0
- Removed cells: 0
- Changed case definitions: 0
- New correctness failures: 0
- Diagnostic timing suspects: 0
- Target matches: yes
- Measurement protocol matches: yes

Timing changes are diagnostic and do not fail this check. Timing is compared only when target and protocol match. No timing baseline is accepted by this comparison.

| Cell | Correctness regression | Operator relative timing | Program relative timing |
| --- | --- | ---: | ---: |
| `metal/bf16/quantized_matmul/decode_m1/grid_m1_k128_n17_b4_g16` | none | 0.03× | 0.30× |
| `metal/bf16/quantized_matmul/decode_m1/grid_m1_k128_n1_b2_g16` | none | 0.02× | 0.36× |
| `metal/bf16/quantized_matmul/decode_m1/grid_m1_k128_n3_b8_g4` | none | 0.03× | 0.26× |
| `metal/bf16/quantized_matmul/decode_m1/grid_m1_k128_n8_b4_g8` | none | 0.02× | 0.26× |
| `metal/bf16/quantized_matmul/decode_m1/grid_m1_k32_n17_b4_g16` | none | 0.02× | 0.29× |
| `metal/bf16/quantized_matmul/decode_m1/grid_m1_k32_n1_b2_g16` | none | 0.02× | 0.28× |
| `metal/bf16/quantized_matmul/decode_m1/grid_m1_k32_n3_b8_g4` | none | 0.03× | 0.23× |
| `metal/bf16/quantized_matmul/decode_m1/grid_m1_k32_n8_b4_g8` | none | 0.02× | 0.26× |
| `metal/bf16/quantized_matmul/decode_m1/grid_m1_k64_n17_b8_g16` | none | 0.02× | 0.38× |
| `metal/bf16/quantized_matmul/decode_m1/grid_m1_k64_n1_b4_g16` | none | 0.02× | 0.32× |
| `metal/bf16/quantized_matmul/decode_m1/grid_m1_k64_n3_b2_g4` | none | 0.02× | 0.29× |
| `metal/bf16/quantized_matmul/decode_m1/grid_m1_k64_n8_b8_g8` | none | 0.02× | 0.33× |
| `metal/bf16/quantized_matmul/decode_m1/grid_m1_k96_n17_b2_g16` | none | 0.03× | 0.43× |
| `metal/bf16/quantized_matmul/decode_m1/grid_m1_k96_n1_b8_g16` | none | 0.02× | 0.37× |
| `metal/bf16/quantized_matmul/decode_m1/grid_m1_k96_n3_b4_g4` | none | 0.02× | 0.32× |
| `metal/bf16/quantized_matmul/decode_m1/grid_m1_k96_n8_b2_g8` | none | 0.02× | 0.31× |
| `metal/bf16/quantized_matmul/matrix/grid_m16_k128_n17_b4_g8` | none | 0.10× | 0.34× |
| `metal/bf16/quantized_matmul/matrix/grid_m16_k128_n1_b2_g8` | none | 0.03× | 0.38× |
| `metal/bf16/quantized_matmul/matrix/grid_m16_k128_n3_b8_g32` | none | 0.03× | 0.41× |
| `metal/bf16/quantized_matmul/matrix/grid_m16_k128_n8_b4_g4` | none | 0.05× | 0.40× |
| `metal/bf16/quantized_matmul/matrix/grid_m16_k32_n17_b4_g8` | none | 0.05× | 0.39× |
| `metal/bf16/quantized_matmul/matrix/grid_m16_k32_n1_b2_g8` | none | 0.02× | 0.32× |
| `metal/bf16/quantized_matmul/matrix/grid_m16_k32_n3_b8_g32` | none | 0.03× | 0.39× |
| `metal/bf16/quantized_matmul/matrix/grid_m16_k32_n8_b4_g4` | none | 0.04× | 0.35× |
| `metal/bf16/quantized_matmul/matrix/grid_m16_k64_n17_b8_g8` | none | 0.06× | 0.37× |
| `metal/bf16/quantized_matmul/matrix/grid_m16_k64_n1_b4_g8` | none | 0.03× | 0.40× |
| `metal/bf16/quantized_matmul/matrix/grid_m16_k64_n3_b2_g32` | none | 0.04× | 0.39× |
| `metal/bf16/quantized_matmul/matrix/grid_m16_k64_n8_b8_g4` | none | 0.05× | 0.37× |
| `metal/bf16/quantized_matmul/matrix/grid_m16_k96_n17_b2_g8` | none | 0.08× | 0.42× |
| `metal/bf16/quantized_matmul/matrix/grid_m16_k96_n1_b8_g8` | none | 0.03× | 0.38× |
| `metal/bf16/quantized_matmul/matrix/grid_m16_k96_n3_b4_g32` | none | 0.04× | 0.36× |
| `metal/bf16/quantized_matmul/matrix/grid_m16_k96_n8_b2_g4` | none | 0.06× | 0.41× |
| `metal/bf16/quantized_matmul/matrix/grid_m8_k128_n17_b8_g8` | none | 0.07× | 0.38× |
| `metal/bf16/quantized_matmul/matrix/grid_m8_k128_n1_b4_g8` | none | 0.03× | 0.43× |
| `metal/bf16/quantized_matmul/matrix/grid_m8_k128_n3_b2_g32` | none | 0.04× | 0.36× |
| `metal/bf16/quantized_matmul/matrix/grid_m8_k128_n8_b8_g4` | none | 0.04× | 0.39× |
| `metal/bf16/quantized_matmul/matrix/grid_m8_k32_n17_b8_g8` | none | 0.03× | 0.30× |
| `metal/bf16/quantized_matmul/matrix/grid_m8_k32_n1_b4_g8` | none | 0.02× | 0.41× |
| `metal/bf16/quantized_matmul/matrix/grid_m8_k32_n3_b2_g32` | none | 0.03× | 0.33× |
| `metal/bf16/quantized_matmul/matrix/grid_m8_k32_n8_b8_g4` | none | 0.03× | 0.41× |
| `metal/bf16/quantized_matmul/matrix/grid_m8_k64_n17_b2_g8` | none | 0.05× | 0.34× |
| `metal/bf16/quantized_matmul/matrix/grid_m8_k64_n1_b8_g8` | none | 0.02× | 0.30× |
| `metal/bf16/quantized_matmul/matrix/grid_m8_k64_n3_b4_g32` | none | 0.02× | 0.37× |
| `metal/bf16/quantized_matmul/matrix/grid_m8_k64_n8_b2_g4` | none | 0.04× | 0.41× |
| `metal/bf16/quantized_matmul/matrix/grid_m8_k96_n17_b4_g8` | none | 0.05× | 0.43× |
| `metal/bf16/quantized_matmul/matrix/grid_m8_k96_n1_b2_g8` | none | 0.02× | 0.33× |
| `metal/bf16/quantized_matmul/matrix/grid_m8_k96_n3_b8_g32` | none | 0.03× | 0.39× |
| `metal/bf16/quantized_matmul/matrix/grid_m8_k96_n8_b4_g4` | none | 0.04× | 0.38× |
| `metal/bf16/quantized_matmul/small_batch/grid_m2_k128_n17_b8_g32` | none | 0.03× | 0.40× |
| `metal/bf16/quantized_matmul/small_batch/grid_m2_k128_n1_b4_g32` | none | 0.02× | 0.36× |
| `metal/bf16/quantized_matmul/small_batch/grid_m2_k128_n3_b2_g8` | none | 0.02× | 0.36× |
| `metal/bf16/quantized_matmul/small_batch/grid_m2_k128_n8_b8_g16` | none | 0.03× | 0.36× |
| `metal/bf16/quantized_matmul/small_batch/grid_m2_k32_n17_b8_g32` | none | 0.02× | 0.44× |
| `metal/bf16/quantized_matmul/small_batch/grid_m2_k32_n1_b4_g32` | none | 0.02× | 0.23× |
| `metal/bf16/quantized_matmul/small_batch/grid_m2_k32_n3_b2_g8` | none | 0.02× | 0.38× |
| `metal/bf16/quantized_matmul/small_batch/grid_m2_k32_n8_b8_g16` | none | 0.02× | 0.36× |
| `metal/bf16/quantized_matmul/small_batch/grid_m2_k64_n17_b2_g32` | none | 0.03× | 0.31× |
| `metal/bf16/quantized_matmul/small_batch/grid_m2_k64_n1_b8_g32` | none | 0.02× | 0.44× |
| `metal/bf16/quantized_matmul/small_batch/grid_m2_k64_n3_b4_g8` | none | 0.02× | 0.53× |
| `metal/bf16/quantized_matmul/small_batch/grid_m2_k64_n8_b2_g16` | none | 0.03× | 0.43× |
| `metal/bf16/quantized_matmul/small_batch/grid_m2_k96_n17_b4_g32` | none | 0.02× | 0.57× |
| `metal/bf16/quantized_matmul/small_batch/grid_m2_k96_n1_b2_g32` | none | 0.02× | 0.32× |
| `metal/bf16/quantized_matmul/small_batch/grid_m2_k96_n3_b8_g8` | none | 0.01× | 0.46× |
| `metal/bf16/quantized_matmul/small_batch/grid_m2_k96_n8_b4_g16` | none | 0.03× | 0.38× |
| `metal/f32/quantized_matmul/decode_m1/grid_m1_k128_n17_b4_g16` | none | 0.03× | 0.47× |
| `metal/f32/quantized_matmul/decode_m1/grid_m1_k128_n1_b2_g16` | none | 0.02× | 0.38× |
| `metal/f32/quantized_matmul/decode_m1/grid_m1_k128_n3_b8_g4` | none | 0.02× | 0.47× |
| `metal/f32/quantized_matmul/decode_m1/grid_m1_k128_n8_b4_g8` | none | 0.03× | 0.44× |
| `metal/f32/quantized_matmul/decode_m1/grid_m1_k32_n17_b4_g16` | none | 0.02× | 0.41× |
| `metal/f32/quantized_matmul/decode_m1/grid_m1_k32_n1_b2_g16` | none | 0.02× | 0.36× |
| `metal/f32/quantized_matmul/decode_m1/grid_m1_k32_n3_b8_g4` | none | 0.02× | 0.48× |
| `metal/f32/quantized_matmul/decode_m1/grid_m1_k32_n8_b4_g8` | none | 0.03× | 0.43× |
| `metal/f32/quantized_matmul/decode_m1/grid_m1_k64_n17_b8_g16` | none | 0.03× | 0.52× |
| `metal/f32/quantized_matmul/decode_m1/grid_m1_k64_n1_b4_g16` | none | 0.02× | 0.39× |
| `metal/f32/quantized_matmul/decode_m1/grid_m1_k64_n3_b2_g4` | none | 0.02× | 0.38× |
| `metal/f32/quantized_matmul/decode_m1/grid_m1_k64_n8_b8_g8` | none | 0.03× | 0.43× |
| `metal/f32/quantized_matmul/decode_m1/grid_m1_k96_n17_b2_g16` | none | 0.02× | 0.49× |
| `metal/f32/quantized_matmul/decode_m1/grid_m1_k96_n1_b8_g16` | none | 0.02× | 0.38× |
| `metal/f32/quantized_matmul/decode_m1/grid_m1_k96_n3_b4_g4` | none | 0.02× | 0.37× |
| `metal/f32/quantized_matmul/decode_m1/grid_m1_k96_n8_b2_g8` | none | 0.02× | 0.39× |
| `metal/f32/quantized_matmul/matrix/grid_m16_k128_n17_b4_g8` | none | 0.09× | 0.25× |
| `metal/f32/quantized_matmul/matrix/grid_m16_k128_n1_b2_g8` | none | 0.03× | 0.23× |
| `metal/f32/quantized_matmul/matrix/grid_m16_k128_n3_b8_g32` | none | 0.04× | 0.27× |
| `metal/f32/quantized_matmul/matrix/grid_m16_k128_n8_b4_g4` | none | 0.06× | 0.21× |
| `metal/f32/quantized_matmul/matrix/grid_m16_k32_n17_b4_g8` | none | 0.05× | 0.30× |
| `metal/f32/quantized_matmul/matrix/grid_m16_k32_n1_b2_g8` | none | 0.03× | 0.24× |
| `metal/f32/quantized_matmul/matrix/grid_m16_k32_n3_b8_g32` | none | 0.03× | 0.26× |
| `metal/f32/quantized_matmul/matrix/grid_m16_k32_n8_b4_g4` | none | 0.03× | 0.18× |
| `metal/f32/quantized_matmul/matrix/grid_m16_k64_n17_b8_g8` | none | 0.06× | 0.34× |
| `metal/f32/quantized_matmul/matrix/grid_m16_k64_n1_b4_g8` | none | 0.03× | 0.24× |
| `metal/f32/quantized_matmul/matrix/grid_m16_k64_n3_b2_g32` | none | 0.04× | 0.30× |
| `metal/f32/quantized_matmul/matrix/grid_m16_k64_n8_b8_g4` | none | 0.05× | 0.30× |
| `metal/f32/quantized_matmul/matrix/grid_m16_k96_n17_b2_g8` | none | 0.08× | 0.32× |
| `metal/f32/quantized_matmul/matrix/grid_m16_k96_n1_b8_g8` | none | 0.03× | 0.27× |
| `metal/f32/quantized_matmul/matrix/grid_m16_k96_n3_b4_g32` | none | 0.03× | 0.29× |
| `metal/f32/quantized_matmul/matrix/grid_m16_k96_n8_b2_g4` | none | 0.06× | 0.22× |
| `metal/f32/quantized_matmul/matrix/grid_m8_k128_n17_b8_g8` | none | 0.06× | 0.28× |
| `metal/f32/quantized_matmul/matrix/grid_m8_k128_n1_b4_g8` | none | 0.03× | 0.23× |
| `metal/f32/quantized_matmul/matrix/grid_m8_k128_n3_b2_g32` | none | 0.03× | 0.25× |
| `metal/f32/quantized_matmul/matrix/grid_m8_k128_n8_b8_g4` | none | 0.04× | 0.31× |
| `metal/f32/quantized_matmul/matrix/grid_m8_k32_n17_b8_g8` | none | 0.03× | 0.45× |
| `metal/f32/quantized_matmul/matrix/grid_m8_k32_n1_b4_g8` | none | 0.02× | 0.42× |
| `metal/f32/quantized_matmul/matrix/grid_m8_k32_n3_b2_g32` | none | 0.03× | 0.42× |
| `metal/f32/quantized_matmul/matrix/grid_m8_k32_n8_b8_g4` | none | 0.03× | 0.23× |
| `metal/f32/quantized_matmul/matrix/grid_m8_k64_n17_b2_g8` | none | 0.04× | 0.29× |
| `metal/f32/quantized_matmul/matrix/grid_m8_k64_n1_b8_g8` | none | 0.03× | 0.31× |
| `metal/f32/quantized_matmul/matrix/grid_m8_k64_n3_b4_g32` | none | 0.02× | 0.41× |
| `metal/f32/quantized_matmul/matrix/grid_m8_k64_n8_b2_g4` | none | 0.03× | 0.35× |
| `metal/f32/quantized_matmul/matrix/grid_m8_k96_n17_b4_g8` | none | 0.06× | 0.27× |
| `metal/f32/quantized_matmul/matrix/grid_m8_k96_n1_b2_g8` | none | 0.02× | 0.22× |
| `metal/f32/quantized_matmul/matrix/grid_m8_k96_n3_b8_g32` | none | 0.03× | 0.21× |
| `metal/f32/quantized_matmul/matrix/grid_m8_k96_n8_b4_g4` | none | 0.04× | 0.34× |
| `metal/f32/quantized_matmul/small_batch/grid_m2_k128_n17_b8_g32` | none | 0.03× | 0.45× |
| `metal/f32/quantized_matmul/small_batch/grid_m2_k128_n1_b4_g32` | none | 0.02× | 0.45× |
| `metal/f32/quantized_matmul/small_batch/grid_m2_k128_n3_b2_g8` | none | 0.02× | 0.40× |
| `metal/f32/quantized_matmul/small_batch/grid_m2_k128_n8_b8_g16` | none | 0.03× | 0.39× |
| `metal/f32/quantized_matmul/small_batch/grid_m2_k32_n17_b8_g32` | none | 0.01× | 0.44× |
| `metal/f32/quantized_matmul/small_batch/grid_m2_k32_n1_b4_g32` | none | 0.02× | 0.38× |
| `metal/f32/quantized_matmul/small_batch/grid_m2_k32_n3_b2_g8` | none | 0.02× | 0.38× |
| `metal/f32/quantized_matmul/small_batch/grid_m2_k32_n8_b8_g16` | none | 0.02× | 0.36× |
| `metal/f32/quantized_matmul/small_batch/grid_m2_k64_n17_b2_g32` | none | 0.03× | 0.45× |
| `metal/f32/quantized_matmul/small_batch/grid_m2_k64_n1_b8_g32` | none | 0.02× | 0.40× |
| `metal/f32/quantized_matmul/small_batch/grid_m2_k64_n3_b4_g8` | none | 0.03× | 0.35× |
| `metal/f32/quantized_matmul/small_batch/grid_m2_k64_n8_b2_g16` | none | 0.02× | 0.42× |
| `metal/f32/quantized_matmul/small_batch/grid_m2_k96_n17_b4_g32` | none | 0.03× | 0.42× |
| `metal/f32/quantized_matmul/small_batch/grid_m2_k96_n1_b2_g32` | none | 0.02× | 0.38× |
| `metal/f32/quantized_matmul/small_batch/grid_m2_k96_n3_b8_g8` | none | 0.03× | 0.37× |
| `metal/f32/quantized_matmul/small_batch/grid_m2_k96_n8_b4_g16` | none | 0.03× | 0.38× |
