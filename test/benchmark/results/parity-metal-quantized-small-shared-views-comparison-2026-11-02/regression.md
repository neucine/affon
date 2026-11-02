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
| `metal/bf16/quantized_matmul/ragged/offset_view_m17_k128_n19_b4_g4` | none | 0.09× | 0.40× |
| `metal/bf16/quantized_matmul/ragged/offset_view_m17_k128_n3_b2_g4` | none | 0.03× | 0.35× |
| `metal/bf16/quantized_matmul/ragged/offset_view_m17_k32_n19_b4_g4` | none | 0.06× | 0.38× |
| `metal/bf16/quantized_matmul/ragged/offset_view_m17_k32_n3_b2_g4` | none | 0.03× | 0.40× |
| `metal/bf16/quantized_matmul/ragged/offset_view_m17_k64_n19_b8_g4` | none | 0.07× | 0.47× |
| `metal/bf16/quantized_matmul/ragged/offset_view_m17_k64_n3_b4_g4` | none | 0.03× | 0.37× |
| `metal/bf16/quantized_matmul/ragged/offset_view_m17_k96_n19_b2_g4` | none | 0.09× | 0.43× |
| `metal/bf16/quantized_matmul/ragged/offset_view_m17_k96_n3_b8_g4` | none | 0.04× | 0.37× |
| `metal/bf16/quantized_matmul/ragged/offset_view_m1_k128_n19_b2_g4` | none | 0.03× | 0.40× |
| `metal/bf16/quantized_matmul/ragged/offset_view_m1_k128_n3_b8_g4` | none | 0.02× | 0.35× |
| `metal/bf16/quantized_matmul/ragged/offset_view_m1_k32_n19_b2_g4` | none | 0.02× | 0.34× |
| `metal/bf16/quantized_matmul/ragged/offset_view_m1_k32_n3_b8_g4` | none | 0.02× | 0.41× |
| `metal/bf16/quantized_matmul/ragged/offset_view_m1_k64_n19_b4_g4` | none | 0.02× | 0.46× |
| `metal/bf16/quantized_matmul/ragged/offset_view_m1_k64_n3_b2_g4` | none | 0.02× | 0.36× |
| `metal/bf16/quantized_matmul/ragged/offset_view_m1_k96_n19_b8_g4` | none | 0.03× | 0.43× |
| `metal/bf16/quantized_matmul/ragged/offset_view_m1_k96_n3_b4_g4` | none | 0.02× | 0.42× |
| `metal/bf16/quantized_matmul/ragged/offset_view_m3_k128_n19_b8_g16` | none | 0.03× | 0.39× |
| `metal/bf16/quantized_matmul/ragged/offset_view_m3_k128_n3_b4_g16` | none | 0.03× | 0.40× |
| `metal/bf16/quantized_matmul/ragged/offset_view_m3_k32_n19_b8_g16` | none | 0.03× | 0.44× |
| `metal/bf16/quantized_matmul/ragged/offset_view_m3_k32_n3_b4_g16` | none | 0.02× | 0.42× |
| `metal/bf16/quantized_matmul/ragged/offset_view_m3_k64_n19_b2_g16` | none | 0.03× | 0.41× |
| `metal/bf16/quantized_matmul/ragged/offset_view_m3_k64_n3_b8_g16` | none | 0.03× | 0.45× |
| `metal/bf16/quantized_matmul/ragged/offset_view_m3_k96_n19_b4_g16` | none | 0.04× | 0.42× |
| `metal/bf16/quantized_matmul/ragged/offset_view_m3_k96_n3_b2_g16` | none | 0.02× | 0.47× |
| `metal/bf16/quantized_matmul/ragged/offset_view_m64_k128_n19_b2_g32` | none | 0.24× | 0.50× |
| `metal/bf16/quantized_matmul/ragged/offset_view_m64_k128_n3_b8_g32` | none | 0.08× | 0.40× |
| `metal/bf16/quantized_matmul/ragged/offset_view_m64_k32_n19_b2_g32` | none | 0.10× | 0.32× |
| `metal/bf16/quantized_matmul/ragged/offset_view_m64_k32_n3_b8_g32` | none | 0.05× | 0.36× |
| `metal/bf16/quantized_matmul/ragged/offset_view_m64_k64_n19_b4_g32` | none | 0.15× | 0.45× |
| `metal/bf16/quantized_matmul/ragged/offset_view_m64_k64_n3_b2_g32` | none | 0.06× | 0.38× |
| `metal/bf16/quantized_matmul/ragged/offset_view_m64_k96_n19_b8_g32` | none | 0.21× | 0.44× |
| `metal/bf16/quantized_matmul/ragged/offset_view_m64_k96_n3_b4_g32` | none | 0.06× | 0.36× |
| `metal/bf16/quantized_matmul/transposed/view_m17_k128_n19_b4_g4` | none | 0.10× | 0.42× |
| `metal/bf16/quantized_matmul/transposed/view_m17_k128_n3_b2_g4` | none | 0.04× | 0.37× |
| `metal/bf16/quantized_matmul/transposed/view_m17_k32_n19_b4_g4` | none | 0.04× | 0.40× |
| `metal/bf16/quantized_matmul/transposed/view_m17_k32_n3_b2_g4` | none | 0.03× | 0.37× |
| `metal/bf16/quantized_matmul/transposed/view_m17_k64_n19_b8_g4` | none | 0.07× | 0.45× |
| `metal/bf16/quantized_matmul/transposed/view_m17_k64_n3_b4_g4` | none | 0.03× | 0.36× |
| `metal/bf16/quantized_matmul/transposed/view_m17_k96_n19_b2_g4` | none | 0.09× | 0.44× |
| `metal/bf16/quantized_matmul/transposed/view_m17_k96_n3_b8_g4` | none | 0.04× | 0.42× |
| `metal/bf16/quantized_matmul/transposed/view_m1_k128_n19_b2_g4` | none | 0.03× | 0.26× |
| `metal/bf16/quantized_matmul/transposed/view_m1_k128_n3_b8_g4` | none | 0.02× | 0.44× |
| `metal/bf16/quantized_matmul/transposed/view_m1_k32_n19_b2_g4` | none | 0.02× | 0.38× |
| `metal/bf16/quantized_matmul/transposed/view_m1_k32_n3_b8_g4` | none | 0.02× | 0.40× |
| `metal/bf16/quantized_matmul/transposed/view_m1_k64_n19_b4_g4` | none | 0.03× | 0.40× |
| `metal/bf16/quantized_matmul/transposed/view_m1_k64_n3_b2_g4` | none | 0.02× | 0.34× |
| `metal/bf16/quantized_matmul/transposed/view_m1_k96_n19_b8_g4` | none | 0.02× | 0.44× |
| `metal/bf16/quantized_matmul/transposed/view_m1_k96_n3_b4_g4` | none | 0.02× | 0.31× |
| `metal/bf16/quantized_matmul/transposed/view_m3_k128_n19_b8_g16` | none | 0.04× | 0.42× |
| `metal/bf16/quantized_matmul/transposed/view_m3_k128_n3_b4_g16` | none | 0.03× | 0.46× |
| `metal/bf16/quantized_matmul/transposed/view_m3_k32_n19_b8_g16` | none | 0.03× | 0.39× |
| `metal/bf16/quantized_matmul/transposed/view_m3_k32_n3_b4_g16` | none | 0.02× | 0.46× |
| `metal/bf16/quantized_matmul/transposed/view_m3_k64_n19_b2_g16` | none | 0.03× | 0.41× |
| `metal/bf16/quantized_matmul/transposed/view_m3_k64_n3_b8_g16` | none | 0.03× | 0.41× |
| `metal/bf16/quantized_matmul/transposed/view_m3_k96_n19_b4_g16` | none | 0.03× | 0.42× |
| `metal/bf16/quantized_matmul/transposed/view_m3_k96_n3_b2_g16` | none | 0.02× | 0.36× |
| `metal/bf16/quantized_matmul/transposed/view_m64_k128_n19_b2_g32` | none | 0.24× | 1.08× |
| `metal/bf16/quantized_matmul/transposed/view_m64_k128_n3_b8_g32` | none | 0.09× | 0.42× |
| `metal/bf16/quantized_matmul/transposed/view_m64_k32_n19_b2_g32` | none | 0.12× | 0.36× |
| `metal/bf16/quantized_matmul/transposed/view_m64_k32_n3_b8_g32` | none | 0.05× | 0.36× |
| `metal/bf16/quantized_matmul/transposed/view_m64_k64_n19_b4_g32` | none | 0.16× | 0.42× |
| `metal/bf16/quantized_matmul/transposed/view_m64_k64_n3_b2_g32` | none | 0.06× | 0.49× |
| `metal/bf16/quantized_matmul/transposed/view_m64_k96_n19_b8_g32` | none | 0.19× | 0.43× |
| `metal/bf16/quantized_matmul/transposed/view_m64_k96_n3_b4_g32` | none | 0.07× | 0.43× |
| `metal/f32/quantized_matmul/ragged/offset_view_m17_k128_n19_b4_g4` | none | 0.10× | 0.42× |
| `metal/f32/quantized_matmul/ragged/offset_view_m17_k128_n3_b2_g4` | none | 0.03× | 0.38× |
| `metal/f32/quantized_matmul/ragged/offset_view_m17_k32_n19_b4_g4` | none | 0.05× | 0.43× |
| `metal/f32/quantized_matmul/ragged/offset_view_m17_k32_n3_b2_g4` | none | 0.03× | 0.34× |
| `metal/f32/quantized_matmul/ragged/offset_view_m17_k64_n19_b8_g4` | none | 0.06× | 0.34× |
| `metal/f32/quantized_matmul/ragged/offset_view_m17_k64_n3_b4_g4` | none | 0.03× | 0.37× |
| `metal/f32/quantized_matmul/ragged/offset_view_m17_k96_n19_b2_g4` | none | 0.08× | 0.62× |
| `metal/f32/quantized_matmul/ragged/offset_view_m17_k96_n3_b8_g4` | none | 0.04× | 0.37× |
| `metal/f32/quantized_matmul/ragged/offset_view_m1_k128_n19_b2_g4` | none | 0.02× | 0.29× |
| `metal/f32/quantized_matmul/ragged/offset_view_m1_k128_n3_b8_g4` | none | 0.02× | 0.43× |
| `metal/f32/quantized_matmul/ragged/offset_view_m1_k32_n19_b2_g4` | none | 0.02× | 0.33× |
| `metal/f32/quantized_matmul/ragged/offset_view_m1_k32_n3_b8_g4` | none | 0.02× | 0.39× |
| `metal/f32/quantized_matmul/ragged/offset_view_m1_k64_n19_b4_g4` | none | 0.02× | 0.33× |
| `metal/f32/quantized_matmul/ragged/offset_view_m1_k64_n3_b2_g4` | none | 0.02× | 0.37× |
| `metal/f32/quantized_matmul/ragged/offset_view_m1_k96_n19_b8_g4` | none | 0.03× | 0.39× |
| `metal/f32/quantized_matmul/ragged/offset_view_m1_k96_n3_b4_g4` | none | 0.02× | 0.35× |
| `metal/f32/quantized_matmul/ragged/offset_view_m3_k128_n19_b8_g16` | none | 0.04× | 0.38× |
| `metal/f32/quantized_matmul/ragged/offset_view_m3_k128_n3_b4_g16` | none | 0.02× | 0.43× |
| `metal/f32/quantized_matmul/ragged/offset_view_m3_k32_n19_b8_g16` | none | 0.03× | 0.40× |
| `metal/f32/quantized_matmul/ragged/offset_view_m3_k32_n3_b4_g16` | none | 0.02× | 0.37× |
| `metal/f32/quantized_matmul/ragged/offset_view_m3_k64_n19_b2_g16` | none | 0.03× | 0.40× |
| `metal/f32/quantized_matmul/ragged/offset_view_m3_k64_n3_b8_g16` | none | 0.03× | 0.32× |
| `metal/f32/quantized_matmul/ragged/offset_view_m3_k96_n19_b4_g16` | none | 0.04× | 0.39× |
| `metal/f32/quantized_matmul/ragged/offset_view_m3_k96_n3_b2_g16` | none | 0.03× | 0.43× |
| `metal/f32/quantized_matmul/ragged/offset_view_m64_k128_n19_b2_g32` | none | 0.19× | 0.51× |
| `metal/f32/quantized_matmul/ragged/offset_view_m64_k128_n3_b8_g32` | none | 0.08× | 0.43× |
| `metal/f32/quantized_matmul/ragged/offset_view_m64_k32_n19_b2_g32` | none | 0.10× | 0.36× |
| `metal/f32/quantized_matmul/ragged/offset_view_m64_k32_n3_b8_g32` | none | 0.05× | 0.42× |
| `metal/f32/quantized_matmul/ragged/offset_view_m64_k64_n19_b4_g32` | none | 0.14× | 0.40× |
| `metal/f32/quantized_matmul/ragged/offset_view_m64_k64_n3_b2_g32` | none | 0.06× | 0.38× |
| `metal/f32/quantized_matmul/ragged/offset_view_m64_k96_n19_b8_g32` | none | 0.24× | 0.28× |
| `metal/f32/quantized_matmul/ragged/offset_view_m64_k96_n3_b4_g32` | none | 0.07× | 0.39× |
| `metal/f32/quantized_matmul/transposed/view_m17_k128_n19_b4_g4` | none | 0.09× | 0.42× |
| `metal/f32/quantized_matmul/transposed/view_m17_k128_n3_b2_g4` | none | 0.04× | 0.34× |
| `metal/f32/quantized_matmul/transposed/view_m17_k32_n19_b4_g4` | none | 0.05× | 0.40× |
| `metal/f32/quantized_matmul/transposed/view_m17_k32_n3_b2_g4` | none | 0.03× | 0.41× |
| `metal/f32/quantized_matmul/transposed/view_m17_k64_n19_b8_g4` | none | 0.07× | 0.22× |
| `metal/f32/quantized_matmul/transposed/view_m17_k64_n3_b4_g4` | none | 0.03× | 0.35× |
| `metal/f32/quantized_matmul/transposed/view_m17_k96_n19_b2_g4` | none | 0.08× | 0.39× |
| `metal/f32/quantized_matmul/transposed/view_m17_k96_n3_b8_g4` | none | 0.03× | 0.40× |
| `metal/f32/quantized_matmul/transposed/view_m1_k128_n19_b2_g4` | none | 0.03× | 0.39× |
| `metal/f32/quantized_matmul/transposed/view_m1_k128_n3_b8_g4` | none | 0.03× | 0.40× |
| `metal/f32/quantized_matmul/transposed/view_m1_k32_n19_b2_g4` | none | 0.02× | 0.35× |
| `metal/f32/quantized_matmul/transposed/view_m1_k32_n3_b8_g4` | none | 0.03× | 0.37× |
| `metal/f32/quantized_matmul/transposed/view_m1_k64_n19_b4_g4` | none | 0.02× | 0.33× |
| `metal/f32/quantized_matmul/transposed/view_m1_k64_n3_b2_g4` | none | 0.02× | 0.47× |
| `metal/f32/quantized_matmul/transposed/view_m1_k96_n19_b8_g4` | none | 0.03× | 0.38× |
| `metal/f32/quantized_matmul/transposed/view_m1_k96_n3_b4_g4` | none | 0.02× | 0.38× |
| `metal/f32/quantized_matmul/transposed/view_m3_k128_n19_b8_g16` | none | 0.04× | 0.44× |
| `metal/f32/quantized_matmul/transposed/view_m3_k128_n3_b4_g16` | none | 0.02× | 0.38× |
| `metal/f32/quantized_matmul/transposed/view_m3_k32_n19_b8_g16` | none | 0.03× | 0.42× |
| `metal/f32/quantized_matmul/transposed/view_m3_k32_n3_b4_g16` | none | 0.03× | 0.37× |
| `metal/f32/quantized_matmul/transposed/view_m3_k64_n19_b2_g16` | none | 0.02× | 0.48× |
| `metal/f32/quantized_matmul/transposed/view_m3_k64_n3_b8_g16` | none | 0.03× | 0.35× |
| `metal/f32/quantized_matmul/transposed/view_m3_k96_n19_b4_g16` | none | 0.03× | 0.36× |
| `metal/f32/quantized_matmul/transposed/view_m3_k96_n3_b2_g16` | none | 0.03× | 0.42× |
| `metal/f32/quantized_matmul/transposed/view_m64_k128_n19_b2_g32` | none | 0.27× | 0.50× |
| `metal/f32/quantized_matmul/transposed/view_m64_k128_n3_b8_g32` | none | 0.08× | 0.40× |
| `metal/f32/quantized_matmul/transposed/view_m64_k32_n19_b2_g32` | none | 0.06× | 0.37× |
| `metal/f32/quantized_matmul/transposed/view_m64_k32_n3_b8_g32` | none | 0.05× | 0.41× |
| `metal/f32/quantized_matmul/transposed/view_m64_k64_n19_b4_g32` | none | 0.13× | 0.45× |
| `metal/f32/quantized_matmul/transposed/view_m64_k64_n3_b2_g32` | none | 0.06× | 0.46× |
| `metal/f32/quantized_matmul/transposed/view_m64_k96_n19_b8_g32` | none | 0.20× | 0.43× |
| `metal/f32/quantized_matmul/transposed/view_m64_k96_n3_b4_g32` | none | 0.07× | 0.38× |
