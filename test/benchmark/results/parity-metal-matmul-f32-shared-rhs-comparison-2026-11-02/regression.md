# Parity regression check

Target: `Mac16,1/26.6.2/arm64`

- Cells compared: 128
- Added cells: 0
- Removed cells: 0
- Changed case definitions: 0
- New correctness failures: 0
- Diagnostic timing suspects: 0
- Target matches: yes
- Measurement protocol matches: no

Timing changes are diagnostic and do not fail this check. Timing is compared only when target and protocol match. No timing baseline is accepted by this comparison.

| Cell | Correctness regression | Operator relative timing | Program relative timing |
| --- | --- | ---: | ---: |
| `metal/bf16/matmul/strided/rhs_m16_k129_n17` | none | — | — |
| `metal/bf16/matmul/strided/rhs_m16_k129_n65` | none | — | — |
| `metal/bf16/matmul/strided/rhs_m16_k31_n17` | none | — | — |
| `metal/bf16/matmul/strided/rhs_m16_k31_n65` | none | — | — |
| `metal/bf16/matmul/strided/rhs_m16_k64_n17` | none | — | — |
| `metal/bf16/matmul/strided/rhs_m16_k64_n65` | none | — | — |
| `metal/bf16/matmul/strided/rhs_m16_k7_n17` | none | — | — |
| `metal/bf16/matmul/strided/rhs_m16_k7_n65` | none | — | — |
| `metal/bf16/matmul/strided/rhs_m1_k129_n17` | none | — | — |
| `metal/bf16/matmul/strided/rhs_m1_k129_n65` | none | — | — |
| `metal/bf16/matmul/strided/rhs_m1_k31_n17` | none | — | — |
| `metal/bf16/matmul/strided/rhs_m1_k31_n65` | none | — | — |
| `metal/bf16/matmul/strided/rhs_m1_k64_n17` | none | — | — |
| `metal/bf16/matmul/strided/rhs_m1_k64_n65` | none | — | — |
| `metal/bf16/matmul/strided/rhs_m1_k7_n17` | none | — | — |
| `metal/bf16/matmul/strided/rhs_m1_k7_n65` | none | — | — |
| `metal/bf16/matmul/strided/rhs_m4_k129_n17` | none | — | — |
| `metal/bf16/matmul/strided/rhs_m4_k129_n65` | none | — | — |
| `metal/bf16/matmul/strided/rhs_m4_k31_n17` | none | — | — |
| `metal/bf16/matmul/strided/rhs_m4_k31_n65` | none | — | — |
| `metal/bf16/matmul/strided/rhs_m4_k64_n17` | none | — | — |
| `metal/bf16/matmul/strided/rhs_m4_k64_n65` | none | — | — |
| `metal/bf16/matmul/strided/rhs_m4_k7_n17` | none | — | — |
| `metal/bf16/matmul/strided/rhs_m4_k7_n65` | none | — | — |
| `metal/bf16/matmul/strided/rhs_m64_k129_n17` | none | — | — |
| `metal/bf16/matmul/strided/rhs_m64_k129_n65` | none | — | — |
| `metal/bf16/matmul/strided/rhs_m64_k31_n17` | none | — | — |
| `metal/bf16/matmul/strided/rhs_m64_k31_n65` | none | — | — |
| `metal/bf16/matmul/strided/rhs_m64_k64_n17` | none | — | — |
| `metal/bf16/matmul/strided/rhs_m64_k64_n65` | none | — | — |
| `metal/bf16/matmul/strided/rhs_m64_k7_n17` | none | — | — |
| `metal/bf16/matmul/strided/rhs_m64_k7_n65` | none | — | — |
| `metal/bf16/matmul/transposed/rhs_m16_k129_n17` | none | — | — |
| `metal/bf16/matmul/transposed/rhs_m16_k129_n65` | none | — | — |
| `metal/bf16/matmul/transposed/rhs_m16_k31_n17` | none | — | — |
| `metal/bf16/matmul/transposed/rhs_m16_k31_n65` | none | — | — |
| `metal/bf16/matmul/transposed/rhs_m16_k64_n17` | none | — | — |
| `metal/bf16/matmul/transposed/rhs_m16_k64_n65` | none | — | — |
| `metal/bf16/matmul/transposed/rhs_m16_k7_n17` | none | — | — |
| `metal/bf16/matmul/transposed/rhs_m16_k7_n65` | none | — | — |
| `metal/bf16/matmul/transposed/rhs_m1_k129_n17` | none | — | — |
| `metal/bf16/matmul/transposed/rhs_m1_k129_n65` | none | — | — |
| `metal/bf16/matmul/transposed/rhs_m1_k31_n17` | none | — | — |
| `metal/bf16/matmul/transposed/rhs_m1_k31_n65` | none | — | — |
| `metal/bf16/matmul/transposed/rhs_m1_k64_n17` | none | — | — |
| `metal/bf16/matmul/transposed/rhs_m1_k64_n65` | none | — | — |
| `metal/bf16/matmul/transposed/rhs_m1_k7_n17` | none | — | — |
| `metal/bf16/matmul/transposed/rhs_m1_k7_n65` | none | — | — |
| `metal/bf16/matmul/transposed/rhs_m4_k129_n17` | none | — | — |
| `metal/bf16/matmul/transposed/rhs_m4_k129_n65` | none | — | — |
| `metal/bf16/matmul/transposed/rhs_m4_k31_n17` | none | — | — |
| `metal/bf16/matmul/transposed/rhs_m4_k31_n65` | none | — | — |
| `metal/bf16/matmul/transposed/rhs_m4_k64_n17` | none | — | — |
| `metal/bf16/matmul/transposed/rhs_m4_k64_n65` | none | — | — |
| `metal/bf16/matmul/transposed/rhs_m4_k7_n17` | none | — | — |
| `metal/bf16/matmul/transposed/rhs_m4_k7_n65` | none | — | — |
| `metal/bf16/matmul/transposed/rhs_m64_k129_n17` | none | — | — |
| `metal/bf16/matmul/transposed/rhs_m64_k129_n65` | none | — | — |
| `metal/bf16/matmul/transposed/rhs_m64_k31_n17` | none | — | — |
| `metal/bf16/matmul/transposed/rhs_m64_k31_n65` | none | — | — |
| `metal/bf16/matmul/transposed/rhs_m64_k64_n17` | none | — | — |
| `metal/bf16/matmul/transposed/rhs_m64_k64_n65` | none | — | — |
| `metal/bf16/matmul/transposed/rhs_m64_k7_n17` | none | — | — |
| `metal/bf16/matmul/transposed/rhs_m64_k7_n65` | none | — | — |
| `metal/f32/matmul/strided/rhs_m16_k129_n17` | none | — | — |
| `metal/f32/matmul/strided/rhs_m16_k129_n65` | none | — | — |
| `metal/f32/matmul/strided/rhs_m16_k31_n17` | none | — | — |
| `metal/f32/matmul/strided/rhs_m16_k31_n65` | none | — | — |
| `metal/f32/matmul/strided/rhs_m16_k64_n17` | none | — | — |
| `metal/f32/matmul/strided/rhs_m16_k64_n65` | none | — | — |
| `metal/f32/matmul/strided/rhs_m16_k7_n17` | none | — | — |
| `metal/f32/matmul/strided/rhs_m16_k7_n65` | none | — | — |
| `metal/f32/matmul/strided/rhs_m1_k129_n17` | none | — | — |
| `metal/f32/matmul/strided/rhs_m1_k129_n65` | none | — | — |
| `metal/f32/matmul/strided/rhs_m1_k31_n17` | none | — | — |
| `metal/f32/matmul/strided/rhs_m1_k31_n65` | none | — | — |
| `metal/f32/matmul/strided/rhs_m1_k64_n17` | none | — | — |
| `metal/f32/matmul/strided/rhs_m1_k64_n65` | none | — | — |
| `metal/f32/matmul/strided/rhs_m1_k7_n17` | none | — | — |
| `metal/f32/matmul/strided/rhs_m1_k7_n65` | none | — | — |
| `metal/f32/matmul/strided/rhs_m4_k129_n17` | none | — | — |
| `metal/f32/matmul/strided/rhs_m4_k129_n65` | none | — | — |
| `metal/f32/matmul/strided/rhs_m4_k31_n17` | none | — | — |
| `metal/f32/matmul/strided/rhs_m4_k31_n65` | none | — | — |
| `metal/f32/matmul/strided/rhs_m4_k64_n17` | none | — | — |
| `metal/f32/matmul/strided/rhs_m4_k64_n65` | none | — | — |
| `metal/f32/matmul/strided/rhs_m4_k7_n17` | none | — | — |
| `metal/f32/matmul/strided/rhs_m4_k7_n65` | none | — | — |
| `metal/f32/matmul/strided/rhs_m64_k129_n17` | none | — | — |
| `metal/f32/matmul/strided/rhs_m64_k129_n65` | none | — | — |
| `metal/f32/matmul/strided/rhs_m64_k31_n17` | none | — | — |
| `metal/f32/matmul/strided/rhs_m64_k31_n65` | none | — | — |
| `metal/f32/matmul/strided/rhs_m64_k64_n17` | none | — | — |
| `metal/f32/matmul/strided/rhs_m64_k64_n65` | none | — | — |
| `metal/f32/matmul/strided/rhs_m64_k7_n17` | none | — | — |
| `metal/f32/matmul/strided/rhs_m64_k7_n65` | none | — | — |
| `metal/f32/matmul/transposed/rhs_m16_k129_n17` | none | — | — |
| `metal/f32/matmul/transposed/rhs_m16_k129_n65` | none | — | — |
| `metal/f32/matmul/transposed/rhs_m16_k31_n17` | none | — | — |
| `metal/f32/matmul/transposed/rhs_m16_k31_n65` | none | — | — |
| `metal/f32/matmul/transposed/rhs_m16_k64_n17` | none | — | — |
| `metal/f32/matmul/transposed/rhs_m16_k64_n65` | none | — | — |
| `metal/f32/matmul/transposed/rhs_m16_k7_n17` | none | — | — |
| `metal/f32/matmul/transposed/rhs_m16_k7_n65` | none | — | — |
| `metal/f32/matmul/transposed/rhs_m1_k129_n17` | none | — | — |
| `metal/f32/matmul/transposed/rhs_m1_k129_n65` | none | — | — |
| `metal/f32/matmul/transposed/rhs_m1_k31_n17` | none | — | — |
| `metal/f32/matmul/transposed/rhs_m1_k31_n65` | none | — | — |
| `metal/f32/matmul/transposed/rhs_m1_k64_n17` | none | — | — |
| `metal/f32/matmul/transposed/rhs_m1_k64_n65` | none | — | — |
| `metal/f32/matmul/transposed/rhs_m1_k7_n17` | none | — | — |
| `metal/f32/matmul/transposed/rhs_m1_k7_n65` | none | — | — |
| `metal/f32/matmul/transposed/rhs_m4_k129_n17` | none | — | — |
| `metal/f32/matmul/transposed/rhs_m4_k129_n65` | none | — | — |
| `metal/f32/matmul/transposed/rhs_m4_k31_n17` | none | — | — |
| `metal/f32/matmul/transposed/rhs_m4_k31_n65` | none | — | — |
| `metal/f32/matmul/transposed/rhs_m4_k64_n17` | none | — | — |
| `metal/f32/matmul/transposed/rhs_m4_k64_n65` | none | — | — |
| `metal/f32/matmul/transposed/rhs_m4_k7_n17` | none | — | — |
| `metal/f32/matmul/transposed/rhs_m4_k7_n65` | none | — | — |
| `metal/f32/matmul/transposed/rhs_m64_k129_n17` | none | — | — |
| `metal/f32/matmul/transposed/rhs_m64_k129_n65` | none | — | — |
| `metal/f32/matmul/transposed/rhs_m64_k31_n17` | none | — | — |
| `metal/f32/matmul/transposed/rhs_m64_k31_n65` | none | — | — |
| `metal/f32/matmul/transposed/rhs_m64_k64_n17` | none | — | — |
| `metal/f32/matmul/transposed/rhs_m64_k64_n65` | none | — | — |
| `metal/f32/matmul/transposed/rhs_m64_k7_n17` | none | — | — |
| `metal/f32/matmul/transposed/rhs_m64_k7_n65` | none | — | — |
