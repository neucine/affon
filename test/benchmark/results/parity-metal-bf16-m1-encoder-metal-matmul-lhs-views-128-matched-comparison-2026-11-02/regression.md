# Parity regression check

Target: `Mac16,1/26.6.2/arm64`

- Cells compared: 12
- Added cells: 0
- Removed cells: 116
- Changed case definitions: 0
- New correctness failures: 0
- Diagnostic timing suspects: 0
- Target matches: yes
- Measurement protocol matches: yes

Timing changes are diagnostic and do not fail this check. Timing is compared only when target and protocol match. No timing baseline is accepted by this comparison.

| Cell | Correctness regression | Operator relative timing | Program relative timing |
| --- | --- | ---: | ---: |
| `metal/bf16/matmul/strided/lhs_m1_k31_n17` | none | 1.01× | 1.00× |
| `metal/bf16/matmul/strided/lhs_m1_k31_n65` | none | 1.09× | 0.96× |
| `metal/bf16/matmul/strided/lhs_m1_k64_n17` | none | 1.07× | 1.09× |
| `metal/bf16/matmul/strided/lhs_m1_k64_n65` | none | 1.02× | 1.24× |
| `metal/bf16/matmul/strided/lhs_m1_k7_n17` | none | 1.04× | 1.05× |
| `metal/bf16/matmul/strided/lhs_m1_k7_n65` | none | 1.00× | 0.98× |
| `metal/bf16/matmul/transposed/lhs_m1_k31_n17` | none | 0.92× | 0.96× |
| `metal/bf16/matmul/transposed/lhs_m1_k31_n65` | none | 0.95× | 0.95× |
| `metal/bf16/matmul/transposed/lhs_m1_k64_n17` | none | 1.02× | 0.77× |
| `metal/bf16/matmul/transposed/lhs_m1_k64_n65` | none | 1.02× | 1.02× |
| `metal/bf16/matmul/transposed/lhs_m1_k7_n17` | none | 0.98× | 0.93× |
| `metal/bf16/matmul/transposed/lhs_m1_k7_n65` | none | 1.02× | 1.03× |

## Removed cells

- `metal/bf16/matmul/strided/lhs_m16_k129_n17`
- `metal/bf16/matmul/strided/lhs_m16_k129_n65`
- `metal/bf16/matmul/strided/lhs_m16_k31_n17`
- `metal/bf16/matmul/strided/lhs_m16_k31_n65`
- `metal/bf16/matmul/strided/lhs_m16_k64_n17`
- `metal/bf16/matmul/strided/lhs_m16_k64_n65`
- `metal/bf16/matmul/strided/lhs_m16_k7_n17`
- `metal/bf16/matmul/strided/lhs_m16_k7_n65`
- `metal/bf16/matmul/strided/lhs_m1_k129_n17`
- `metal/bf16/matmul/strided/lhs_m1_k129_n65`
- `metal/bf16/matmul/strided/lhs_m4_k129_n17`
- `metal/bf16/matmul/strided/lhs_m4_k129_n65`
- `metal/bf16/matmul/strided/lhs_m4_k31_n17`
- `metal/bf16/matmul/strided/lhs_m4_k31_n65`
- `metal/bf16/matmul/strided/lhs_m4_k64_n17`
- `metal/bf16/matmul/strided/lhs_m4_k64_n65`
- `metal/bf16/matmul/strided/lhs_m4_k7_n17`
- `metal/bf16/matmul/strided/lhs_m4_k7_n65`
- `metal/bf16/matmul/strided/lhs_m64_k129_n17`
- `metal/bf16/matmul/strided/lhs_m64_k129_n65`
- `metal/bf16/matmul/strided/lhs_m64_k31_n17`
- `metal/bf16/matmul/strided/lhs_m64_k31_n65`
- `metal/bf16/matmul/strided/lhs_m64_k64_n17`
- `metal/bf16/matmul/strided/lhs_m64_k64_n65`
- `metal/bf16/matmul/strided/lhs_m64_k7_n17`
- `metal/bf16/matmul/strided/lhs_m64_k7_n65`
- `metal/bf16/matmul/transposed/lhs_m16_k129_n17`
- `metal/bf16/matmul/transposed/lhs_m16_k129_n65`
- `metal/bf16/matmul/transposed/lhs_m16_k31_n17`
- `metal/bf16/matmul/transposed/lhs_m16_k31_n65`
- `metal/bf16/matmul/transposed/lhs_m16_k64_n17`
- `metal/bf16/matmul/transposed/lhs_m16_k64_n65`
- `metal/bf16/matmul/transposed/lhs_m16_k7_n17`
- `metal/bf16/matmul/transposed/lhs_m16_k7_n65`
- `metal/bf16/matmul/transposed/lhs_m1_k129_n17`
- `metal/bf16/matmul/transposed/lhs_m1_k129_n65`
- `metal/bf16/matmul/transposed/lhs_m4_k129_n17`
- `metal/bf16/matmul/transposed/lhs_m4_k129_n65`
- `metal/bf16/matmul/transposed/lhs_m4_k31_n17`
- `metal/bf16/matmul/transposed/lhs_m4_k31_n65`
- `metal/bf16/matmul/transposed/lhs_m4_k64_n17`
- `metal/bf16/matmul/transposed/lhs_m4_k64_n65`
- `metal/bf16/matmul/transposed/lhs_m4_k7_n17`
- `metal/bf16/matmul/transposed/lhs_m4_k7_n65`
- `metal/bf16/matmul/transposed/lhs_m64_k129_n17`
- `metal/bf16/matmul/transposed/lhs_m64_k129_n65`
- `metal/bf16/matmul/transposed/lhs_m64_k31_n17`
- `metal/bf16/matmul/transposed/lhs_m64_k31_n65`
- `metal/bf16/matmul/transposed/lhs_m64_k64_n17`
- `metal/bf16/matmul/transposed/lhs_m64_k64_n65`
- `metal/bf16/matmul/transposed/lhs_m64_k7_n17`
- `metal/bf16/matmul/transposed/lhs_m64_k7_n65`
- `metal/f32/matmul/strided/lhs_m16_k129_n17`
- `metal/f32/matmul/strided/lhs_m16_k129_n65`
- `metal/f32/matmul/strided/lhs_m16_k31_n17`
- `metal/f32/matmul/strided/lhs_m16_k31_n65`
- `metal/f32/matmul/strided/lhs_m16_k64_n17`
- `metal/f32/matmul/strided/lhs_m16_k64_n65`
- `metal/f32/matmul/strided/lhs_m16_k7_n17`
- `metal/f32/matmul/strided/lhs_m16_k7_n65`
- `metal/f32/matmul/strided/lhs_m1_k129_n17`
- `metal/f32/matmul/strided/lhs_m1_k129_n65`
- `metal/f32/matmul/strided/lhs_m1_k31_n17`
- `metal/f32/matmul/strided/lhs_m1_k31_n65`
- `metal/f32/matmul/strided/lhs_m1_k64_n17`
- `metal/f32/matmul/strided/lhs_m1_k64_n65`
- `metal/f32/matmul/strided/lhs_m1_k7_n17`
- `metal/f32/matmul/strided/lhs_m1_k7_n65`
- `metal/f32/matmul/strided/lhs_m4_k129_n17`
- `metal/f32/matmul/strided/lhs_m4_k129_n65`
- `metal/f32/matmul/strided/lhs_m4_k31_n17`
- `metal/f32/matmul/strided/lhs_m4_k31_n65`
- `metal/f32/matmul/strided/lhs_m4_k64_n17`
- `metal/f32/matmul/strided/lhs_m4_k64_n65`
- `metal/f32/matmul/strided/lhs_m4_k7_n17`
- `metal/f32/matmul/strided/lhs_m4_k7_n65`
- `metal/f32/matmul/strided/lhs_m64_k129_n17`
- `metal/f32/matmul/strided/lhs_m64_k129_n65`
- `metal/f32/matmul/strided/lhs_m64_k31_n17`
- `metal/f32/matmul/strided/lhs_m64_k31_n65`
- `metal/f32/matmul/strided/lhs_m64_k64_n17`
- `metal/f32/matmul/strided/lhs_m64_k64_n65`
- `metal/f32/matmul/strided/lhs_m64_k7_n17`
- `metal/f32/matmul/strided/lhs_m64_k7_n65`
- `metal/f32/matmul/transposed/lhs_m16_k129_n17`
- `metal/f32/matmul/transposed/lhs_m16_k129_n65`
- `metal/f32/matmul/transposed/lhs_m16_k31_n17`
- `metal/f32/matmul/transposed/lhs_m16_k31_n65`
- `metal/f32/matmul/transposed/lhs_m16_k64_n17`
- `metal/f32/matmul/transposed/lhs_m16_k64_n65`
- `metal/f32/matmul/transposed/lhs_m16_k7_n17`
- `metal/f32/matmul/transposed/lhs_m16_k7_n65`
- `metal/f32/matmul/transposed/lhs_m1_k129_n17`
- `metal/f32/matmul/transposed/lhs_m1_k129_n65`
- `metal/f32/matmul/transposed/lhs_m1_k31_n17`
- `metal/f32/matmul/transposed/lhs_m1_k31_n65`
- `metal/f32/matmul/transposed/lhs_m1_k64_n17`
- `metal/f32/matmul/transposed/lhs_m1_k64_n65`
- `metal/f32/matmul/transposed/lhs_m1_k7_n17`
- `metal/f32/matmul/transposed/lhs_m1_k7_n65`
- `metal/f32/matmul/transposed/lhs_m4_k129_n17`
- `metal/f32/matmul/transposed/lhs_m4_k129_n65`
- `metal/f32/matmul/transposed/lhs_m4_k31_n17`
- `metal/f32/matmul/transposed/lhs_m4_k31_n65`
- `metal/f32/matmul/transposed/lhs_m4_k64_n17`
- `metal/f32/matmul/transposed/lhs_m4_k64_n65`
- `metal/f32/matmul/transposed/lhs_m4_k7_n17`
- `metal/f32/matmul/transposed/lhs_m4_k7_n65`
- `metal/f32/matmul/transposed/lhs_m64_k129_n17`
- `metal/f32/matmul/transposed/lhs_m64_k129_n65`
- `metal/f32/matmul/transposed/lhs_m64_k31_n17`
- `metal/f32/matmul/transposed/lhs_m64_k31_n65`
- `metal/f32/matmul/transposed/lhs_m64_k64_n17`
- `metal/f32/matmul/transposed/lhs_m64_k64_n65`
- `metal/f32/matmul/transposed/lhs_m64_k7_n17`
- `metal/f32/matmul/transposed/lhs_m64_k7_n65`
