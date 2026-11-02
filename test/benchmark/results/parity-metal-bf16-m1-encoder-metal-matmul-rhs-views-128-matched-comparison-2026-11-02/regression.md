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
| `metal/bf16/matmul/strided/rhs_m1_k31_n17` | none | 0.97× | 0.95× |
| `metal/bf16/matmul/strided/rhs_m1_k31_n65` | none | 0.94× | 1.01× |
| `metal/bf16/matmul/strided/rhs_m1_k64_n17` | none | 0.92× | 0.98× |
| `metal/bf16/matmul/strided/rhs_m1_k64_n65` | none | 1.28× | 1.03× |
| `metal/bf16/matmul/strided/rhs_m1_k7_n17` | none | 0.97× | 0.89× |
| `metal/bf16/matmul/strided/rhs_m1_k7_n65` | none | 0.95× | 1.05× |
| `metal/bf16/matmul/transposed/rhs_m1_k31_n17` | none | 0.99× | 0.90× |
| `metal/bf16/matmul/transposed/rhs_m1_k31_n65` | none | 0.75× | 0.91× |
| `metal/bf16/matmul/transposed/rhs_m1_k64_n17` | none | 1.00× | 1.02× |
| `metal/bf16/matmul/transposed/rhs_m1_k64_n65` | none | 0.94× | 0.84× |
| `metal/bf16/matmul/transposed/rhs_m1_k7_n17` | none | 1.07× | 0.93× |
| `metal/bf16/matmul/transposed/rhs_m1_k7_n65` | none | 0.88× | 0.99× |

## Removed cells

- `metal/bf16/matmul/strided/rhs_m16_k129_n17`
- `metal/bf16/matmul/strided/rhs_m16_k129_n65`
- `metal/bf16/matmul/strided/rhs_m16_k31_n17`
- `metal/bf16/matmul/strided/rhs_m16_k31_n65`
- `metal/bf16/matmul/strided/rhs_m16_k64_n17`
- `metal/bf16/matmul/strided/rhs_m16_k64_n65`
- `metal/bf16/matmul/strided/rhs_m16_k7_n17`
- `metal/bf16/matmul/strided/rhs_m16_k7_n65`
- `metal/bf16/matmul/strided/rhs_m1_k129_n17`
- `metal/bf16/matmul/strided/rhs_m1_k129_n65`
- `metal/bf16/matmul/strided/rhs_m4_k129_n17`
- `metal/bf16/matmul/strided/rhs_m4_k129_n65`
- `metal/bf16/matmul/strided/rhs_m4_k31_n17`
- `metal/bf16/matmul/strided/rhs_m4_k31_n65`
- `metal/bf16/matmul/strided/rhs_m4_k64_n17`
- `metal/bf16/matmul/strided/rhs_m4_k64_n65`
- `metal/bf16/matmul/strided/rhs_m4_k7_n17`
- `metal/bf16/matmul/strided/rhs_m4_k7_n65`
- `metal/bf16/matmul/strided/rhs_m64_k129_n17`
- `metal/bf16/matmul/strided/rhs_m64_k129_n65`
- `metal/bf16/matmul/strided/rhs_m64_k31_n17`
- `metal/bf16/matmul/strided/rhs_m64_k31_n65`
- `metal/bf16/matmul/strided/rhs_m64_k64_n17`
- `metal/bf16/matmul/strided/rhs_m64_k64_n65`
- `metal/bf16/matmul/strided/rhs_m64_k7_n17`
- `metal/bf16/matmul/strided/rhs_m64_k7_n65`
- `metal/bf16/matmul/transposed/rhs_m16_k129_n17`
- `metal/bf16/matmul/transposed/rhs_m16_k129_n65`
- `metal/bf16/matmul/transposed/rhs_m16_k31_n17`
- `metal/bf16/matmul/transposed/rhs_m16_k31_n65`
- `metal/bf16/matmul/transposed/rhs_m16_k64_n17`
- `metal/bf16/matmul/transposed/rhs_m16_k64_n65`
- `metal/bf16/matmul/transposed/rhs_m16_k7_n17`
- `metal/bf16/matmul/transposed/rhs_m16_k7_n65`
- `metal/bf16/matmul/transposed/rhs_m1_k129_n17`
- `metal/bf16/matmul/transposed/rhs_m1_k129_n65`
- `metal/bf16/matmul/transposed/rhs_m4_k129_n17`
- `metal/bf16/matmul/transposed/rhs_m4_k129_n65`
- `metal/bf16/matmul/transposed/rhs_m4_k31_n17`
- `metal/bf16/matmul/transposed/rhs_m4_k31_n65`
- `metal/bf16/matmul/transposed/rhs_m4_k64_n17`
- `metal/bf16/matmul/transposed/rhs_m4_k64_n65`
- `metal/bf16/matmul/transposed/rhs_m4_k7_n17`
- `metal/bf16/matmul/transposed/rhs_m4_k7_n65`
- `metal/bf16/matmul/transposed/rhs_m64_k129_n17`
- `metal/bf16/matmul/transposed/rhs_m64_k129_n65`
- `metal/bf16/matmul/transposed/rhs_m64_k31_n17`
- `metal/bf16/matmul/transposed/rhs_m64_k31_n65`
- `metal/bf16/matmul/transposed/rhs_m64_k64_n17`
- `metal/bf16/matmul/transposed/rhs_m64_k64_n65`
- `metal/bf16/matmul/transposed/rhs_m64_k7_n17`
- `metal/bf16/matmul/transposed/rhs_m64_k7_n65`
- `metal/f32/matmul/strided/rhs_m16_k129_n17`
- `metal/f32/matmul/strided/rhs_m16_k129_n65`
- `metal/f32/matmul/strided/rhs_m16_k31_n17`
- `metal/f32/matmul/strided/rhs_m16_k31_n65`
- `metal/f32/matmul/strided/rhs_m16_k64_n17`
- `metal/f32/matmul/strided/rhs_m16_k64_n65`
- `metal/f32/matmul/strided/rhs_m16_k7_n17`
- `metal/f32/matmul/strided/rhs_m16_k7_n65`
- `metal/f32/matmul/strided/rhs_m1_k129_n17`
- `metal/f32/matmul/strided/rhs_m1_k129_n65`
- `metal/f32/matmul/strided/rhs_m1_k31_n17`
- `metal/f32/matmul/strided/rhs_m1_k31_n65`
- `metal/f32/matmul/strided/rhs_m1_k64_n17`
- `metal/f32/matmul/strided/rhs_m1_k64_n65`
- `metal/f32/matmul/strided/rhs_m1_k7_n17`
- `metal/f32/matmul/strided/rhs_m1_k7_n65`
- `metal/f32/matmul/strided/rhs_m4_k129_n17`
- `metal/f32/matmul/strided/rhs_m4_k129_n65`
- `metal/f32/matmul/strided/rhs_m4_k31_n17`
- `metal/f32/matmul/strided/rhs_m4_k31_n65`
- `metal/f32/matmul/strided/rhs_m4_k64_n17`
- `metal/f32/matmul/strided/rhs_m4_k64_n65`
- `metal/f32/matmul/strided/rhs_m4_k7_n17`
- `metal/f32/matmul/strided/rhs_m4_k7_n65`
- `metal/f32/matmul/strided/rhs_m64_k129_n17`
- `metal/f32/matmul/strided/rhs_m64_k129_n65`
- `metal/f32/matmul/strided/rhs_m64_k31_n17`
- `metal/f32/matmul/strided/rhs_m64_k31_n65`
- `metal/f32/matmul/strided/rhs_m64_k64_n17`
- `metal/f32/matmul/strided/rhs_m64_k64_n65`
- `metal/f32/matmul/strided/rhs_m64_k7_n17`
- `metal/f32/matmul/strided/rhs_m64_k7_n65`
- `metal/f32/matmul/transposed/rhs_m16_k129_n17`
- `metal/f32/matmul/transposed/rhs_m16_k129_n65`
- `metal/f32/matmul/transposed/rhs_m16_k31_n17`
- `metal/f32/matmul/transposed/rhs_m16_k31_n65`
- `metal/f32/matmul/transposed/rhs_m16_k64_n17`
- `metal/f32/matmul/transposed/rhs_m16_k64_n65`
- `metal/f32/matmul/transposed/rhs_m16_k7_n17`
- `metal/f32/matmul/transposed/rhs_m16_k7_n65`
- `metal/f32/matmul/transposed/rhs_m1_k129_n17`
- `metal/f32/matmul/transposed/rhs_m1_k129_n65`
- `metal/f32/matmul/transposed/rhs_m1_k31_n17`
- `metal/f32/matmul/transposed/rhs_m1_k31_n65`
- `metal/f32/matmul/transposed/rhs_m1_k64_n17`
- `metal/f32/matmul/transposed/rhs_m1_k64_n65`
- `metal/f32/matmul/transposed/rhs_m1_k7_n17`
- `metal/f32/matmul/transposed/rhs_m1_k7_n65`
- `metal/f32/matmul/transposed/rhs_m4_k129_n17`
- `metal/f32/matmul/transposed/rhs_m4_k129_n65`
- `metal/f32/matmul/transposed/rhs_m4_k31_n17`
- `metal/f32/matmul/transposed/rhs_m4_k31_n65`
- `metal/f32/matmul/transposed/rhs_m4_k64_n17`
- `metal/f32/matmul/transposed/rhs_m4_k64_n65`
- `metal/f32/matmul/transposed/rhs_m4_k7_n17`
- `metal/f32/matmul/transposed/rhs_m4_k7_n65`
- `metal/f32/matmul/transposed/rhs_m64_k129_n17`
- `metal/f32/matmul/transposed/rhs_m64_k129_n65`
- `metal/f32/matmul/transposed/rhs_m64_k31_n17`
- `metal/f32/matmul/transposed/rhs_m64_k31_n65`
- `metal/f32/matmul/transposed/rhs_m64_k64_n17`
- `metal/f32/matmul/transposed/rhs_m64_k64_n65`
- `metal/f32/matmul/transposed/rhs_m64_k7_n17`
- `metal/f32/matmul/transposed/rhs_m64_k7_n65`
