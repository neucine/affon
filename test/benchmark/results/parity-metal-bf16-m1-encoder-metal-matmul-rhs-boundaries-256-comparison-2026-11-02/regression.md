# Parity regression check

Target: `Mac16,1/26.6.2/arm64`

- Cells compared: 24
- Added cells: 0
- Removed cells: 232
- Changed case definitions: 0
- New correctness failures: 0
- Diagnostic timing suspects: 0
- Target matches: yes
- Measurement protocol matches: no

Timing changes are diagnostic and do not fail this check. Timing is compared only when target and protocol match. No timing baseline is accepted by this comparison.

| Cell | Correctness regression | Operator relative timing | Program relative timing |
| --- | --- | ---: | ---: |
| `metal/bf16/matmul/strided/rhs_m1_k32_n128` | none | — | — |
| `metal/bf16/matmul/strided/rhs_m1_k32_n32` | none | — | — |
| `metal/bf16/matmul/strided/rhs_m1_k32_n64` | none | — | — |
| `metal/bf16/matmul/strided/rhs_m1_k32_n8` | none | — | — |
| `metal/bf16/matmul/strided/rhs_m1_k63_n128` | none | — | — |
| `metal/bf16/matmul/strided/rhs_m1_k63_n32` | none | — | — |
| `metal/bf16/matmul/strided/rhs_m1_k63_n64` | none | — | — |
| `metal/bf16/matmul/strided/rhs_m1_k63_n8` | none | — | — |
| `metal/bf16/matmul/strided/rhs_m1_k8_n128` | none | — | — |
| `metal/bf16/matmul/strided/rhs_m1_k8_n32` | none | — | — |
| `metal/bf16/matmul/strided/rhs_m1_k8_n64` | none | — | — |
| `metal/bf16/matmul/strided/rhs_m1_k8_n8` | none | — | — |
| `metal/bf16/matmul/transposed/rhs_m1_k32_n128` | none | — | — |
| `metal/bf16/matmul/transposed/rhs_m1_k32_n32` | none | — | — |
| `metal/bf16/matmul/transposed/rhs_m1_k32_n64` | none | — | — |
| `metal/bf16/matmul/transposed/rhs_m1_k32_n8` | none | — | — |
| `metal/bf16/matmul/transposed/rhs_m1_k63_n128` | none | — | — |
| `metal/bf16/matmul/transposed/rhs_m1_k63_n32` | none | — | — |
| `metal/bf16/matmul/transposed/rhs_m1_k63_n64` | none | — | — |
| `metal/bf16/matmul/transposed/rhs_m1_k63_n8` | none | — | — |
| `metal/bf16/matmul/transposed/rhs_m1_k8_n128` | none | — | — |
| `metal/bf16/matmul/transposed/rhs_m1_k8_n32` | none | — | — |
| `metal/bf16/matmul/transposed/rhs_m1_k8_n64` | none | — | — |
| `metal/bf16/matmul/transposed/rhs_m1_k8_n8` | none | — | — |

## Removed cells

- `metal/bf16/matmul/strided/rhs_m1_k128_n128`
- `metal/bf16/matmul/strided/rhs_m1_k128_n32`
- `metal/bf16/matmul/strided/rhs_m1_k128_n64`
- `metal/bf16/matmul/strided/rhs_m1_k128_n8`
- `metal/bf16/matmul/strided/rhs_m33_k128_n128`
- `metal/bf16/matmul/strided/rhs_m33_k128_n32`
- `metal/bf16/matmul/strided/rhs_m33_k128_n64`
- `metal/bf16/matmul/strided/rhs_m33_k128_n8`
- `metal/bf16/matmul/strided/rhs_m33_k32_n128`
- `metal/bf16/matmul/strided/rhs_m33_k32_n32`
- `metal/bf16/matmul/strided/rhs_m33_k32_n64`
- `metal/bf16/matmul/strided/rhs_m33_k32_n8`
- `metal/bf16/matmul/strided/rhs_m33_k63_n128`
- `metal/bf16/matmul/strided/rhs_m33_k63_n32`
- `metal/bf16/matmul/strided/rhs_m33_k63_n64`
- `metal/bf16/matmul/strided/rhs_m33_k63_n8`
- `metal/bf16/matmul/strided/rhs_m33_k8_n128`
- `metal/bf16/matmul/strided/rhs_m33_k8_n32`
- `metal/bf16/matmul/strided/rhs_m33_k8_n64`
- `metal/bf16/matmul/strided/rhs_m33_k8_n8`
- `metal/bf16/matmul/strided/rhs_m65_k128_n128`
- `metal/bf16/matmul/strided/rhs_m65_k128_n32`
- `metal/bf16/matmul/strided/rhs_m65_k128_n64`
- `metal/bf16/matmul/strided/rhs_m65_k128_n8`
- `metal/bf16/matmul/strided/rhs_m65_k32_n128`
- `metal/bf16/matmul/strided/rhs_m65_k32_n32`
- `metal/bf16/matmul/strided/rhs_m65_k32_n64`
- `metal/bf16/matmul/strided/rhs_m65_k32_n8`
- `metal/bf16/matmul/strided/rhs_m65_k63_n128`
- `metal/bf16/matmul/strided/rhs_m65_k63_n32`
- `metal/bf16/matmul/strided/rhs_m65_k63_n64`
- `metal/bf16/matmul/strided/rhs_m65_k63_n8`
- `metal/bf16/matmul/strided/rhs_m65_k8_n128`
- `metal/bf16/matmul/strided/rhs_m65_k8_n32`
- `metal/bf16/matmul/strided/rhs_m65_k8_n64`
- `metal/bf16/matmul/strided/rhs_m65_k8_n8`
- `metal/bf16/matmul/strided/rhs_m7_k128_n128`
- `metal/bf16/matmul/strided/rhs_m7_k128_n32`
- `metal/bf16/matmul/strided/rhs_m7_k128_n64`
- `metal/bf16/matmul/strided/rhs_m7_k128_n8`
- `metal/bf16/matmul/strided/rhs_m7_k32_n128`
- `metal/bf16/matmul/strided/rhs_m7_k32_n32`
- `metal/bf16/matmul/strided/rhs_m7_k32_n64`
- `metal/bf16/matmul/strided/rhs_m7_k32_n8`
- `metal/bf16/matmul/strided/rhs_m7_k63_n128`
- `metal/bf16/matmul/strided/rhs_m7_k63_n32`
- `metal/bf16/matmul/strided/rhs_m7_k63_n64`
- `metal/bf16/matmul/strided/rhs_m7_k63_n8`
- `metal/bf16/matmul/strided/rhs_m7_k8_n128`
- `metal/bf16/matmul/strided/rhs_m7_k8_n32`
- `metal/bf16/matmul/strided/rhs_m7_k8_n64`
- `metal/bf16/matmul/strided/rhs_m7_k8_n8`
- `metal/bf16/matmul/transposed/rhs_m1_k128_n128`
- `metal/bf16/matmul/transposed/rhs_m1_k128_n32`
- `metal/bf16/matmul/transposed/rhs_m1_k128_n64`
- `metal/bf16/matmul/transposed/rhs_m1_k128_n8`
- `metal/bf16/matmul/transposed/rhs_m33_k128_n128`
- `metal/bf16/matmul/transposed/rhs_m33_k128_n32`
- `metal/bf16/matmul/transposed/rhs_m33_k128_n64`
- `metal/bf16/matmul/transposed/rhs_m33_k128_n8`
- `metal/bf16/matmul/transposed/rhs_m33_k32_n128`
- `metal/bf16/matmul/transposed/rhs_m33_k32_n32`
- `metal/bf16/matmul/transposed/rhs_m33_k32_n64`
- `metal/bf16/matmul/transposed/rhs_m33_k32_n8`
- `metal/bf16/matmul/transposed/rhs_m33_k63_n128`
- `metal/bf16/matmul/transposed/rhs_m33_k63_n32`
- `metal/bf16/matmul/transposed/rhs_m33_k63_n64`
- `metal/bf16/matmul/transposed/rhs_m33_k63_n8`
- `metal/bf16/matmul/transposed/rhs_m33_k8_n128`
- `metal/bf16/matmul/transposed/rhs_m33_k8_n32`
- `metal/bf16/matmul/transposed/rhs_m33_k8_n64`
- `metal/bf16/matmul/transposed/rhs_m33_k8_n8`
- `metal/bf16/matmul/transposed/rhs_m65_k128_n128`
- `metal/bf16/matmul/transposed/rhs_m65_k128_n32`
- `metal/bf16/matmul/transposed/rhs_m65_k128_n64`
- `metal/bf16/matmul/transposed/rhs_m65_k128_n8`
- `metal/bf16/matmul/transposed/rhs_m65_k32_n128`
- `metal/bf16/matmul/transposed/rhs_m65_k32_n32`
- `metal/bf16/matmul/transposed/rhs_m65_k32_n64`
- `metal/bf16/matmul/transposed/rhs_m65_k32_n8`
- `metal/bf16/matmul/transposed/rhs_m65_k63_n128`
- `metal/bf16/matmul/transposed/rhs_m65_k63_n32`
- `metal/bf16/matmul/transposed/rhs_m65_k63_n64`
- `metal/bf16/matmul/transposed/rhs_m65_k63_n8`
- `metal/bf16/matmul/transposed/rhs_m65_k8_n128`
- `metal/bf16/matmul/transposed/rhs_m65_k8_n32`
- `metal/bf16/matmul/transposed/rhs_m65_k8_n64`
- `metal/bf16/matmul/transposed/rhs_m65_k8_n8`
- `metal/bf16/matmul/transposed/rhs_m7_k128_n128`
- `metal/bf16/matmul/transposed/rhs_m7_k128_n32`
- `metal/bf16/matmul/transposed/rhs_m7_k128_n64`
- `metal/bf16/matmul/transposed/rhs_m7_k128_n8`
- `metal/bf16/matmul/transposed/rhs_m7_k32_n128`
- `metal/bf16/matmul/transposed/rhs_m7_k32_n32`
- `metal/bf16/matmul/transposed/rhs_m7_k32_n64`
- `metal/bf16/matmul/transposed/rhs_m7_k32_n8`
- `metal/bf16/matmul/transposed/rhs_m7_k63_n128`
- `metal/bf16/matmul/transposed/rhs_m7_k63_n32`
- `metal/bf16/matmul/transposed/rhs_m7_k63_n64`
- `metal/bf16/matmul/transposed/rhs_m7_k63_n8`
- `metal/bf16/matmul/transposed/rhs_m7_k8_n128`
- `metal/bf16/matmul/transposed/rhs_m7_k8_n32`
- `metal/bf16/matmul/transposed/rhs_m7_k8_n64`
- `metal/bf16/matmul/transposed/rhs_m7_k8_n8`
- `metal/f32/matmul/strided/rhs_m1_k128_n128`
- `metal/f32/matmul/strided/rhs_m1_k128_n32`
- `metal/f32/matmul/strided/rhs_m1_k128_n64`
- `metal/f32/matmul/strided/rhs_m1_k128_n8`
- `metal/f32/matmul/strided/rhs_m1_k32_n128`
- `metal/f32/matmul/strided/rhs_m1_k32_n32`
- `metal/f32/matmul/strided/rhs_m1_k32_n64`
- `metal/f32/matmul/strided/rhs_m1_k32_n8`
- `metal/f32/matmul/strided/rhs_m1_k63_n128`
- `metal/f32/matmul/strided/rhs_m1_k63_n32`
- `metal/f32/matmul/strided/rhs_m1_k63_n64`
- `metal/f32/matmul/strided/rhs_m1_k63_n8`
- `metal/f32/matmul/strided/rhs_m1_k8_n128`
- `metal/f32/matmul/strided/rhs_m1_k8_n32`
- `metal/f32/matmul/strided/rhs_m1_k8_n64`
- `metal/f32/matmul/strided/rhs_m1_k8_n8`
- `metal/f32/matmul/strided/rhs_m33_k128_n128`
- `metal/f32/matmul/strided/rhs_m33_k128_n32`
- `metal/f32/matmul/strided/rhs_m33_k128_n64`
- `metal/f32/matmul/strided/rhs_m33_k128_n8`
- `metal/f32/matmul/strided/rhs_m33_k32_n128`
- `metal/f32/matmul/strided/rhs_m33_k32_n32`
- `metal/f32/matmul/strided/rhs_m33_k32_n64`
- `metal/f32/matmul/strided/rhs_m33_k32_n8`
- `metal/f32/matmul/strided/rhs_m33_k63_n128`
- `metal/f32/matmul/strided/rhs_m33_k63_n32`
- `metal/f32/matmul/strided/rhs_m33_k63_n64`
- `metal/f32/matmul/strided/rhs_m33_k63_n8`
- `metal/f32/matmul/strided/rhs_m33_k8_n128`
- `metal/f32/matmul/strided/rhs_m33_k8_n32`
- `metal/f32/matmul/strided/rhs_m33_k8_n64`
- `metal/f32/matmul/strided/rhs_m33_k8_n8`
- `metal/f32/matmul/strided/rhs_m65_k128_n128`
- `metal/f32/matmul/strided/rhs_m65_k128_n32`
- `metal/f32/matmul/strided/rhs_m65_k128_n64`
- `metal/f32/matmul/strided/rhs_m65_k128_n8`
- `metal/f32/matmul/strided/rhs_m65_k32_n128`
- `metal/f32/matmul/strided/rhs_m65_k32_n32`
- `metal/f32/matmul/strided/rhs_m65_k32_n64`
- `metal/f32/matmul/strided/rhs_m65_k32_n8`
- `metal/f32/matmul/strided/rhs_m65_k63_n128`
- `metal/f32/matmul/strided/rhs_m65_k63_n32`
- `metal/f32/matmul/strided/rhs_m65_k63_n64`
- `metal/f32/matmul/strided/rhs_m65_k63_n8`
- `metal/f32/matmul/strided/rhs_m65_k8_n128`
- `metal/f32/matmul/strided/rhs_m65_k8_n32`
- `metal/f32/matmul/strided/rhs_m65_k8_n64`
- `metal/f32/matmul/strided/rhs_m65_k8_n8`
- `metal/f32/matmul/strided/rhs_m7_k128_n128`
- `metal/f32/matmul/strided/rhs_m7_k128_n32`
- `metal/f32/matmul/strided/rhs_m7_k128_n64`
- `metal/f32/matmul/strided/rhs_m7_k128_n8`
- `metal/f32/matmul/strided/rhs_m7_k32_n128`
- `metal/f32/matmul/strided/rhs_m7_k32_n32`
- `metal/f32/matmul/strided/rhs_m7_k32_n64`
- `metal/f32/matmul/strided/rhs_m7_k32_n8`
- `metal/f32/matmul/strided/rhs_m7_k63_n128`
- `metal/f32/matmul/strided/rhs_m7_k63_n32`
- `metal/f32/matmul/strided/rhs_m7_k63_n64`
- `metal/f32/matmul/strided/rhs_m7_k63_n8`
- `metal/f32/matmul/strided/rhs_m7_k8_n128`
- `metal/f32/matmul/strided/rhs_m7_k8_n32`
- `metal/f32/matmul/strided/rhs_m7_k8_n64`
- `metal/f32/matmul/strided/rhs_m7_k8_n8`
- `metal/f32/matmul/transposed/rhs_m1_k128_n128`
- `metal/f32/matmul/transposed/rhs_m1_k128_n32`
- `metal/f32/matmul/transposed/rhs_m1_k128_n64`
- `metal/f32/matmul/transposed/rhs_m1_k128_n8`
- `metal/f32/matmul/transposed/rhs_m1_k32_n128`
- `metal/f32/matmul/transposed/rhs_m1_k32_n32`
- `metal/f32/matmul/transposed/rhs_m1_k32_n64`
- `metal/f32/matmul/transposed/rhs_m1_k32_n8`
- `metal/f32/matmul/transposed/rhs_m1_k63_n128`
- `metal/f32/matmul/transposed/rhs_m1_k63_n32`
- `metal/f32/matmul/transposed/rhs_m1_k63_n64`
- `metal/f32/matmul/transposed/rhs_m1_k63_n8`
- `metal/f32/matmul/transposed/rhs_m1_k8_n128`
- `metal/f32/matmul/transposed/rhs_m1_k8_n32`
- `metal/f32/matmul/transposed/rhs_m1_k8_n64`
- `metal/f32/matmul/transposed/rhs_m1_k8_n8`
- `metal/f32/matmul/transposed/rhs_m33_k128_n128`
- `metal/f32/matmul/transposed/rhs_m33_k128_n32`
- `metal/f32/matmul/transposed/rhs_m33_k128_n64`
- `metal/f32/matmul/transposed/rhs_m33_k128_n8`
- `metal/f32/matmul/transposed/rhs_m33_k32_n128`
- `metal/f32/matmul/transposed/rhs_m33_k32_n32`
- `metal/f32/matmul/transposed/rhs_m33_k32_n64`
- `metal/f32/matmul/transposed/rhs_m33_k32_n8`
- `metal/f32/matmul/transposed/rhs_m33_k63_n128`
- `metal/f32/matmul/transposed/rhs_m33_k63_n32`
- `metal/f32/matmul/transposed/rhs_m33_k63_n64`
- `metal/f32/matmul/transposed/rhs_m33_k63_n8`
- `metal/f32/matmul/transposed/rhs_m33_k8_n128`
- `metal/f32/matmul/transposed/rhs_m33_k8_n32`
- `metal/f32/matmul/transposed/rhs_m33_k8_n64`
- `metal/f32/matmul/transposed/rhs_m33_k8_n8`
- `metal/f32/matmul/transposed/rhs_m65_k128_n128`
- `metal/f32/matmul/transposed/rhs_m65_k128_n32`
- `metal/f32/matmul/transposed/rhs_m65_k128_n64`
- `metal/f32/matmul/transposed/rhs_m65_k128_n8`
- `metal/f32/matmul/transposed/rhs_m65_k32_n128`
- `metal/f32/matmul/transposed/rhs_m65_k32_n32`
- `metal/f32/matmul/transposed/rhs_m65_k32_n64`
- `metal/f32/matmul/transposed/rhs_m65_k32_n8`
- `metal/f32/matmul/transposed/rhs_m65_k63_n128`
- `metal/f32/matmul/transposed/rhs_m65_k63_n32`
- `metal/f32/matmul/transposed/rhs_m65_k63_n64`
- `metal/f32/matmul/transposed/rhs_m65_k63_n8`
- `metal/f32/matmul/transposed/rhs_m65_k8_n128`
- `metal/f32/matmul/transposed/rhs_m65_k8_n32`
- `metal/f32/matmul/transposed/rhs_m65_k8_n64`
- `metal/f32/matmul/transposed/rhs_m65_k8_n8`
- `metal/f32/matmul/transposed/rhs_m7_k128_n128`
- `metal/f32/matmul/transposed/rhs_m7_k128_n32`
- `metal/f32/matmul/transposed/rhs_m7_k128_n64`
- `metal/f32/matmul/transposed/rhs_m7_k128_n8`
- `metal/f32/matmul/transposed/rhs_m7_k32_n128`
- `metal/f32/matmul/transposed/rhs_m7_k32_n32`
- `metal/f32/matmul/transposed/rhs_m7_k32_n64`
- `metal/f32/matmul/transposed/rhs_m7_k32_n8`
- `metal/f32/matmul/transposed/rhs_m7_k63_n128`
- `metal/f32/matmul/transposed/rhs_m7_k63_n32`
- `metal/f32/matmul/transposed/rhs_m7_k63_n64`
- `metal/f32/matmul/transposed/rhs_m7_k63_n8`
- `metal/f32/matmul/transposed/rhs_m7_k8_n128`
- `metal/f32/matmul/transposed/rhs_m7_k8_n32`
- `metal/f32/matmul/transposed/rhs_m7_k8_n64`
- `metal/f32/matmul/transposed/rhs_m7_k8_n8`
