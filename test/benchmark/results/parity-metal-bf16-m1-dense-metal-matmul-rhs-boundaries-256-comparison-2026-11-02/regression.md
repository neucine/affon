# Parity regression check

Target: `Mac16,1/26.6.2/arm64`

- Cells compared: 24
- Added cells: 0
- Removed cells: 232
- Changed case definitions: 0
- New correctness failures: 0
- Diagnostic timing suspects: 3
- Target matches: yes
- Measurement protocol matches: yes

Timing changes are diagnostic and do not fail this check. Timing is compared only when target and protocol match. No timing baseline is accepted by this comparison.

| Cell | Correctness regression | Operator relative timing | Program relative timing |
| --- | --- | ---: | ---: |
| `metal/bf16/matmul/strided/rhs_m1_k32_n128` | none | 0.90× | 1.32× |
| `metal/bf16/matmul/strided/rhs_m1_k32_n32` | none | 0.94× | 1.06× |
| `metal/bf16/matmul/strided/rhs_m1_k32_n64` | none | 1.04× | 0.94× |
| `metal/bf16/matmul/strided/rhs_m1_k32_n8` | none | 0.96× | 1.24× |
| `metal/bf16/matmul/strided/rhs_m1_k63_n128` | none | 1.85× | 1.63× |
| `metal/bf16/matmul/strided/rhs_m1_k63_n32` | none | 1.09× | 0.67× |
| `metal/bf16/matmul/strided/rhs_m1_k63_n64` | none | 0.87× | 1.16× |
| `metal/bf16/matmul/strided/rhs_m1_k63_n8` | none | 1.04× | 1.39× |
| `metal/bf16/matmul/strided/rhs_m1_k8_n128` | none | 1.04× | 1.23× |
| `metal/bf16/matmul/strided/rhs_m1_k8_n32` | none | 0.99× | 0.89× |
| `metal/bf16/matmul/strided/rhs_m1_k8_n64` | none | 1.66× | 0.87× |
| `metal/bf16/matmul/strided/rhs_m1_k8_n8` | none | 1.05× | 1.40× |
| `metal/bf16/matmul/transposed/rhs_m1_k32_n128` | none | 1.21× | 0.99× |
| `metal/bf16/matmul/transposed/rhs_m1_k32_n32` | none | 1.68× | 1.06× |
| `metal/bf16/matmul/transposed/rhs_m1_k32_n64` | none | 1.00× | 1.24× |
| `metal/bf16/matmul/transposed/rhs_m1_k32_n8` | none | 0.92× | 1.24× |
| `metal/bf16/matmul/transposed/rhs_m1_k63_n128` | none | 0.68× | 1.37× |
| `metal/bf16/matmul/transposed/rhs_m1_k63_n32` | none | 1.02× | 1.05× |
| `metal/bf16/matmul/transposed/rhs_m1_k63_n64` | none | 1.14× | 1.48× |
| `metal/bf16/matmul/transposed/rhs_m1_k63_n8` | none | 1.29× | 1.38× |
| `metal/bf16/matmul/transposed/rhs_m1_k8_n128` | none | 1.37× | 0.89× |
| `metal/bf16/matmul/transposed/rhs_m1_k8_n32` | none | 1.11× | 1.09× |
| `metal/bf16/matmul/transposed/rhs_m1_k8_n64` | none | 0.93× | 1.12× |
| `metal/bf16/matmul/transposed/rhs_m1_k8_n8` | none | 1.02× | 1.12× |

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
