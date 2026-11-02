# Parity regression check

Target: `Mac16,1/26.6.2/arm64`

- Cells compared: 24
- Added cells: 0
- Removed cells: 232
- Changed case definitions: 0
- New correctness failures: 0
- Diagnostic timing suspects: 2
- Target matches: yes
- Measurement protocol matches: yes

Timing changes are diagnostic and do not fail this check. Timing is compared only when target and protocol match. No timing baseline is accepted by this comparison.

| Cell | Correctness regression | Operator relative timing | Program relative timing |
| --- | --- | ---: | ---: |
| `metal/bf16/matmul/strided/rhs_m1_k32_n128` | none | 1.20× | 1.47× |
| `metal/bf16/matmul/strided/rhs_m1_k32_n32` | none | 1.06× | 1.16× |
| `metal/bf16/matmul/strided/rhs_m1_k32_n64` | none | 1.09× | 1.09× |
| `metal/bf16/matmul/strided/rhs_m1_k32_n8` | none | 1.08× | 0.93× |
| `metal/bf16/matmul/strided/rhs_m1_k63_n128` | none | 1.78× | 0.97× |
| `metal/bf16/matmul/strided/rhs_m1_k63_n32` | none | 0.87× | 0.78× |
| `metal/bf16/matmul/strided/rhs_m1_k63_n64` | none | 1.00× | 1.11× |
| `metal/bf16/matmul/strided/rhs_m1_k63_n8` | none | 0.86× | 1.36× |
| `metal/bf16/matmul/strided/rhs_m1_k8_n128` | none | 1.00× | 0.85× |
| `metal/bf16/matmul/strided/rhs_m1_k8_n32` | none | 0.95× | 1.04× |
| `metal/bf16/matmul/strided/rhs_m1_k8_n64` | none | 1.44× | 0.84× |
| `metal/bf16/matmul/strided/rhs_m1_k8_n8` | none | 1.05× | 1.53× |
| `metal/bf16/matmul/transposed/rhs_m1_k32_n128` | none | 1.16× | 1.04× |
| `metal/bf16/matmul/transposed/rhs_m1_k32_n32` | none | 1.39× | 0.93× |
| `metal/bf16/matmul/transposed/rhs_m1_k32_n64` | none | 0.84× | 0.97× |
| `metal/bf16/matmul/transposed/rhs_m1_k32_n8` | none | 1.09× | 1.05× |
| `metal/bf16/matmul/transposed/rhs_m1_k63_n128` | none | 0.88× | 1.27× |
| `metal/bf16/matmul/transposed/rhs_m1_k63_n32` | none | 1.00× | 1.11× |
| `metal/bf16/matmul/transposed/rhs_m1_k63_n64` | none | 1.13× | 1.38× |
| `metal/bf16/matmul/transposed/rhs_m1_k63_n8` | none | 1.12× | 1.07× |
| `metal/bf16/matmul/transposed/rhs_m1_k8_n128` | none | 1.25× | 1.08× |
| `metal/bf16/matmul/transposed/rhs_m1_k8_n32` | none | 0.87× | 1.03× |
| `metal/bf16/matmul/transposed/rhs_m1_k8_n64` | none | 0.97× | 1.07× |
| `metal/bf16/matmul/transposed/rhs_m1_k8_n8` | none | 0.94× | 1.25× |

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
