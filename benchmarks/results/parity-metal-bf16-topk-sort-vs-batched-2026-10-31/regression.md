# Parity regression check

Target: `Mac16,1/26.6.2/arm64`

- Cells compared: 24
- Added cells: 61
- Removed cells: 106
- Changed case definitions: 0
- New correctness failures: 0
- Diagnostic timing suspects: 0
- Target matches: yes
- Measurement protocol matches: yes

Timing changes are diagnostic and do not fail this check. Timing is compared only when target and protocol match. No timing baseline is accepted by this comparison.

| Cell | Correctness regression | Operator relative timing | Program relative timing |
| --- | --- | ---: | ---: |
| `metal/bf16/topk/batched/dense_axis1_b2_r33_c129_k16` | none | 1.10× | 0.75× |
| `metal/bf16/topk/batched/dense_axis1_b2_r33_c17_k16` | none | 1.17× | 0.70× |
| `metal/bf16/topk/batched/dense_axis1_b2_r33_c5_k16` | none | 0.62× | 1.21× |
| `metal/bf16/topk/batched/dense_axis1_b2_r33_c65_k16` | none | 0.88× | 0.83× |
| `metal/bf16/topk/batched/dense_axis1_b4_r33_c129_k16` | none | 0.74× | 0.72× |
| `metal/bf16/topk/batched/dense_axis1_b4_r33_c17_k16` | none | 1.10× | 0.99× |
| `metal/bf16/topk/batched/dense_axis1_b4_r33_c5_k16` | none | 1.14× | 0.80× |
| `metal/bf16/topk/batched/dense_axis1_b4_r33_c65_k16` | none | 1.32× | 0.76× |
| `metal/bf16/topk/batched/dense_axis2_b2_r16_c129_k64` | none | 0.96× | 0.75× |
| `metal/bf16/topk/batched/dense_axis2_b2_r16_c65_k32` | none | 0.99× | 1.11× |
| `metal/bf16/topk/batched/dense_axis2_b2_r33_c129_k64` | none | 0.71× | 0.86× |
| `metal/bf16/topk/batched/dense_axis2_b2_r33_c65_k32` | none | 1.16× | 1.05× |
| `metal/bf16/topk/batched/dense_axis2_b2_r3_c129_k64` | none | 0.92× | 0.65× |
| `metal/bf16/topk/batched/dense_axis2_b2_r3_c65_k32` | none | 0.79× | 1.08× |
| `metal/bf16/topk/batched/dense_axis2_b2_r7_c129_k64` | none | 0.70× | 0.97× |
| `metal/bf16/topk/batched/dense_axis2_b2_r7_c65_k32` | none | 1.13× | 1.12× |
| `metal/bf16/topk/batched/dense_axis2_b4_r16_c129_k64` | none | 0.76× | 0.70× |
| `metal/bf16/topk/batched/dense_axis2_b4_r16_c65_k32` | none | 1.18× | 1.47× |
| `metal/bf16/topk/batched/dense_axis2_b4_r33_c129_k64` | none | 0.57× | 0.52× |
| `metal/bf16/topk/batched/dense_axis2_b4_r33_c65_k32` | none | 1.04× | 0.72× |
| `metal/bf16/topk/batched/dense_axis2_b4_r3_c129_k64` | none | 0.73× | 0.90× |
| `metal/bf16/topk/batched/dense_axis2_b4_r3_c65_k32` | none | 1.05× | 1.10× |
| `metal/bf16/topk/batched/dense_axis2_b4_r7_c129_k64` | none | 0.94× | 0.87× |
| `metal/bf16/topk/batched/dense_axis2_b4_r7_c65_k32` | none | 0.99× | 1.19× |

## Added cells

- `metal/bf16/topk/large/dense_axis1_r128_c257_k256_smallest`
- `metal/bf16/topk/large/dense_axis1_r128_c65_k64_smallest`
- `metal/bf16/topk/large/dense_axis1_r32_c257_k256_smallest`
- `metal/bf16/topk/large/dense_axis1_r32_c65_k64_smallest`
- `metal/bf16/topk/large/dense_axis1_r64_c257_k256_smallest`
- `metal/bf16/topk/large/dense_axis1_r64_c65_k64_smallest`
- `metal/bf16/topk/large/dense_axis1_r8_c257_k256_smallest`
- `metal/bf16/topk/large/dense_axis1_r8_c65_k64_smallest`
- `metal/bf16/topk/large/offset_axis1_r128_c257_k256_smallest`
- `metal/bf16/topk/large/offset_axis1_r128_c65_k64_smallest`
- `metal/bf16/topk/large/offset_axis1_r32_c257_k256_smallest`
- `metal/bf16/topk/large/offset_axis1_r32_c65_k64_smallest`
- `metal/bf16/topk/large/offset_axis1_r64_c257_k256_smallest`
- `metal/bf16/topk/large/offset_axis1_r64_c65_k64_smallest`
- `metal/bf16/topk/large/offset_axis1_r8_c257_k256_smallest`
- `metal/bf16/topk/large/offset_axis1_r8_c65_k64_smallest`
- `metal/bf16/topk/large/strided_axis1_r128_c257_k256_smallest`
- `metal/bf16/topk/large/strided_axis1_r128_c65_k64_smallest`
- `metal/bf16/topk/large/strided_axis1_r32_c257_k256_smallest`
- `metal/bf16/topk/large/strided_axis1_r32_c65_k64_smallest`
- `metal/bf16/topk/large/strided_axis1_r64_c257_k256_smallest`
- `metal/bf16/topk/large/strided_axis1_r64_c65_k64_smallest`
- `metal/bf16/topk/large/strided_axis1_r8_c257_k256_smallest`
- `metal/bf16/topk/large/strided_axis1_r8_c65_k64_smallest`
- `metal/bf16/topk/small/dense_axis0_r33_c129_k17`
- `metal/bf16/topk/small/dense_axis0_r33_c17_k17`
- `metal/bf16/topk/small/dense_axis0_r33_c3_k17`
- `metal/bf16/topk/small/dense_axis0_r33_c64_k17`
- `metal/bf16/topk/small/dense_axis1_r16_c129_k65`
- `metal/bf16/topk/small/dense_axis1_r16_c64_k33`
- `metal/bf16/topk/small/dense_axis1_r33_c129_k65`
- `metal/bf16/topk/small/dense_axis1_r33_c64_k33`
- `metal/bf16/topk/small/dense_axis1_r3_c129_k65`
- `metal/bf16/topk/small/dense_axis1_r3_c64_k33`
- `metal/bf16/topk/small/dense_axis1_r7_c129_k65`
- `metal/bf16/topk/small/dense_axis1_r7_c64_k33`
- `metal/bf16/topk/small/offset_axis0_r33_c129_k17`
- `metal/bf16/topk/small/offset_axis0_r33_c17_k17`
- `metal/bf16/topk/small/offset_axis0_r33_c3_k17`
- `metal/bf16/topk/small/offset_axis0_r33_c64_k17`
- `metal/bf16/topk/small/offset_axis1_r16_c129_k65`
- `metal/bf16/topk/small/offset_axis1_r16_c64_k33`
- `metal/bf16/topk/small/offset_axis1_r33_c129_k65`
- `metal/bf16/topk/small/offset_axis1_r33_c64_k33`
- `metal/bf16/topk/small/offset_axis1_r3_c129_k65`
- `metal/bf16/topk/small/offset_axis1_r3_c64_k33`
- `metal/bf16/topk/small/offset_axis1_r7_c129_k65`
- `metal/bf16/topk/small/offset_axis1_r7_c64_k33`
- `metal/bf16/topk/small/strided_axis0_r33_c129_k17`
- `metal/bf16/topk/small/strided_axis0_r33_c17_k17`
- `metal/bf16/topk/small/strided_axis0_r33_c3_k17`
- `metal/bf16/topk/small/strided_axis0_r33_c64_k17`
- `metal/bf16/topk/small/strided_axis1_r16_c129_k65`
- `metal/bf16/topk/small/strided_axis1_r16_c64_k33`
- `metal/bf16/topk/small/strided_axis1_r33_c129_k65`
- `metal/bf16/topk/small/strided_axis1_r33_c64_k33`
- `metal/bf16/topk/small/strided_axis1_r3_c129_k65`
- `metal/bf16/topk/small/strided_axis1_r3_c64_k33`
- `metal/bf16/topk/small/strided_axis1_r7_c129_k65`
- `metal/bf16/topk/small/strided_axis1_r7_c64_k33`
- `metal/bf16/topk/strided`

## Removed cells

- `metal/bf16/topk/batched`
- `metal/bf16/topk/batched/dense_axis1_b2_r16_c129_k8`
- `metal/bf16/topk/batched/dense_axis1_b2_r16_c17_k8`
- `metal/bf16/topk/batched/dense_axis1_b2_r16_c5_k8`
- `metal/bf16/topk/batched/dense_axis1_b2_r16_c65_k8`
- `metal/bf16/topk/batched/dense_axis1_b2_r3_c129_k1`
- `metal/bf16/topk/batched/dense_axis1_b2_r3_c17_k1`
- `metal/bf16/topk/batched/dense_axis1_b2_r3_c5_k1`
- `metal/bf16/topk/batched/dense_axis1_b2_r3_c65_k1`
- `metal/bf16/topk/batched/dense_axis1_b2_r7_c129_k3`
- `metal/bf16/topk/batched/dense_axis1_b2_r7_c17_k3`
- `metal/bf16/topk/batched/dense_axis1_b2_r7_c5_k3`
- `metal/bf16/topk/batched/dense_axis1_b2_r7_c65_k3`
- `metal/bf16/topk/batched/dense_axis1_b4_r16_c129_k8`
- `metal/bf16/topk/batched/dense_axis1_b4_r16_c17_k8`
- `metal/bf16/topk/batched/dense_axis1_b4_r16_c5_k8`
- `metal/bf16/topk/batched/dense_axis1_b4_r16_c65_k8`
- `metal/bf16/topk/batched/dense_axis1_b4_r3_c129_k1`
- `metal/bf16/topk/batched/dense_axis1_b4_r3_c17_k1`
- `metal/bf16/topk/batched/dense_axis1_b4_r3_c5_k1`
- `metal/bf16/topk/batched/dense_axis1_b4_r3_c65_k1`
- `metal/bf16/topk/batched/dense_axis1_b4_r7_c129_k3`
- `metal/bf16/topk/batched/dense_axis1_b4_r7_c17_k3`
- `metal/bf16/topk/batched/dense_axis1_b4_r7_c5_k3`
- `metal/bf16/topk/batched/dense_axis1_b4_r7_c65_k3`
- `metal/bf16/topk/batched/dense_axis2_b2_r16_c17_k8`
- `metal/bf16/topk/batched/dense_axis2_b2_r16_c5_k2`
- `metal/bf16/topk/batched/dense_axis2_b2_r33_c17_k8`
- `metal/bf16/topk/batched/dense_axis2_b2_r33_c5_k2`
- `metal/bf16/topk/batched/dense_axis2_b2_r3_c17_k8`
- `metal/bf16/topk/batched/dense_axis2_b2_r3_c5_k2`
- `metal/bf16/topk/batched/dense_axis2_b2_r7_c17_k8`
- `metal/bf16/topk/batched/dense_axis2_b2_r7_c5_k2`
- `metal/bf16/topk/batched/dense_axis2_b4_r16_c17_k8`
- `metal/bf16/topk/batched/dense_axis2_b4_r16_c5_k2`
- `metal/bf16/topk/batched/dense_axis2_b4_r33_c17_k8`
- `metal/bf16/topk/batched/dense_axis2_b4_r33_c5_k2`
- `metal/bf16/topk/batched/dense_axis2_b4_r3_c17_k8`
- `metal/bf16/topk/batched/dense_axis2_b4_r3_c5_k2`
- `metal/bf16/topk/batched/dense_axis2_b4_r7_c17_k8`
- `metal/bf16/topk/batched/dense_axis2_b4_r7_c5_k2`
- `metal/f32/topk/batched`
- `metal/f32/topk/batched/dense_axis1_b2_r16_c129_k8`
- `metal/f32/topk/batched/dense_axis1_b2_r16_c17_k8`
- `metal/f32/topk/batched/dense_axis1_b2_r16_c5_k8`
- `metal/f32/topk/batched/dense_axis1_b2_r16_c65_k8`
- `metal/f32/topk/batched/dense_axis1_b2_r33_c129_k16`
- `metal/f32/topk/batched/dense_axis1_b2_r33_c17_k16`
- `metal/f32/topk/batched/dense_axis1_b2_r33_c5_k16`
- `metal/f32/topk/batched/dense_axis1_b2_r33_c65_k16`
- `metal/f32/topk/batched/dense_axis1_b2_r3_c129_k1`
- `metal/f32/topk/batched/dense_axis1_b2_r3_c17_k1`
- `metal/f32/topk/batched/dense_axis1_b2_r3_c5_k1`
- `metal/f32/topk/batched/dense_axis1_b2_r3_c65_k1`
- `metal/f32/topk/batched/dense_axis1_b2_r7_c129_k3`
- `metal/f32/topk/batched/dense_axis1_b2_r7_c17_k3`
- `metal/f32/topk/batched/dense_axis1_b2_r7_c5_k3`
- `metal/f32/topk/batched/dense_axis1_b2_r7_c65_k3`
- `metal/f32/topk/batched/dense_axis1_b4_r16_c129_k8`
- `metal/f32/topk/batched/dense_axis1_b4_r16_c17_k8`
- `metal/f32/topk/batched/dense_axis1_b4_r16_c5_k8`
- `metal/f32/topk/batched/dense_axis1_b4_r16_c65_k8`
- `metal/f32/topk/batched/dense_axis1_b4_r33_c129_k16`
- `metal/f32/topk/batched/dense_axis1_b4_r33_c17_k16`
- `metal/f32/topk/batched/dense_axis1_b4_r33_c5_k16`
- `metal/f32/topk/batched/dense_axis1_b4_r33_c65_k16`
- `metal/f32/topk/batched/dense_axis1_b4_r3_c129_k1`
- `metal/f32/topk/batched/dense_axis1_b4_r3_c17_k1`
- `metal/f32/topk/batched/dense_axis1_b4_r3_c5_k1`
- `metal/f32/topk/batched/dense_axis1_b4_r3_c65_k1`
- `metal/f32/topk/batched/dense_axis1_b4_r7_c129_k3`
- `metal/f32/topk/batched/dense_axis1_b4_r7_c17_k3`
- `metal/f32/topk/batched/dense_axis1_b4_r7_c5_k3`
- `metal/f32/topk/batched/dense_axis1_b4_r7_c65_k3`
- `metal/f32/topk/batched/dense_axis2_b2_r16_c129_k64`
- `metal/f32/topk/batched/dense_axis2_b2_r16_c17_k8`
- `metal/f32/topk/batched/dense_axis2_b2_r16_c5_k2`
- `metal/f32/topk/batched/dense_axis2_b2_r16_c65_k32`
- `metal/f32/topk/batched/dense_axis2_b2_r33_c129_k64`
- `metal/f32/topk/batched/dense_axis2_b2_r33_c17_k8`
- `metal/f32/topk/batched/dense_axis2_b2_r33_c5_k2`
- `metal/f32/topk/batched/dense_axis2_b2_r33_c65_k32`
- `metal/f32/topk/batched/dense_axis2_b2_r3_c129_k64`
- `metal/f32/topk/batched/dense_axis2_b2_r3_c17_k8`
- `metal/f32/topk/batched/dense_axis2_b2_r3_c5_k2`
- `metal/f32/topk/batched/dense_axis2_b2_r3_c65_k32`
- `metal/f32/topk/batched/dense_axis2_b2_r7_c129_k64`
- `metal/f32/topk/batched/dense_axis2_b2_r7_c17_k8`
- `metal/f32/topk/batched/dense_axis2_b2_r7_c5_k2`
- `metal/f32/topk/batched/dense_axis2_b2_r7_c65_k32`
- `metal/f32/topk/batched/dense_axis2_b4_r16_c129_k64`
- `metal/f32/topk/batched/dense_axis2_b4_r16_c17_k8`
- `metal/f32/topk/batched/dense_axis2_b4_r16_c5_k2`
- `metal/f32/topk/batched/dense_axis2_b4_r16_c65_k32`
- `metal/f32/topk/batched/dense_axis2_b4_r33_c129_k64`
- `metal/f32/topk/batched/dense_axis2_b4_r33_c17_k8`
- `metal/f32/topk/batched/dense_axis2_b4_r33_c5_k2`
- `metal/f32/topk/batched/dense_axis2_b4_r33_c65_k32`
- `metal/f32/topk/batched/dense_axis2_b4_r3_c129_k64`
- `metal/f32/topk/batched/dense_axis2_b4_r3_c17_k8`
- `metal/f32/topk/batched/dense_axis2_b4_r3_c5_k2`
- `metal/f32/topk/batched/dense_axis2_b4_r3_c65_k32`
- `metal/f32/topk/batched/dense_axis2_b4_r7_c129_k64`
- `metal/f32/topk/batched/dense_axis2_b4_r7_c17_k8`
- `metal/f32/topk/batched/dense_axis2_b4_r7_c5_k2`
- `metal/f32/topk/batched/dense_axis2_b4_r7_c65_k32`
