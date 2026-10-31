# Parity regression check

Target: `Mac16,1/26.6.2/arm64`

- Cells compared: 36
- Added cells: 49
- Removed cells: 156
- Changed case definitions: 0
- New correctness failures: 0
- Diagnostic timing suspects: 0
- Target matches: yes
- Measurement protocol matches: yes

Timing changes are diagnostic and do not fail this check. Timing is compared only when target and protocol match. No timing baseline is accepted by this comparison.

| Cell | Correctness regression | Operator relative timing | Program relative timing |
| --- | --- | ---: | ---: |
| `metal/bf16/topk/small/dense_axis0_r33_c129_k17` | none | 1.26× | 1.10× |
| `metal/bf16/topk/small/dense_axis0_r33_c17_k17` | none | 0.69× | 1.02× |
| `metal/bf16/topk/small/dense_axis0_r33_c3_k17` | none | 0.60× | 0.65× |
| `metal/bf16/topk/small/dense_axis0_r33_c64_k17` | none | 1.04× | 1.06× |
| `metal/bf16/topk/small/dense_axis1_r16_c129_k65` | none | 0.90× | 0.91× |
| `metal/bf16/topk/small/dense_axis1_r16_c64_k33` | none | 1.10× | 0.68× |
| `metal/bf16/topk/small/dense_axis1_r33_c129_k65` | none | 1.01× | 0.64× |
| `metal/bf16/topk/small/dense_axis1_r33_c64_k33` | none | 0.75× | 0.88× |
| `metal/bf16/topk/small/dense_axis1_r3_c129_k65` | none | 0.89× | 0.89× |
| `metal/bf16/topk/small/dense_axis1_r3_c64_k33` | none | 1.04× | 0.85× |
| `metal/bf16/topk/small/dense_axis1_r7_c129_k65` | none | 0.73× | 0.80× |
| `metal/bf16/topk/small/dense_axis1_r7_c64_k33` | none | 0.80× | 1.04× |
| `metal/bf16/topk/small/offset_axis0_r33_c129_k17` | none | 0.80× | 0.90× |
| `metal/bf16/topk/small/offset_axis0_r33_c17_k17` | none | 0.90× | 0.73× |
| `metal/bf16/topk/small/offset_axis0_r33_c3_k17` | none | 0.91× | 0.63× |
| `metal/bf16/topk/small/offset_axis0_r33_c64_k17` | none | 1.01× | 0.75× |
| `metal/bf16/topk/small/offset_axis1_r16_c129_k65` | none | 0.69× | 1.00× |
| `metal/bf16/topk/small/offset_axis1_r16_c64_k33` | none | 0.62× | 0.87× |
| `metal/bf16/topk/small/offset_axis1_r33_c129_k65` | none | 1.03× | 1.07× |
| `metal/bf16/topk/small/offset_axis1_r33_c64_k33` | none | 0.80× | 0.75× |
| `metal/bf16/topk/small/offset_axis1_r3_c129_k65` | none | 0.65× | 0.46× |
| `metal/bf16/topk/small/offset_axis1_r3_c64_k33` | none | 1.00× | 0.76× |
| `metal/bf16/topk/small/offset_axis1_r7_c129_k65` | none | 0.89× | 0.72× |
| `metal/bf16/topk/small/offset_axis1_r7_c64_k33` | none | 0.98× | 0.62× |
| `metal/bf16/topk/small/strided_axis0_r33_c129_k17` | none | 0.97× | 1.35× |
| `metal/bf16/topk/small/strided_axis0_r33_c17_k17` | none | 1.01× | 0.57× |
| `metal/bf16/topk/small/strided_axis0_r33_c3_k17` | none | 0.89× | 0.62× |
| `metal/bf16/topk/small/strided_axis0_r33_c64_k17` | none | 0.71× | 1.21× |
| `metal/bf16/topk/small/strided_axis1_r16_c129_k65` | none | 0.90× | 0.92× |
| `metal/bf16/topk/small/strided_axis1_r16_c64_k33` | none | 0.93× | 1.04× |
| `metal/bf16/topk/small/strided_axis1_r33_c129_k65` | none | 1.00× | 0.92× |
| `metal/bf16/topk/small/strided_axis1_r33_c64_k33` | none | 1.03× | 0.83× |
| `metal/bf16/topk/small/strided_axis1_r3_c129_k65` | none | 0.87× | 0.71× |
| `metal/bf16/topk/small/strided_axis1_r3_c64_k33` | none | 0.86× | 0.84× |
| `metal/bf16/topk/small/strided_axis1_r7_c129_k65` | none | 0.82× | 0.94× |
| `metal/bf16/topk/small/strided_axis1_r7_c64_k33` | none | 1.02× | 1.07× |

## Added cells

- `metal/bf16/topk/batched/dense_axis1_b2_r33_c129_k16`
- `metal/bf16/topk/batched/dense_axis1_b2_r33_c17_k16`
- `metal/bf16/topk/batched/dense_axis1_b2_r33_c5_k16`
- `metal/bf16/topk/batched/dense_axis1_b2_r33_c65_k16`
- `metal/bf16/topk/batched/dense_axis1_b4_r33_c129_k16`
- `metal/bf16/topk/batched/dense_axis1_b4_r33_c17_k16`
- `metal/bf16/topk/batched/dense_axis1_b4_r33_c5_k16`
- `metal/bf16/topk/batched/dense_axis1_b4_r33_c65_k16`
- `metal/bf16/topk/batched/dense_axis2_b2_r16_c129_k64`
- `metal/bf16/topk/batched/dense_axis2_b2_r16_c65_k32`
- `metal/bf16/topk/batched/dense_axis2_b2_r33_c129_k64`
- `metal/bf16/topk/batched/dense_axis2_b2_r33_c65_k32`
- `metal/bf16/topk/batched/dense_axis2_b2_r3_c129_k64`
- `metal/bf16/topk/batched/dense_axis2_b2_r3_c65_k32`
- `metal/bf16/topk/batched/dense_axis2_b2_r7_c129_k64`
- `metal/bf16/topk/batched/dense_axis2_b2_r7_c65_k32`
- `metal/bf16/topk/batched/dense_axis2_b4_r16_c129_k64`
- `metal/bf16/topk/batched/dense_axis2_b4_r16_c65_k32`
- `metal/bf16/topk/batched/dense_axis2_b4_r33_c129_k64`
- `metal/bf16/topk/batched/dense_axis2_b4_r33_c65_k32`
- `metal/bf16/topk/batched/dense_axis2_b4_r3_c129_k64`
- `metal/bf16/topk/batched/dense_axis2_b4_r3_c65_k32`
- `metal/bf16/topk/batched/dense_axis2_b4_r7_c129_k64`
- `metal/bf16/topk/batched/dense_axis2_b4_r7_c65_k32`
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
- `metal/bf16/topk/strided`

## Removed cells

- `metal/bf16/topk/small/dense_axis0_r16_c129_k9`
- `metal/bf16/topk/small/dense_axis0_r16_c17_k9`
- `metal/bf16/topk/small/dense_axis0_r16_c3_k9`
- `metal/bf16/topk/small/dense_axis0_r16_c64_k9`
- `metal/bf16/topk/small/dense_axis0_r3_c129_k2`
- `metal/bf16/topk/small/dense_axis0_r3_c17_k2`
- `metal/bf16/topk/small/dense_axis0_r3_c3_k2`
- `metal/bf16/topk/small/dense_axis0_r3_c64_k2`
- `metal/bf16/topk/small/dense_axis0_r7_c129_k4`
- `metal/bf16/topk/small/dense_axis0_r7_c17_k4`
- `metal/bf16/topk/small/dense_axis0_r7_c3_k4`
- `metal/bf16/topk/small/dense_axis0_r7_c64_k4`
- `metal/bf16/topk/small/dense_axis1_r16_c17_k9`
- `metal/bf16/topk/small/dense_axis1_r16_c3_k2`
- `metal/bf16/topk/small/dense_axis1_r33_c17_k9`
- `metal/bf16/topk/small/dense_axis1_r33_c3_k2`
- `metal/bf16/topk/small/dense_axis1_r3_c17_k9`
- `metal/bf16/topk/small/dense_axis1_r3_c3_k2`
- `metal/bf16/topk/small/dense_axis1_r7_c17_k9`
- `metal/bf16/topk/small/dense_axis1_r7_c3_k2`
- `metal/bf16/topk/small/offset_axis0_r16_c129_k9`
- `metal/bf16/topk/small/offset_axis0_r16_c17_k9`
- `metal/bf16/topk/small/offset_axis0_r16_c3_k9`
- `metal/bf16/topk/small/offset_axis0_r16_c64_k9`
- `metal/bf16/topk/small/offset_axis0_r3_c129_k2`
- `metal/bf16/topk/small/offset_axis0_r3_c17_k2`
- `metal/bf16/topk/small/offset_axis0_r3_c3_k2`
- `metal/bf16/topk/small/offset_axis0_r3_c64_k2`
- `metal/bf16/topk/small/offset_axis0_r7_c129_k4`
- `metal/bf16/topk/small/offset_axis0_r7_c17_k4`
- `metal/bf16/topk/small/offset_axis0_r7_c3_k4`
- `metal/bf16/topk/small/offset_axis0_r7_c64_k4`
- `metal/bf16/topk/small/offset_axis1_r16_c17_k9`
- `metal/bf16/topk/small/offset_axis1_r16_c3_k2`
- `metal/bf16/topk/small/offset_axis1_r33_c17_k9`
- `metal/bf16/topk/small/offset_axis1_r33_c3_k2`
- `metal/bf16/topk/small/offset_axis1_r3_c17_k9`
- `metal/bf16/topk/small/offset_axis1_r3_c3_k2`
- `metal/bf16/topk/small/offset_axis1_r7_c17_k9`
- `metal/bf16/topk/small/offset_axis1_r7_c3_k2`
- `metal/bf16/topk/small/strided_axis0_r16_c129_k9`
- `metal/bf16/topk/small/strided_axis0_r16_c17_k9`
- `metal/bf16/topk/small/strided_axis0_r16_c3_k9`
- `metal/bf16/topk/small/strided_axis0_r16_c64_k9`
- `metal/bf16/topk/small/strided_axis0_r3_c129_k2`
- `metal/bf16/topk/small/strided_axis0_r3_c17_k2`
- `metal/bf16/topk/small/strided_axis0_r3_c3_k2`
- `metal/bf16/topk/small/strided_axis0_r3_c64_k2`
- `metal/bf16/topk/small/strided_axis0_r7_c129_k4`
- `metal/bf16/topk/small/strided_axis0_r7_c17_k4`
- `metal/bf16/topk/small/strided_axis0_r7_c3_k4`
- `metal/bf16/topk/small/strided_axis0_r7_c64_k4`
- `metal/bf16/topk/small/strided_axis1_r16_c17_k9`
- `metal/bf16/topk/small/strided_axis1_r16_c3_k2`
- `metal/bf16/topk/small/strided_axis1_r33_c17_k9`
- `metal/bf16/topk/small/strided_axis1_r33_c3_k2`
- `metal/bf16/topk/small/strided_axis1_r3_c17_k9`
- `metal/bf16/topk/small/strided_axis1_r3_c3_k2`
- `metal/bf16/topk/small/strided_axis1_r7_c17_k9`
- `metal/bf16/topk/small/strided_axis1_r7_c3_k2`
- `metal/f32/topk/small/dense_axis0_r16_c129_k9`
- `metal/f32/topk/small/dense_axis0_r16_c17_k9`
- `metal/f32/topk/small/dense_axis0_r16_c3_k9`
- `metal/f32/topk/small/dense_axis0_r16_c64_k9`
- `metal/f32/topk/small/dense_axis0_r33_c129_k17`
- `metal/f32/topk/small/dense_axis0_r33_c17_k17`
- `metal/f32/topk/small/dense_axis0_r33_c3_k17`
- `metal/f32/topk/small/dense_axis0_r33_c64_k17`
- `metal/f32/topk/small/dense_axis0_r3_c129_k2`
- `metal/f32/topk/small/dense_axis0_r3_c17_k2`
- `metal/f32/topk/small/dense_axis0_r3_c3_k2`
- `metal/f32/topk/small/dense_axis0_r3_c64_k2`
- `metal/f32/topk/small/dense_axis0_r7_c129_k4`
- `metal/f32/topk/small/dense_axis0_r7_c17_k4`
- `metal/f32/topk/small/dense_axis0_r7_c3_k4`
- `metal/f32/topk/small/dense_axis0_r7_c64_k4`
- `metal/f32/topk/small/dense_axis1_r16_c129_k65`
- `metal/f32/topk/small/dense_axis1_r16_c17_k9`
- `metal/f32/topk/small/dense_axis1_r16_c3_k2`
- `metal/f32/topk/small/dense_axis1_r16_c64_k33`
- `metal/f32/topk/small/dense_axis1_r33_c129_k65`
- `metal/f32/topk/small/dense_axis1_r33_c17_k9`
- `metal/f32/topk/small/dense_axis1_r33_c3_k2`
- `metal/f32/topk/small/dense_axis1_r33_c64_k33`
- `metal/f32/topk/small/dense_axis1_r3_c129_k65`
- `metal/f32/topk/small/dense_axis1_r3_c17_k9`
- `metal/f32/topk/small/dense_axis1_r3_c3_k2`
- `metal/f32/topk/small/dense_axis1_r3_c64_k33`
- `metal/f32/topk/small/dense_axis1_r7_c129_k65`
- `metal/f32/topk/small/dense_axis1_r7_c17_k9`
- `metal/f32/topk/small/dense_axis1_r7_c3_k2`
- `metal/f32/topk/small/dense_axis1_r7_c64_k33`
- `metal/f32/topk/small/offset_axis0_r16_c129_k9`
- `metal/f32/topk/small/offset_axis0_r16_c17_k9`
- `metal/f32/topk/small/offset_axis0_r16_c3_k9`
- `metal/f32/topk/small/offset_axis0_r16_c64_k9`
- `metal/f32/topk/small/offset_axis0_r33_c129_k17`
- `metal/f32/topk/small/offset_axis0_r33_c17_k17`
- `metal/f32/topk/small/offset_axis0_r33_c3_k17`
- `metal/f32/topk/small/offset_axis0_r33_c64_k17`
- `metal/f32/topk/small/offset_axis0_r3_c129_k2`
- `metal/f32/topk/small/offset_axis0_r3_c17_k2`
- `metal/f32/topk/small/offset_axis0_r3_c3_k2`
- `metal/f32/topk/small/offset_axis0_r3_c64_k2`
- `metal/f32/topk/small/offset_axis0_r7_c129_k4`
- `metal/f32/topk/small/offset_axis0_r7_c17_k4`
- `metal/f32/topk/small/offset_axis0_r7_c3_k4`
- `metal/f32/topk/small/offset_axis0_r7_c64_k4`
- `metal/f32/topk/small/offset_axis1_r16_c129_k65`
- `metal/f32/topk/small/offset_axis1_r16_c17_k9`
- `metal/f32/topk/small/offset_axis1_r16_c3_k2`
- `metal/f32/topk/small/offset_axis1_r16_c64_k33`
- `metal/f32/topk/small/offset_axis1_r33_c129_k65`
- `metal/f32/topk/small/offset_axis1_r33_c17_k9`
- `metal/f32/topk/small/offset_axis1_r33_c3_k2`
- `metal/f32/topk/small/offset_axis1_r33_c64_k33`
- `metal/f32/topk/small/offset_axis1_r3_c129_k65`
- `metal/f32/topk/small/offset_axis1_r3_c17_k9`
- `metal/f32/topk/small/offset_axis1_r3_c3_k2`
- `metal/f32/topk/small/offset_axis1_r3_c64_k33`
- `metal/f32/topk/small/offset_axis1_r7_c129_k65`
- `metal/f32/topk/small/offset_axis1_r7_c17_k9`
- `metal/f32/topk/small/offset_axis1_r7_c3_k2`
- `metal/f32/topk/small/offset_axis1_r7_c64_k33`
- `metal/f32/topk/small/strided_axis0_r16_c129_k9`
- `metal/f32/topk/small/strided_axis0_r16_c17_k9`
- `metal/f32/topk/small/strided_axis0_r16_c3_k9`
- `metal/f32/topk/small/strided_axis0_r16_c64_k9`
- `metal/f32/topk/small/strided_axis0_r33_c129_k17`
- `metal/f32/topk/small/strided_axis0_r33_c17_k17`
- `metal/f32/topk/small/strided_axis0_r33_c3_k17`
- `metal/f32/topk/small/strided_axis0_r33_c64_k17`
- `metal/f32/topk/small/strided_axis0_r3_c129_k2`
- `metal/f32/topk/small/strided_axis0_r3_c17_k2`
- `metal/f32/topk/small/strided_axis0_r3_c3_k2`
- `metal/f32/topk/small/strided_axis0_r3_c64_k2`
- `metal/f32/topk/small/strided_axis0_r7_c129_k4`
- `metal/f32/topk/small/strided_axis0_r7_c17_k4`
- `metal/f32/topk/small/strided_axis0_r7_c3_k4`
- `metal/f32/topk/small/strided_axis0_r7_c64_k4`
- `metal/f32/topk/small/strided_axis1_r16_c129_k65`
- `metal/f32/topk/small/strided_axis1_r16_c17_k9`
- `metal/f32/topk/small/strided_axis1_r16_c3_k2`
- `metal/f32/topk/small/strided_axis1_r16_c64_k33`
- `metal/f32/topk/small/strided_axis1_r33_c129_k65`
- `metal/f32/topk/small/strided_axis1_r33_c17_k9`
- `metal/f32/topk/small/strided_axis1_r33_c3_k2`
- `metal/f32/topk/small/strided_axis1_r33_c64_k33`
- `metal/f32/topk/small/strided_axis1_r3_c129_k65`
- `metal/f32/topk/small/strided_axis1_r3_c17_k9`
- `metal/f32/topk/small/strided_axis1_r3_c3_k2`
- `metal/f32/topk/small/strided_axis1_r3_c64_k33`
- `metal/f32/topk/small/strided_axis1_r7_c129_k65`
- `metal/f32/topk/small/strided_axis1_r7_c17_k9`
- `metal/f32/topk/small/strided_axis1_r7_c3_k2`
- `metal/f32/topk/small/strided_axis1_r7_c64_k33`
