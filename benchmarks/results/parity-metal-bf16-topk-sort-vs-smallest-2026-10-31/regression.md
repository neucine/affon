# Parity regression check

Target: `Mac16,1/26.6.2/arm64`

- Cells compared: 24
- Added cells: 61
- Removed cells: 168
- Changed case definitions: 0
- New correctness failures: 0
- Diagnostic timing suspects: 0
- Target matches: yes
- Measurement protocol matches: yes

Timing changes are diagnostic and do not fail this check. Timing is compared only when target and protocol match. No timing baseline is accepted by this comparison.

| Cell | Correctness regression | Operator relative timing | Program relative timing |
| --- | --- | ---: | ---: |
| `metal/bf16/topk/large/dense_axis1_r128_c257_k256_smallest` | none | 0.28× | 0.34× |
| `metal/bf16/topk/large/dense_axis1_r128_c65_k64_smallest` | none | 0.56× | 0.54× |
| `metal/bf16/topk/large/dense_axis1_r32_c257_k256_smallest` | none | 0.35× | 0.37× |
| `metal/bf16/topk/large/dense_axis1_r32_c65_k64_smallest` | none | 0.83× | 0.99× |
| `metal/bf16/topk/large/dense_axis1_r64_c257_k256_smallest` | none | 0.46× | 0.46× |
| `metal/bf16/topk/large/dense_axis1_r64_c65_k64_smallest` | none | 0.76× | 0.82× |
| `metal/bf16/topk/large/dense_axis1_r8_c257_k256_smallest` | none | 0.38× | 0.44× |
| `metal/bf16/topk/large/dense_axis1_r8_c65_k64_smallest` | none | 0.87× | 0.97× |
| `metal/bf16/topk/large/offset_axis1_r128_c257_k256_smallest` | none | 0.24× | 0.27× |
| `metal/bf16/topk/large/offset_axis1_r128_c65_k64_smallest` | none | 0.66× | 0.53× |
| `metal/bf16/topk/large/offset_axis1_r32_c257_k256_smallest` | none | 0.43× | 0.41× |
| `metal/bf16/topk/large/offset_axis1_r32_c65_k64_smallest` | none | 0.71× | 0.98× |
| `metal/bf16/topk/large/offset_axis1_r64_c257_k256_smallest` | none | 0.37× | 0.50× |
| `metal/bf16/topk/large/offset_axis1_r64_c65_k64_smallest` | none | 0.84× | 1.00× |
| `metal/bf16/topk/large/offset_axis1_r8_c257_k256_smallest` | none | 0.46× | 0.55× |
| `metal/bf16/topk/large/offset_axis1_r8_c65_k64_smallest` | none | 0.89× | 0.94× |
| `metal/bf16/topk/large/strided_axis1_r128_c257_k256_smallest` | none | 0.31× | 0.32× |
| `metal/bf16/topk/large/strided_axis1_r128_c65_k64_smallest` | none | 0.52× | 0.53× |
| `metal/bf16/topk/large/strided_axis1_r32_c257_k256_smallest` | none | 0.52× | 0.28× |
| `metal/bf16/topk/large/strided_axis1_r32_c65_k64_smallest` | none | 0.89× | 0.97× |
| `metal/bf16/topk/large/strided_axis1_r64_c257_k256_smallest` | none | 0.24× | 0.47× |
| `metal/bf16/topk/large/strided_axis1_r64_c65_k64_smallest` | none | 0.69× | 0.98× |
| `metal/bf16/topk/large/strided_axis1_r8_c257_k256_smallest` | none | 0.29× | 0.35× |
| `metal/bf16/topk/large/strided_axis1_r8_c65_k64_smallest` | none | 0.86× | 0.93× |

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

- `metal/bf16/topk/large/dense_axis0_r128_c257_k1_smallest`
- `metal/bf16/topk/large/dense_axis0_r128_c31_k1_smallest`
- `metal/bf16/topk/large/dense_axis0_r128_c65_k1_smallest`
- `metal/bf16/topk/large/dense_axis0_r128_c7_k1_smallest`
- `metal/bf16/topk/large/dense_axis0_r32_c257_k1_smallest`
- `metal/bf16/topk/large/dense_axis0_r32_c31_k1_smallest`
- `metal/bf16/topk/large/dense_axis0_r32_c65_k1_smallest`
- `metal/bf16/topk/large/dense_axis0_r32_c7_k1_smallest`
- `metal/bf16/topk/large/dense_axis0_r64_c257_k1_smallest`
- `metal/bf16/topk/large/dense_axis0_r64_c31_k1_smallest`
- `metal/bf16/topk/large/dense_axis0_r64_c65_k1_smallest`
- `metal/bf16/topk/large/dense_axis0_r64_c7_k1_smallest`
- `metal/bf16/topk/large/dense_axis0_r8_c257_k1_smallest`
- `metal/bf16/topk/large/dense_axis0_r8_c31_k1_smallest`
- `metal/bf16/topk/large/dense_axis0_r8_c65_k1_smallest`
- `metal/bf16/topk/large/dense_axis0_r8_c7_k1_smallest`
- `metal/bf16/topk/large/dense_axis1_r128_c31_k30_smallest`
- `metal/bf16/topk/large/dense_axis1_r128_c7_k6_smallest`
- `metal/bf16/topk/large/dense_axis1_r32_c31_k30_smallest`
- `metal/bf16/topk/large/dense_axis1_r32_c7_k6_smallest`
- `metal/bf16/topk/large/dense_axis1_r64_c31_k30_smallest`
- `metal/bf16/topk/large/dense_axis1_r64_c7_k6_smallest`
- `metal/bf16/topk/large/dense_axis1_r8_c31_k30_smallest`
- `metal/bf16/topk/large/dense_axis1_r8_c7_k6_smallest`
- `metal/bf16/topk/large/offset_axis0_r128_c257_k1_smallest`
- `metal/bf16/topk/large/offset_axis0_r128_c31_k1_smallest`
- `metal/bf16/topk/large/offset_axis0_r128_c65_k1_smallest`
- `metal/bf16/topk/large/offset_axis0_r128_c7_k1_smallest`
- `metal/bf16/topk/large/offset_axis0_r32_c257_k1_smallest`
- `metal/bf16/topk/large/offset_axis0_r32_c31_k1_smallest`
- `metal/bf16/topk/large/offset_axis0_r32_c65_k1_smallest`
- `metal/bf16/topk/large/offset_axis0_r32_c7_k1_smallest`
- `metal/bf16/topk/large/offset_axis0_r64_c257_k1_smallest`
- `metal/bf16/topk/large/offset_axis0_r64_c31_k1_smallest`
- `metal/bf16/topk/large/offset_axis0_r64_c65_k1_smallest`
- `metal/bf16/topk/large/offset_axis0_r64_c7_k1_smallest`
- `metal/bf16/topk/large/offset_axis0_r8_c257_k1_smallest`
- `metal/bf16/topk/large/offset_axis0_r8_c31_k1_smallest`
- `metal/bf16/topk/large/offset_axis0_r8_c65_k1_smallest`
- `metal/bf16/topk/large/offset_axis0_r8_c7_k1_smallest`
- `metal/bf16/topk/large/offset_axis1_r128_c31_k30_smallest`
- `metal/bf16/topk/large/offset_axis1_r128_c7_k6_smallest`
- `metal/bf16/topk/large/offset_axis1_r32_c31_k30_smallest`
- `metal/bf16/topk/large/offset_axis1_r32_c7_k6_smallest`
- `metal/bf16/topk/large/offset_axis1_r64_c31_k30_smallest`
- `metal/bf16/topk/large/offset_axis1_r64_c7_k6_smallest`
- `metal/bf16/topk/large/offset_axis1_r8_c31_k30_smallest`
- `metal/bf16/topk/large/offset_axis1_r8_c7_k6_smallest`
- `metal/bf16/topk/large/strided_axis0_r128_c257_k1_smallest`
- `metal/bf16/topk/large/strided_axis0_r128_c31_k1_smallest`
- `metal/bf16/topk/large/strided_axis0_r128_c65_k1_smallest`
- `metal/bf16/topk/large/strided_axis0_r128_c7_k1_smallest`
- `metal/bf16/topk/large/strided_axis0_r32_c257_k1_smallest`
- `metal/bf16/topk/large/strided_axis0_r32_c31_k1_smallest`
- `metal/bf16/topk/large/strided_axis0_r32_c65_k1_smallest`
- `metal/bf16/topk/large/strided_axis0_r32_c7_k1_smallest`
- `metal/bf16/topk/large/strided_axis0_r64_c257_k1_smallest`
- `metal/bf16/topk/large/strided_axis0_r64_c31_k1_smallest`
- `metal/bf16/topk/large/strided_axis0_r64_c65_k1_smallest`
- `metal/bf16/topk/large/strided_axis0_r64_c7_k1_smallest`
- `metal/bf16/topk/large/strided_axis0_r8_c257_k1_smallest`
- `metal/bf16/topk/large/strided_axis0_r8_c31_k1_smallest`
- `metal/bf16/topk/large/strided_axis0_r8_c65_k1_smallest`
- `metal/bf16/topk/large/strided_axis0_r8_c7_k1_smallest`
- `metal/bf16/topk/large/strided_axis1_r128_c31_k30_smallest`
- `metal/bf16/topk/large/strided_axis1_r128_c7_k6_smallest`
- `metal/bf16/topk/large/strided_axis1_r32_c31_k30_smallest`
- `metal/bf16/topk/large/strided_axis1_r32_c7_k6_smallest`
- `metal/bf16/topk/large/strided_axis1_r64_c31_k30_smallest`
- `metal/bf16/topk/large/strided_axis1_r64_c7_k6_smallest`
- `metal/bf16/topk/large/strided_axis1_r8_c31_k30_smallest`
- `metal/bf16/topk/large/strided_axis1_r8_c7_k6_smallest`
- `metal/f32/topk/large/dense_axis0_r128_c257_k1_smallest`
- `metal/f32/topk/large/dense_axis0_r128_c31_k1_smallest`
- `metal/f32/topk/large/dense_axis0_r128_c65_k1_smallest`
- `metal/f32/topk/large/dense_axis0_r128_c7_k1_smallest`
- `metal/f32/topk/large/dense_axis0_r32_c257_k1_smallest`
- `metal/f32/topk/large/dense_axis0_r32_c31_k1_smallest`
- `metal/f32/topk/large/dense_axis0_r32_c65_k1_smallest`
- `metal/f32/topk/large/dense_axis0_r32_c7_k1_smallest`
- `metal/f32/topk/large/dense_axis0_r64_c257_k1_smallest`
- `metal/f32/topk/large/dense_axis0_r64_c31_k1_smallest`
- `metal/f32/topk/large/dense_axis0_r64_c65_k1_smallest`
- `metal/f32/topk/large/dense_axis0_r64_c7_k1_smallest`
- `metal/f32/topk/large/dense_axis0_r8_c257_k1_smallest`
- `metal/f32/topk/large/dense_axis0_r8_c31_k1_smallest`
- `metal/f32/topk/large/dense_axis0_r8_c65_k1_smallest`
- `metal/f32/topk/large/dense_axis0_r8_c7_k1_smallest`
- `metal/f32/topk/large/dense_axis1_r128_c257_k256_smallest`
- `metal/f32/topk/large/dense_axis1_r128_c31_k30_smallest`
- `metal/f32/topk/large/dense_axis1_r128_c65_k64_smallest`
- `metal/f32/topk/large/dense_axis1_r128_c7_k6_smallest`
- `metal/f32/topk/large/dense_axis1_r32_c257_k256_smallest`
- `metal/f32/topk/large/dense_axis1_r32_c31_k30_smallest`
- `metal/f32/topk/large/dense_axis1_r32_c65_k64_smallest`
- `metal/f32/topk/large/dense_axis1_r32_c7_k6_smallest`
- `metal/f32/topk/large/dense_axis1_r64_c257_k256_smallest`
- `metal/f32/topk/large/dense_axis1_r64_c31_k30_smallest`
- `metal/f32/topk/large/dense_axis1_r64_c65_k64_smallest`
- `metal/f32/topk/large/dense_axis1_r64_c7_k6_smallest`
- `metal/f32/topk/large/dense_axis1_r8_c257_k256_smallest`
- `metal/f32/topk/large/dense_axis1_r8_c31_k30_smallest`
- `metal/f32/topk/large/dense_axis1_r8_c65_k64_smallest`
- `metal/f32/topk/large/dense_axis1_r8_c7_k6_smallest`
- `metal/f32/topk/large/offset_axis0_r128_c257_k1_smallest`
- `metal/f32/topk/large/offset_axis0_r128_c31_k1_smallest`
- `metal/f32/topk/large/offset_axis0_r128_c65_k1_smallest`
- `metal/f32/topk/large/offset_axis0_r128_c7_k1_smallest`
- `metal/f32/topk/large/offset_axis0_r32_c257_k1_smallest`
- `metal/f32/topk/large/offset_axis0_r32_c31_k1_smallest`
- `metal/f32/topk/large/offset_axis0_r32_c65_k1_smallest`
- `metal/f32/topk/large/offset_axis0_r32_c7_k1_smallest`
- `metal/f32/topk/large/offset_axis0_r64_c257_k1_smallest`
- `metal/f32/topk/large/offset_axis0_r64_c31_k1_smallest`
- `metal/f32/topk/large/offset_axis0_r64_c65_k1_smallest`
- `metal/f32/topk/large/offset_axis0_r64_c7_k1_smallest`
- `metal/f32/topk/large/offset_axis0_r8_c257_k1_smallest`
- `metal/f32/topk/large/offset_axis0_r8_c31_k1_smallest`
- `metal/f32/topk/large/offset_axis0_r8_c65_k1_smallest`
- `metal/f32/topk/large/offset_axis0_r8_c7_k1_smallest`
- `metal/f32/topk/large/offset_axis1_r128_c257_k256_smallest`
- `metal/f32/topk/large/offset_axis1_r128_c31_k30_smallest`
- `metal/f32/topk/large/offset_axis1_r128_c65_k64_smallest`
- `metal/f32/topk/large/offset_axis1_r128_c7_k6_smallest`
- `metal/f32/topk/large/offset_axis1_r32_c257_k256_smallest`
- `metal/f32/topk/large/offset_axis1_r32_c31_k30_smallest`
- `metal/f32/topk/large/offset_axis1_r32_c65_k64_smallest`
- `metal/f32/topk/large/offset_axis1_r32_c7_k6_smallest`
- `metal/f32/topk/large/offset_axis1_r64_c257_k256_smallest`
- `metal/f32/topk/large/offset_axis1_r64_c31_k30_smallest`
- `metal/f32/topk/large/offset_axis1_r64_c65_k64_smallest`
- `metal/f32/topk/large/offset_axis1_r64_c7_k6_smallest`
- `metal/f32/topk/large/offset_axis1_r8_c257_k256_smallest`
- `metal/f32/topk/large/offset_axis1_r8_c31_k30_smallest`
- `metal/f32/topk/large/offset_axis1_r8_c65_k64_smallest`
- `metal/f32/topk/large/offset_axis1_r8_c7_k6_smallest`
- `metal/f32/topk/large/strided_axis0_r128_c257_k1_smallest`
- `metal/f32/topk/large/strided_axis0_r128_c31_k1_smallest`
- `metal/f32/topk/large/strided_axis0_r128_c65_k1_smallest`
- `metal/f32/topk/large/strided_axis0_r128_c7_k1_smallest`
- `metal/f32/topk/large/strided_axis0_r32_c257_k1_smallest`
- `metal/f32/topk/large/strided_axis0_r32_c31_k1_smallest`
- `metal/f32/topk/large/strided_axis0_r32_c65_k1_smallest`
- `metal/f32/topk/large/strided_axis0_r32_c7_k1_smallest`
- `metal/f32/topk/large/strided_axis0_r64_c257_k1_smallest`
- `metal/f32/topk/large/strided_axis0_r64_c31_k1_smallest`
- `metal/f32/topk/large/strided_axis0_r64_c65_k1_smallest`
- `metal/f32/topk/large/strided_axis0_r64_c7_k1_smallest`
- `metal/f32/topk/large/strided_axis0_r8_c257_k1_smallest`
- `metal/f32/topk/large/strided_axis0_r8_c31_k1_smallest`
- `metal/f32/topk/large/strided_axis0_r8_c65_k1_smallest`
- `metal/f32/topk/large/strided_axis0_r8_c7_k1_smallest`
- `metal/f32/topk/large/strided_axis1_r128_c257_k256_smallest`
- `metal/f32/topk/large/strided_axis1_r128_c31_k30_smallest`
- `metal/f32/topk/large/strided_axis1_r128_c65_k64_smallest`
- `metal/f32/topk/large/strided_axis1_r128_c7_k6_smallest`
- `metal/f32/topk/large/strided_axis1_r32_c257_k256_smallest`
- `metal/f32/topk/large/strided_axis1_r32_c31_k30_smallest`
- `metal/f32/topk/large/strided_axis1_r32_c65_k64_smallest`
- `metal/f32/topk/large/strided_axis1_r32_c7_k6_smallest`
- `metal/f32/topk/large/strided_axis1_r64_c257_k256_smallest`
- `metal/f32/topk/large/strided_axis1_r64_c31_k30_smallest`
- `metal/f32/topk/large/strided_axis1_r64_c65_k64_smallest`
- `metal/f32/topk/large/strided_axis1_r64_c7_k6_smallest`
- `metal/f32/topk/large/strided_axis1_r8_c257_k256_smallest`
- `metal/f32/topk/large/strided_axis1_r8_c31_k30_smallest`
- `metal/f32/topk/large/strided_axis1_r8_c65_k64_smallest`
- `metal/f32/topk/large/strided_axis1_r8_c7_k6_smallest`
