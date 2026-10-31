# Parity regression check

Target: `Mac16,1/26.6.2/arm64`

- Cells compared: 1
- Added cells: 84
- Removed cells: 24
- Changed case definitions: 0
- New correctness failures: 0
- Diagnostic timing suspects: 0
- Target matches: yes
- Measurement protocol matches: yes

Timing changes are diagnostic and do not fail this check. Timing is compared only when target and protocol match. No timing baseline is accepted by this comparison.

| Cell | Correctness regression | Operator relative timing | Program relative timing |
| --- | --- | ---: | ---: |
| `metal/bf16/topk/strided` | none | 0.98× | 1.25× |

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

## Removed cells

- `metal/bf16/contiguous/view_only`
- `metal/bf16/quantized_matmul/groups_and_bits`
- `metal/f32/add/offset`
- `metal/f32/add/offset/r32_c127`
- `metal/f32/contiguous/view_only`
- `metal/f32/div/offset`
- `metal/f32/div/offset/r32_c127`
- `metal/f32/eq/offset`
- `metal/f32/eq/offset/r32_c127`
- `metal/f32/gt/offset`
- `metal/f32/gt/offset/r32_c127`
- `metal/f32/lt/offset`
- `metal/f32/lt/offset/r32_c127`
- `metal/f32/mul/offset`
- `metal/f32/mul/offset/r32_c127`
- `metal/f32/quantized_matmul/groups_and_bits`
- `metal/f32/reduce_to_shape/strided`
- `metal/f32/std_all/long_axis`
- `metal/f32/std_all/strided/r64_c127`
- `metal/f32/sub/offset`
- `metal/f32/sub/offset/r32_c127`
- `metal/f32/topk/strided`
- `metal/f32/variance_all/long_axis`
- `metal/f32/variance_all/strided/r64_c127`
