# Parity regression check

Target: `Mac16,1/26.6.2/arm64`

- Cells compared: 1
- Added cells: 5
- Removed cells: 24
- Changed case definitions: 0
- New correctness failures: 0
- Diagnostic timing suspects: 0
- Target matches: yes
- Measurement protocol matches: yes

Timing changes are diagnostic and do not fail this check. Timing is compared only when target and protocol match. No timing baseline is accepted by this comparison.

| Cell | Correctness regression | Operator relative timing | Program relative timing |
| --- | --- | ---: | ---: |
| `metal/f32/reduce_to_shape/strided` | none | 0.16× | 0.17× |

## Added cells

- `metal/f32/reduce_to_shape/long_axis`
- `metal/f32/reduce_to_shape/outer_axis`
- `metal/f32/reduce_to_shape/ragged`
- `metal/f32/reduce_to_shape/scalar`
- `metal/f32/reduce_to_shape/short_axis`

## Removed cells

- `metal/bf16/contiguous/view_only`
- `metal/bf16/quantized_matmul/groups_and_bits`
- `metal/bf16/topk/strided`
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
- `metal/f32/std_all/long_axis`
- `metal/f32/std_all/strided/r64_c127`
- `metal/f32/sub/offset`
- `metal/f32/sub/offset/r32_c127`
- `metal/f32/topk/strided`
- `metal/f32/variance_all/long_axis`
- `metal/f32/variance_all/strided/r64_c127`
