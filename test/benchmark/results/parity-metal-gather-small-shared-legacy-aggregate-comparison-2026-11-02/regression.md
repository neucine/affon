# Small shared gather legacy comparison

All 192 cells passed numerical, operator, and Program parity. Old and new raw reports match cell ID, kind, target, peer engine, shape, dtype, layout, input hash, iterations, and rounds.

| Tier | Median new / old execution |
| --- | ---: |
| operator_peer | 0.033× |
| program_peer | 0.794× |

10 GPU fallback cells retain old reports after repeat timing review:
- `metal/bf16/gather/large/offset_axis0_r65_c127`
- `metal/bf16/gather/large/offset_axis1_r65_c127`
- `metal/bf16/gather/large/r128_c127_i64`
- `metal/bf16/gather/strided/axis0_r65_c127`
- `metal/bf16/gather/strided/axis1_r65_c127`
- `metal/f32/gather/large/offset_axis0_r65_c127`
- `metal/f32/gather/large/offset_axis1_r65_c127`
- `metal/f32/gather/large/r128_c127_i64`
- `metal/f32/gather/strided/axis0_r65_c127`
- `metal/f32/gather/strided/axis1_r65_c127`

The other 182 legacy cells are eligible for promotion. The matched November regression comparisons report no timing suspects relative to the rank-two GPU candidate.
