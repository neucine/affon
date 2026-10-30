# Parity regression check

Target: `Mac16,1/26.6.2/arm64`

- Cells compared: 1
- Added cells: 5
- Removed cells: 0
- Changed case definitions: 0
- New correctness failures: 0
- Diagnostic timing suspects: 0
- Target matches: yes
- Measurement protocol matches: yes

Timing changes are diagnostic and do not fail this check. Timing is compared only when target and protocol match. No timing baseline is accepted by this comparison.

| Cell | Correctness regression | Operator relative timing | Program relative timing |
| --- | --- | ---: | ---: |
| `metal/f32/reduce_to_shape/scalar` | none | 1.11× | 0.91× |

## Added cells

- `metal/f32/reduce_to_shape/long_axis`
- `metal/f32/reduce_to_shape/outer_axis`
- `metal/f32/reduce_to_shape/ragged`
- `metal/f32/reduce_to_shape/short_axis`
- `metal/f32/reduce_to_shape/strided`
