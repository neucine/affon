# Parity regression check

Target: `Mac16,1/26.6.2/arm64`

- Cells compared: 16
- Added cells: 0
- Removed cells: 112
- Changed case definitions: 0
- New correctness failures: 0
- Diagnostic timing suspects: 0
- Target matches: yes
- Measurement protocol matches: yes

Timing changes are diagnostic and do not fail this check. Timing is compared only when target and protocol match. No timing baseline is accepted by this comparison.

| Cell | Correctness regression | Operator relative timing | Program relative timing |
| --- | --- | ---: | ---: |
| `metal/f32/reduce_to_shape/ragged/dense_r128_c129` | none | 0.02× | 0.02× |
| `metal/f32/reduce_to_shape/ragged/dense_r128_c32` | none | 0.06× | 0.09× |
| `metal/f32/reduce_to_shape/ragged/dense_r128_c65` | none | 0.04× | 0.03× |
| `metal/f32/reduce_to_shape/ragged/dense_r16_c129` | none | 0.14× | 0.15× |
| `metal/f32/reduce_to_shape/ragged/dense_r16_c65` | none | 0.19× | 0.21× |
| `metal/f32/reduce_to_shape/ragged/dense_r33_c129` | none | 0.08× | 0.06× |
| `metal/f32/reduce_to_shape/ragged/dense_r33_c32` | none | 0.18× | 0.21× |
| `metal/f32/reduce_to_shape/ragged/dense_r33_c65` | none | 0.13× | 0.13× |
| `metal/f32/reduce_to_shape/ragged/offset_r128_c129` | none | 0.02× | 0.02× |
| `metal/f32/reduce_to_shape/ragged/offset_r128_c32` | none | 0.08× | 0.08× |
| `metal/f32/reduce_to_shape/ragged/offset_r128_c65` | none | 0.04× | 0.03× |
| `metal/f32/reduce_to_shape/ragged/offset_r16_c129` | none | 0.14× | 0.16× |
| `metal/f32/reduce_to_shape/ragged/offset_r16_c65` | none | 0.18× | 0.20× |
| `metal/f32/reduce_to_shape/ragged/offset_r33_c129` | none | 0.06× | 0.08× |
| `metal/f32/reduce_to_shape/ragged/offset_r33_c32` | none | 0.18× | 0.22× |
| `metal/f32/reduce_to_shape/ragged/offset_r33_c65` | none | 0.11× | 0.14× |

## Removed cells

- `metal/f32/reduce_to_shape/long_axis/dense_r128_c129`
- `metal/f32/reduce_to_shape/long_axis/dense_r128_c32`
- `metal/f32/reduce_to_shape/long_axis/dense_r128_c5`
- `metal/f32/reduce_to_shape/long_axis/dense_r128_c65`
- `metal/f32/reduce_to_shape/long_axis/dense_r16_c129`
- `metal/f32/reduce_to_shape/long_axis/dense_r16_c32`
- `metal/f32/reduce_to_shape/long_axis/dense_r16_c5`
- `metal/f32/reduce_to_shape/long_axis/dense_r16_c65`
- `metal/f32/reduce_to_shape/long_axis/dense_r33_c129`
- `metal/f32/reduce_to_shape/long_axis/dense_r33_c32`
- `metal/f32/reduce_to_shape/long_axis/dense_r33_c5`
- `metal/f32/reduce_to_shape/long_axis/dense_r33_c65`
- `metal/f32/reduce_to_shape/long_axis/dense_r3_c129`
- `metal/f32/reduce_to_shape/long_axis/dense_r3_c32`
- `metal/f32/reduce_to_shape/long_axis/dense_r3_c5`
- `metal/f32/reduce_to_shape/long_axis/dense_r3_c65`
- `metal/f32/reduce_to_shape/long_axis/offset_r128_c129`
- `metal/f32/reduce_to_shape/long_axis/offset_r128_c32`
- `metal/f32/reduce_to_shape/long_axis/offset_r128_c5`
- `metal/f32/reduce_to_shape/long_axis/offset_r128_c65`
- `metal/f32/reduce_to_shape/long_axis/offset_r16_c129`
- `metal/f32/reduce_to_shape/long_axis/offset_r16_c32`
- `metal/f32/reduce_to_shape/long_axis/offset_r16_c5`
- `metal/f32/reduce_to_shape/long_axis/offset_r16_c65`
- `metal/f32/reduce_to_shape/long_axis/offset_r33_c129`
- `metal/f32/reduce_to_shape/long_axis/offset_r33_c32`
- `metal/f32/reduce_to_shape/long_axis/offset_r33_c5`
- `metal/f32/reduce_to_shape/long_axis/offset_r33_c65`
- `metal/f32/reduce_to_shape/long_axis/offset_r3_c129`
- `metal/f32/reduce_to_shape/long_axis/offset_r3_c32`
- `metal/f32/reduce_to_shape/long_axis/offset_r3_c5`
- `metal/f32/reduce_to_shape/long_axis/offset_r3_c65`
- `metal/f32/reduce_to_shape/outer_axis/dense_r128_c129`
- `metal/f32/reduce_to_shape/outer_axis/dense_r128_c32`
- `metal/f32/reduce_to_shape/outer_axis/dense_r128_c5`
- `metal/f32/reduce_to_shape/outer_axis/dense_r128_c65`
- `metal/f32/reduce_to_shape/outer_axis/dense_r16_c129`
- `metal/f32/reduce_to_shape/outer_axis/dense_r16_c32`
- `metal/f32/reduce_to_shape/outer_axis/dense_r16_c5`
- `metal/f32/reduce_to_shape/outer_axis/dense_r16_c65`
- `metal/f32/reduce_to_shape/outer_axis/dense_r33_c129`
- `metal/f32/reduce_to_shape/outer_axis/dense_r33_c32`
- `metal/f32/reduce_to_shape/outer_axis/dense_r33_c5`
- `metal/f32/reduce_to_shape/outer_axis/dense_r33_c65`
- `metal/f32/reduce_to_shape/outer_axis/dense_r3_c129`
- `metal/f32/reduce_to_shape/outer_axis/dense_r3_c32`
- `metal/f32/reduce_to_shape/outer_axis/dense_r3_c5`
- `metal/f32/reduce_to_shape/outer_axis/dense_r3_c65`
- `metal/f32/reduce_to_shape/outer_axis/offset_r128_c129`
- `metal/f32/reduce_to_shape/outer_axis/offset_r128_c32`
- `metal/f32/reduce_to_shape/outer_axis/offset_r128_c5`
- `metal/f32/reduce_to_shape/outer_axis/offset_r128_c65`
- `metal/f32/reduce_to_shape/outer_axis/offset_r16_c129`
- `metal/f32/reduce_to_shape/outer_axis/offset_r16_c32`
- `metal/f32/reduce_to_shape/outer_axis/offset_r16_c5`
- `metal/f32/reduce_to_shape/outer_axis/offset_r16_c65`
- `metal/f32/reduce_to_shape/outer_axis/offset_r33_c129`
- `metal/f32/reduce_to_shape/outer_axis/offset_r33_c32`
- `metal/f32/reduce_to_shape/outer_axis/offset_r33_c5`
- `metal/f32/reduce_to_shape/outer_axis/offset_r33_c65`
- `metal/f32/reduce_to_shape/outer_axis/offset_r3_c129`
- `metal/f32/reduce_to_shape/outer_axis/offset_r3_c32`
- `metal/f32/reduce_to_shape/outer_axis/offset_r3_c5`
- `metal/f32/reduce_to_shape/outer_axis/offset_r3_c65`
- `metal/f32/reduce_to_shape/ragged/dense_r128_c5`
- `metal/f32/reduce_to_shape/ragged/dense_r16_c32`
- `metal/f32/reduce_to_shape/ragged/dense_r16_c5`
- `metal/f32/reduce_to_shape/ragged/dense_r33_c5`
- `metal/f32/reduce_to_shape/ragged/dense_r3_c129`
- `metal/f32/reduce_to_shape/ragged/dense_r3_c32`
- `metal/f32/reduce_to_shape/ragged/dense_r3_c5`
- `metal/f32/reduce_to_shape/ragged/dense_r3_c65`
- `metal/f32/reduce_to_shape/ragged/offset_r128_c5`
- `metal/f32/reduce_to_shape/ragged/offset_r16_c32`
- `metal/f32/reduce_to_shape/ragged/offset_r16_c5`
- `metal/f32/reduce_to_shape/ragged/offset_r33_c5`
- `metal/f32/reduce_to_shape/ragged/offset_r3_c129`
- `metal/f32/reduce_to_shape/ragged/offset_r3_c32`
- `metal/f32/reduce_to_shape/ragged/offset_r3_c5`
- `metal/f32/reduce_to_shape/ragged/offset_r3_c65`
- `metal/f32/reduce_to_shape/short_axis/dense_r128_c129`
- `metal/f32/reduce_to_shape/short_axis/dense_r128_c32`
- `metal/f32/reduce_to_shape/short_axis/dense_r128_c5`
- `metal/f32/reduce_to_shape/short_axis/dense_r128_c65`
- `metal/f32/reduce_to_shape/short_axis/dense_r16_c129`
- `metal/f32/reduce_to_shape/short_axis/dense_r16_c32`
- `metal/f32/reduce_to_shape/short_axis/dense_r16_c5`
- `metal/f32/reduce_to_shape/short_axis/dense_r16_c65`
- `metal/f32/reduce_to_shape/short_axis/dense_r33_c129`
- `metal/f32/reduce_to_shape/short_axis/dense_r33_c32`
- `metal/f32/reduce_to_shape/short_axis/dense_r33_c5`
- `metal/f32/reduce_to_shape/short_axis/dense_r33_c65`
- `metal/f32/reduce_to_shape/short_axis/dense_r3_c129`
- `metal/f32/reduce_to_shape/short_axis/dense_r3_c32`
- `metal/f32/reduce_to_shape/short_axis/dense_r3_c5`
- `metal/f32/reduce_to_shape/short_axis/dense_r3_c65`
- `metal/f32/reduce_to_shape/short_axis/offset_r128_c129`
- `metal/f32/reduce_to_shape/short_axis/offset_r128_c32`
- `metal/f32/reduce_to_shape/short_axis/offset_r128_c5`
- `metal/f32/reduce_to_shape/short_axis/offset_r128_c65`
- `metal/f32/reduce_to_shape/short_axis/offset_r16_c129`
- `metal/f32/reduce_to_shape/short_axis/offset_r16_c32`
- `metal/f32/reduce_to_shape/short_axis/offset_r16_c5`
- `metal/f32/reduce_to_shape/short_axis/offset_r16_c65`
- `metal/f32/reduce_to_shape/short_axis/offset_r33_c129`
- `metal/f32/reduce_to_shape/short_axis/offset_r33_c32`
- `metal/f32/reduce_to_shape/short_axis/offset_r33_c5`
- `metal/f32/reduce_to_shape/short_axis/offset_r33_c65`
- `metal/f32/reduce_to_shape/short_axis/offset_r3_c129`
- `metal/f32/reduce_to_shape/short_axis/offset_r3_c32`
- `metal/f32/reduce_to_shape/short_axis/offset_r3_c5`
- `metal/f32/reduce_to_shape/short_axis/offset_r3_c65`
