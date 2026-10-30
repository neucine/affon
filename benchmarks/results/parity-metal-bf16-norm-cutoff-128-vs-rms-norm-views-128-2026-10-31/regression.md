# Parity regression check

Target: `Mac16,1/26.6.2/arm64`

- Cells compared: 12
- Added cells: 17
- Removed cells: 116
- Changed case definitions: 0
- New correctness failures: 0
- Diagnostic timing suspects: 0
- Target matches: yes
- Measurement protocol matches: no

Timing changes are diagnostic and do not fail this check. Timing is compared only when target and protocol match. No timing baseline is accepted by this comparison.

| Cell | Correctness regression | Operator relative timing | Program relative timing |
| --- | --- | ---: | ---: |
| `metal/bf16/rms_norm/long_row/offset_r128_c129` | none | — | — |
| `metal/bf16/rms_norm/long_row/offset_r16_c129` | none | — | — |
| `metal/bf16/rms_norm/long_row/offset_r33_c129` | none | — | — |
| `metal/bf16/rms_norm/long_row/offset_r3_c129` | none | — | — |
| `metal/bf16/rms_norm/outer_axis/offset_r33_c5` | none | — | — |
| `metal/bf16/rms_norm/outer_axis/offset_r3_c65` | none | — | — |
| `metal/bf16/rms_norm/outer_axis/strided_r33_c5` | none | — | — |
| `metal/bf16/rms_norm/outer_axis/strided_r3_c65` | none | — | — |
| `metal/bf16/rms_norm/strided/strided_r128_c129` | none | — | — |
| `metal/bf16/rms_norm/strided/strided_r16_c129` | none | — | — |
| `metal/bf16/rms_norm/strided/strided_r33_c129` | none | — | — |
| `metal/bf16/rms_norm/strided/strided_r3_c129` | none | — | — |

## Added cells

- `metal/bf16/layer_norm/outer_axis/outer_offset_r2_c127`
- `metal/bf16/layer_norm/outer_axis/outer_offset_r33_c5`
- `metal/bf16/layer_norm/outer_axis/outer_offset_r3_c65`
- `metal/bf16/layer_norm/outer_axis/outer_strided_r2_c127`
- `metal/bf16/layer_norm/outer_axis/outer_strided_r33_c5`
- `metal/bf16/layer_norm/outer_axis/outer_strided_r3_c65`
- `metal/bf16/layer_norm/strided/inner_offset_r128_c129`
- `metal/bf16/layer_norm/strided/inner_offset_r16_c129`
- `metal/bf16/layer_norm/strided/inner_offset_r33_c129`
- `metal/bf16/layer_norm/strided/inner_offset_r3_c129`
- `metal/bf16/layer_norm/strided/inner_strided_r128_c129`
- `metal/bf16/layer_norm/strided/inner_strided_r16_c129`
- `metal/bf16/layer_norm/strided/inner_strided_r33_c129`
- `metal/bf16/layer_norm/strided/inner_strided_r3_c129`
- `metal/bf16/rms_norm/long_row/r64_c255`
- `metal/bf16/rms_norm/outer_axis/offset_r2_c127`
- `metal/bf16/rms_norm/outer_axis/strided_r2_c127`

## Removed cells

- `metal/bf16/rms_norm/long_row/offset_r128_c32`
- `metal/bf16/rms_norm/long_row/offset_r128_c5`
- `metal/bf16/rms_norm/long_row/offset_r128_c65`
- `metal/bf16/rms_norm/long_row/offset_r16_c32`
- `metal/bf16/rms_norm/long_row/offset_r16_c5`
- `metal/bf16/rms_norm/long_row/offset_r16_c65`
- `metal/bf16/rms_norm/long_row/offset_r33_c32`
- `metal/bf16/rms_norm/long_row/offset_r33_c5`
- `metal/bf16/rms_norm/long_row/offset_r33_c65`
- `metal/bf16/rms_norm/long_row/offset_r3_c32`
- `metal/bf16/rms_norm/long_row/offset_r3_c5`
- `metal/bf16/rms_norm/long_row/offset_r3_c65`
- `metal/bf16/rms_norm/outer_axis/offset_r128_c129`
- `metal/bf16/rms_norm/outer_axis/offset_r128_c32`
- `metal/bf16/rms_norm/outer_axis/offset_r128_c5`
- `metal/bf16/rms_norm/outer_axis/offset_r128_c65`
- `metal/bf16/rms_norm/outer_axis/offset_r16_c129`
- `metal/bf16/rms_norm/outer_axis/offset_r16_c32`
- `metal/bf16/rms_norm/outer_axis/offset_r16_c5`
- `metal/bf16/rms_norm/outer_axis/offset_r16_c65`
- `metal/bf16/rms_norm/outer_axis/offset_r33_c129`
- `metal/bf16/rms_norm/outer_axis/offset_r33_c32`
- `metal/bf16/rms_norm/outer_axis/offset_r33_c65`
- `metal/bf16/rms_norm/outer_axis/offset_r3_c129`
- `metal/bf16/rms_norm/outer_axis/offset_r3_c32`
- `metal/bf16/rms_norm/outer_axis/offset_r3_c5`
- `metal/bf16/rms_norm/outer_axis/strided_r128_c129`
- `metal/bf16/rms_norm/outer_axis/strided_r128_c32`
- `metal/bf16/rms_norm/outer_axis/strided_r128_c5`
- `metal/bf16/rms_norm/outer_axis/strided_r128_c65`
- `metal/bf16/rms_norm/outer_axis/strided_r16_c129`
- `metal/bf16/rms_norm/outer_axis/strided_r16_c32`
- `metal/bf16/rms_norm/outer_axis/strided_r16_c5`
- `metal/bf16/rms_norm/outer_axis/strided_r16_c65`
- `metal/bf16/rms_norm/outer_axis/strided_r33_c129`
- `metal/bf16/rms_norm/outer_axis/strided_r33_c32`
- `metal/bf16/rms_norm/outer_axis/strided_r33_c65`
- `metal/bf16/rms_norm/outer_axis/strided_r3_c129`
- `metal/bf16/rms_norm/outer_axis/strided_r3_c32`
- `metal/bf16/rms_norm/outer_axis/strided_r3_c5`
- `metal/bf16/rms_norm/strided/strided_r128_c32`
- `metal/bf16/rms_norm/strided/strided_r128_c5`
- `metal/bf16/rms_norm/strided/strided_r128_c65`
- `metal/bf16/rms_norm/strided/strided_r16_c32`
- `metal/bf16/rms_norm/strided/strided_r16_c5`
- `metal/bf16/rms_norm/strided/strided_r16_c65`
- `metal/bf16/rms_norm/strided/strided_r33_c32`
- `metal/bf16/rms_norm/strided/strided_r33_c5`
- `metal/bf16/rms_norm/strided/strided_r33_c65`
- `metal/bf16/rms_norm/strided/strided_r3_c32`
- `metal/bf16/rms_norm/strided/strided_r3_c5`
- `metal/bf16/rms_norm/strided/strided_r3_c65`
- `metal/f32/rms_norm/long_row/offset_r128_c129`
- `metal/f32/rms_norm/long_row/offset_r128_c32`
- `metal/f32/rms_norm/long_row/offset_r128_c5`
- `metal/f32/rms_norm/long_row/offset_r128_c65`
- `metal/f32/rms_norm/long_row/offset_r16_c129`
- `metal/f32/rms_norm/long_row/offset_r16_c32`
- `metal/f32/rms_norm/long_row/offset_r16_c5`
- `metal/f32/rms_norm/long_row/offset_r16_c65`
- `metal/f32/rms_norm/long_row/offset_r33_c129`
- `metal/f32/rms_norm/long_row/offset_r33_c32`
- `metal/f32/rms_norm/long_row/offset_r33_c5`
- `metal/f32/rms_norm/long_row/offset_r33_c65`
- `metal/f32/rms_norm/long_row/offset_r3_c129`
- `metal/f32/rms_norm/long_row/offset_r3_c32`
- `metal/f32/rms_norm/long_row/offset_r3_c5`
- `metal/f32/rms_norm/long_row/offset_r3_c65`
- `metal/f32/rms_norm/outer_axis/offset_r128_c129`
- `metal/f32/rms_norm/outer_axis/offset_r128_c32`
- `metal/f32/rms_norm/outer_axis/offset_r128_c5`
- `metal/f32/rms_norm/outer_axis/offset_r128_c65`
- `metal/f32/rms_norm/outer_axis/offset_r16_c129`
- `metal/f32/rms_norm/outer_axis/offset_r16_c32`
- `metal/f32/rms_norm/outer_axis/offset_r16_c5`
- `metal/f32/rms_norm/outer_axis/offset_r16_c65`
- `metal/f32/rms_norm/outer_axis/offset_r33_c129`
- `metal/f32/rms_norm/outer_axis/offset_r33_c32`
- `metal/f32/rms_norm/outer_axis/offset_r33_c5`
- `metal/f32/rms_norm/outer_axis/offset_r33_c65`
- `metal/f32/rms_norm/outer_axis/offset_r3_c129`
- `metal/f32/rms_norm/outer_axis/offset_r3_c32`
- `metal/f32/rms_norm/outer_axis/offset_r3_c5`
- `metal/f32/rms_norm/outer_axis/offset_r3_c65`
- `metal/f32/rms_norm/outer_axis/strided_r128_c129`
- `metal/f32/rms_norm/outer_axis/strided_r128_c32`
- `metal/f32/rms_norm/outer_axis/strided_r128_c5`
- `metal/f32/rms_norm/outer_axis/strided_r128_c65`
- `metal/f32/rms_norm/outer_axis/strided_r16_c129`
- `metal/f32/rms_norm/outer_axis/strided_r16_c32`
- `metal/f32/rms_norm/outer_axis/strided_r16_c5`
- `metal/f32/rms_norm/outer_axis/strided_r16_c65`
- `metal/f32/rms_norm/outer_axis/strided_r33_c129`
- `metal/f32/rms_norm/outer_axis/strided_r33_c32`
- `metal/f32/rms_norm/outer_axis/strided_r33_c5`
- `metal/f32/rms_norm/outer_axis/strided_r33_c65`
- `metal/f32/rms_norm/outer_axis/strided_r3_c129`
- `metal/f32/rms_norm/outer_axis/strided_r3_c32`
- `metal/f32/rms_norm/outer_axis/strided_r3_c5`
- `metal/f32/rms_norm/outer_axis/strided_r3_c65`
- `metal/f32/rms_norm/strided/strided_r128_c129`
- `metal/f32/rms_norm/strided/strided_r128_c32`
- `metal/f32/rms_norm/strided/strided_r128_c5`
- `metal/f32/rms_norm/strided/strided_r128_c65`
- `metal/f32/rms_norm/strided/strided_r16_c129`
- `metal/f32/rms_norm/strided/strided_r16_c32`
- `metal/f32/rms_norm/strided/strided_r16_c5`
- `metal/f32/rms_norm/strided/strided_r16_c65`
- `metal/f32/rms_norm/strided/strided_r33_c129`
- `metal/f32/rms_norm/strided/strided_r33_c32`
- `metal/f32/rms_norm/strided/strided_r33_c5`
- `metal/f32/rms_norm/strided/strided_r33_c65`
- `metal/f32/rms_norm/strided/strided_r3_c129`
- `metal/f32/rms_norm/strided/strided_r3_c32`
- `metal/f32/rms_norm/strided/strided_r3_c5`
- `metal/f32/rms_norm/strided/strided_r3_c65`
