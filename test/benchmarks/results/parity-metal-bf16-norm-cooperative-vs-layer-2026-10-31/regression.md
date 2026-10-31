# Parity regression check

Target: `Mac16,1/26.6.2/arm64`

- Cells compared: 16
- Added cells: 52
- Removed cells: 112
- Changed case definitions: 0
- New correctness failures: 0
- Diagnostic timing suspects: 0
- Target matches: yes
- Measurement protocol matches: yes

Timing changes are diagnostic and do not fail this check. Timing is compared only when target and protocol match. No timing baseline is accepted by this comparison.

| Cell | Correctness regression | Operator relative timing | Program relative timing |
| --- | --- | ---: | ---: |
| `metal/bf16/layer_norm/outer_axis/outer_offset_r128_c129` | none | 0.02× | 0.02× |
| `metal/bf16/layer_norm/outer_axis/outer_offset_r128_c32` | none | 0.08× | 0.10× |
| `metal/bf16/layer_norm/outer_axis/outer_offset_r128_c65` | none | 0.03× | 0.03× |
| `metal/bf16/layer_norm/outer_axis/outer_offset_r16_c129` | none | 0.12× | 0.12× |
| `metal/bf16/layer_norm/outer_axis/outer_offset_r16_c65` | none | 0.19× | 0.20× |
| `metal/bf16/layer_norm/outer_axis/outer_offset_r33_c129` | none | 0.08× | 0.08× |
| `metal/bf16/layer_norm/outer_axis/outer_offset_r33_c32` | none | 0.18× | 0.14× |
| `metal/bf16/layer_norm/outer_axis/outer_offset_r33_c65` | none | 0.12× | 0.13× |
| `metal/bf16/layer_norm/outer_axis/outer_strided_r128_c129` | none | 0.02× | 0.02× |
| `metal/bf16/layer_norm/outer_axis/outer_strided_r128_c32` | none | 0.06× | 0.07× |
| `metal/bf16/layer_norm/outer_axis/outer_strided_r128_c65` | none | 0.03× | 0.06× |
| `metal/bf16/layer_norm/outer_axis/outer_strided_r16_c129` | none | 0.12× | 0.14× |
| `metal/bf16/layer_norm/outer_axis/outer_strided_r16_c65` | none | 0.19× | 0.21× |
| `metal/bf16/layer_norm/outer_axis/outer_strided_r33_c129` | none | 0.06× | 0.05× |
| `metal/bf16/layer_norm/outer_axis/outer_strided_r33_c32` | none | 0.18× | 0.19× |
| `metal/bf16/layer_norm/outer_axis/outer_strided_r33_c65` | none | 0.09× | 0.14× |

## Added cells

- `metal/bf16/layer_norm/outer_axis/outer_offset_r129_c127`
- `metal/bf16/layer_norm/outer_axis/outer_offset_r129_c17`
- `metal/bf16/layer_norm/outer_axis/outer_offset_r129_c257`
- `metal/bf16/layer_norm/outer_axis/outer_offset_r129_c63`
- `metal/bf16/layer_norm/outer_axis/outer_offset_r64_c127`
- `metal/bf16/layer_norm/outer_axis/outer_offset_r64_c17`
- `metal/bf16/layer_norm/outer_axis/outer_offset_r64_c257`
- `metal/bf16/layer_norm/outer_axis/outer_offset_r64_c63`
- `metal/bf16/layer_norm/outer_axis/outer_offset_r7_c257`
- `metal/bf16/layer_norm/outer_axis/outer_strided_r129_c127`
- `metal/bf16/layer_norm/outer_axis/outer_strided_r129_c17`
- `metal/bf16/layer_norm/outer_axis/outer_strided_r129_c257`
- `metal/bf16/layer_norm/outer_axis/outer_strided_r129_c63`
- `metal/bf16/layer_norm/outer_axis/outer_strided_r64_c127`
- `metal/bf16/layer_norm/outer_axis/outer_strided_r64_c17`
- `metal/bf16/layer_norm/outer_axis/outer_strided_r64_c257`
- `metal/bf16/layer_norm/outer_axis/outer_strided_r64_c63`
- `metal/bf16/layer_norm/outer_axis/outer_strided_r7_c257`
- `metal/bf16/rms_norm/outer_axis/offset_r128_c129`
- `metal/bf16/rms_norm/outer_axis/offset_r128_c32`
- `metal/bf16/rms_norm/outer_axis/offset_r128_c65`
- `metal/bf16/rms_norm/outer_axis/offset_r129_c127`
- `metal/bf16/rms_norm/outer_axis/offset_r129_c17`
- `metal/bf16/rms_norm/outer_axis/offset_r129_c257`
- `metal/bf16/rms_norm/outer_axis/offset_r129_c63`
- `metal/bf16/rms_norm/outer_axis/offset_r16_c129`
- `metal/bf16/rms_norm/outer_axis/offset_r16_c65`
- `metal/bf16/rms_norm/outer_axis/offset_r33_c129`
- `metal/bf16/rms_norm/outer_axis/offset_r33_c32`
- `metal/bf16/rms_norm/outer_axis/offset_r33_c65`
- `metal/bf16/rms_norm/outer_axis/offset_r64_c127`
- `metal/bf16/rms_norm/outer_axis/offset_r64_c17`
- `metal/bf16/rms_norm/outer_axis/offset_r64_c257`
- `metal/bf16/rms_norm/outer_axis/offset_r64_c63`
- `metal/bf16/rms_norm/outer_axis/offset_r7_c257`
- `metal/bf16/rms_norm/outer_axis/strided_r128_c129`
- `metal/bf16/rms_norm/outer_axis/strided_r128_c32`
- `metal/bf16/rms_norm/outer_axis/strided_r128_c65`
- `metal/bf16/rms_norm/outer_axis/strided_r129_c127`
- `metal/bf16/rms_norm/outer_axis/strided_r129_c17`
- `metal/bf16/rms_norm/outer_axis/strided_r129_c257`
- `metal/bf16/rms_norm/outer_axis/strided_r129_c63`
- `metal/bf16/rms_norm/outer_axis/strided_r16_c129`
- `metal/bf16/rms_norm/outer_axis/strided_r16_c65`
- `metal/bf16/rms_norm/outer_axis/strided_r33_c129`
- `metal/bf16/rms_norm/outer_axis/strided_r33_c32`
- `metal/bf16/rms_norm/outer_axis/strided_r33_c65`
- `metal/bf16/rms_norm/outer_axis/strided_r64_c127`
- `metal/bf16/rms_norm/outer_axis/strided_r64_c17`
- `metal/bf16/rms_norm/outer_axis/strided_r64_c257`
- `metal/bf16/rms_norm/outer_axis/strided_r64_c63`
- `metal/bf16/rms_norm/outer_axis/strided_r7_c257`

## Removed cells

- `metal/bf16/layer_norm/outer_axis/outer_offset_r128_c5`
- `metal/bf16/layer_norm/outer_axis/outer_offset_r16_c32`
- `metal/bf16/layer_norm/outer_axis/outer_offset_r16_c5`
- `metal/bf16/layer_norm/outer_axis/outer_offset_r33_c5`
- `metal/bf16/layer_norm/outer_axis/outer_offset_r3_c129`
- `metal/bf16/layer_norm/outer_axis/outer_offset_r3_c32`
- `metal/bf16/layer_norm/outer_axis/outer_offset_r3_c5`
- `metal/bf16/layer_norm/outer_axis/outer_offset_r3_c65`
- `metal/bf16/layer_norm/outer_axis/outer_strided_r128_c5`
- `metal/bf16/layer_norm/outer_axis/outer_strided_r16_c32`
- `metal/bf16/layer_norm/outer_axis/outer_strided_r16_c5`
- `metal/bf16/layer_norm/outer_axis/outer_strided_r33_c5`
- `metal/bf16/layer_norm/outer_axis/outer_strided_r3_c129`
- `metal/bf16/layer_norm/outer_axis/outer_strided_r3_c32`
- `metal/bf16/layer_norm/outer_axis/outer_strided_r3_c5`
- `metal/bf16/layer_norm/outer_axis/outer_strided_r3_c65`
- `metal/bf16/layer_norm/strided/inner_offset_r128_c129`
- `metal/bf16/layer_norm/strided/inner_offset_r128_c32`
- `metal/bf16/layer_norm/strided/inner_offset_r128_c5`
- `metal/bf16/layer_norm/strided/inner_offset_r128_c65`
- `metal/bf16/layer_norm/strided/inner_offset_r16_c129`
- `metal/bf16/layer_norm/strided/inner_offset_r16_c32`
- `metal/bf16/layer_norm/strided/inner_offset_r16_c5`
- `metal/bf16/layer_norm/strided/inner_offset_r16_c65`
- `metal/bf16/layer_norm/strided/inner_offset_r33_c129`
- `metal/bf16/layer_norm/strided/inner_offset_r33_c32`
- `metal/bf16/layer_norm/strided/inner_offset_r33_c5`
- `metal/bf16/layer_norm/strided/inner_offset_r33_c65`
- `metal/bf16/layer_norm/strided/inner_offset_r3_c129`
- `metal/bf16/layer_norm/strided/inner_offset_r3_c32`
- `metal/bf16/layer_norm/strided/inner_offset_r3_c5`
- `metal/bf16/layer_norm/strided/inner_offset_r3_c65`
- `metal/bf16/layer_norm/strided/inner_strided_r128_c129`
- `metal/bf16/layer_norm/strided/inner_strided_r128_c32`
- `metal/bf16/layer_norm/strided/inner_strided_r128_c5`
- `metal/bf16/layer_norm/strided/inner_strided_r128_c65`
- `metal/bf16/layer_norm/strided/inner_strided_r16_c129`
- `metal/bf16/layer_norm/strided/inner_strided_r16_c32`
- `metal/bf16/layer_norm/strided/inner_strided_r16_c5`
- `metal/bf16/layer_norm/strided/inner_strided_r16_c65`
- `metal/bf16/layer_norm/strided/inner_strided_r33_c129`
- `metal/bf16/layer_norm/strided/inner_strided_r33_c32`
- `metal/bf16/layer_norm/strided/inner_strided_r33_c5`
- `metal/bf16/layer_norm/strided/inner_strided_r33_c65`
- `metal/bf16/layer_norm/strided/inner_strided_r3_c129`
- `metal/bf16/layer_norm/strided/inner_strided_r3_c32`
- `metal/bf16/layer_norm/strided/inner_strided_r3_c5`
- `metal/bf16/layer_norm/strided/inner_strided_r3_c65`
- `metal/f32/layer_norm/outer_axis/outer_offset_r128_c129`
- `metal/f32/layer_norm/outer_axis/outer_offset_r128_c32`
- `metal/f32/layer_norm/outer_axis/outer_offset_r128_c5`
- `metal/f32/layer_norm/outer_axis/outer_offset_r128_c65`
- `metal/f32/layer_norm/outer_axis/outer_offset_r16_c129`
- `metal/f32/layer_norm/outer_axis/outer_offset_r16_c32`
- `metal/f32/layer_norm/outer_axis/outer_offset_r16_c5`
- `metal/f32/layer_norm/outer_axis/outer_offset_r16_c65`
- `metal/f32/layer_norm/outer_axis/outer_offset_r33_c129`
- `metal/f32/layer_norm/outer_axis/outer_offset_r33_c32`
- `metal/f32/layer_norm/outer_axis/outer_offset_r33_c5`
- `metal/f32/layer_norm/outer_axis/outer_offset_r33_c65`
- `metal/f32/layer_norm/outer_axis/outer_offset_r3_c129`
- `metal/f32/layer_norm/outer_axis/outer_offset_r3_c32`
- `metal/f32/layer_norm/outer_axis/outer_offset_r3_c5`
- `metal/f32/layer_norm/outer_axis/outer_offset_r3_c65`
- `metal/f32/layer_norm/outer_axis/outer_strided_r128_c129`
- `metal/f32/layer_norm/outer_axis/outer_strided_r128_c32`
- `metal/f32/layer_norm/outer_axis/outer_strided_r128_c5`
- `metal/f32/layer_norm/outer_axis/outer_strided_r128_c65`
- `metal/f32/layer_norm/outer_axis/outer_strided_r16_c129`
- `metal/f32/layer_norm/outer_axis/outer_strided_r16_c32`
- `metal/f32/layer_norm/outer_axis/outer_strided_r16_c5`
- `metal/f32/layer_norm/outer_axis/outer_strided_r16_c65`
- `metal/f32/layer_norm/outer_axis/outer_strided_r33_c129`
- `metal/f32/layer_norm/outer_axis/outer_strided_r33_c32`
- `metal/f32/layer_norm/outer_axis/outer_strided_r33_c5`
- `metal/f32/layer_norm/outer_axis/outer_strided_r33_c65`
- `metal/f32/layer_norm/outer_axis/outer_strided_r3_c129`
- `metal/f32/layer_norm/outer_axis/outer_strided_r3_c32`
- `metal/f32/layer_norm/outer_axis/outer_strided_r3_c5`
- `metal/f32/layer_norm/outer_axis/outer_strided_r3_c65`
- `metal/f32/layer_norm/strided/inner_offset_r128_c129`
- `metal/f32/layer_norm/strided/inner_offset_r128_c32`
- `metal/f32/layer_norm/strided/inner_offset_r128_c5`
- `metal/f32/layer_norm/strided/inner_offset_r128_c65`
- `metal/f32/layer_norm/strided/inner_offset_r16_c129`
- `metal/f32/layer_norm/strided/inner_offset_r16_c32`
- `metal/f32/layer_norm/strided/inner_offset_r16_c5`
- `metal/f32/layer_norm/strided/inner_offset_r16_c65`
- `metal/f32/layer_norm/strided/inner_offset_r33_c129`
- `metal/f32/layer_norm/strided/inner_offset_r33_c32`
- `metal/f32/layer_norm/strided/inner_offset_r33_c5`
- `metal/f32/layer_norm/strided/inner_offset_r33_c65`
- `metal/f32/layer_norm/strided/inner_offset_r3_c129`
- `metal/f32/layer_norm/strided/inner_offset_r3_c32`
- `metal/f32/layer_norm/strided/inner_offset_r3_c5`
- `metal/f32/layer_norm/strided/inner_offset_r3_c65`
- `metal/f32/layer_norm/strided/inner_strided_r128_c129`
- `metal/f32/layer_norm/strided/inner_strided_r128_c32`
- `metal/f32/layer_norm/strided/inner_strided_r128_c5`
- `metal/f32/layer_norm/strided/inner_strided_r128_c65`
- `metal/f32/layer_norm/strided/inner_strided_r16_c129`
- `metal/f32/layer_norm/strided/inner_strided_r16_c32`
- `metal/f32/layer_norm/strided/inner_strided_r16_c5`
- `metal/f32/layer_norm/strided/inner_strided_r16_c65`
- `metal/f32/layer_norm/strided/inner_strided_r33_c129`
- `metal/f32/layer_norm/strided/inner_strided_r33_c32`
- `metal/f32/layer_norm/strided/inner_strided_r33_c5`
- `metal/f32/layer_norm/strided/inner_strided_r33_c65`
- `metal/f32/layer_norm/strided/inner_strided_r3_c129`
- `metal/f32/layer_norm/strided/inner_strided_r3_c32`
- `metal/f32/layer_norm/strided/inner_strided_r3_c5`
- `metal/f32/layer_norm/strided/inner_strided_r3_c65`
