# Parity regression check

Target: `Mac16,1/26.6.2/arm64`

- Cells compared: 64
- Added cells: 0
- Removed cells: 64
- Changed case definitions: 0
- New correctness failures: 0
- Diagnostic timing suspects: 1
- Target matches: yes
- Measurement protocol matches: yes

Timing changes are diagnostic and do not fail this check. Timing is compared only when target and protocol match. No timing baseline is accepted by this comparison.

| Cell | Correctness regression | Operator relative timing | Program relative timing |
| --- | --- | ---: | ---: |
| `metal/bf16/layer_norm/outer_axis/outer_offset_r128_c129` | none | 0.02× | 0.02× |
| `metal/bf16/layer_norm/outer_axis/outer_offset_r128_c32` | none | 0.05× | 0.05× |
| `metal/bf16/layer_norm/outer_axis/outer_offset_r128_c5` | none | 0.22× | 0.24× |
| `metal/bf16/layer_norm/outer_axis/outer_offset_r128_c65` | none | 0.03× | 0.03× |
| `metal/bf16/layer_norm/outer_axis/outer_offset_r16_c129` | none | 0.13× | 0.13× |
| `metal/bf16/layer_norm/outer_axis/outer_offset_r16_c32` | none | 0.26× | 0.28× |
| `metal/bf16/layer_norm/outer_axis/outer_offset_r16_c5` | none | 0.77× | 0.89× |
| `metal/bf16/layer_norm/outer_axis/outer_offset_r16_c65` | none | 0.19× | 0.21× |
| `metal/bf16/layer_norm/outer_axis/outer_offset_r33_c129` | none | 0.09× | 0.08× |
| `metal/bf16/layer_norm/outer_axis/outer_offset_r33_c32` | none | 0.18× | 0.19× |
| `metal/bf16/layer_norm/outer_axis/outer_offset_r33_c5` | none | 0.63× | 0.64× |
| `metal/bf16/layer_norm/outer_axis/outer_offset_r33_c65` | none | 0.12× | 0.08× |
| `metal/bf16/layer_norm/outer_axis/outer_offset_r3_c129` | none | 0.36× | 0.39× |
| `metal/bf16/layer_norm/outer_axis/outer_offset_r3_c32` | none | 0.77× | 0.77× |
| `metal/bf16/layer_norm/outer_axis/outer_offset_r3_c5` | none | 1.52× | 1.59× |
| `metal/bf16/layer_norm/outer_axis/outer_offset_r3_c65` | none | 0.53× | 0.45× |
| `metal/bf16/layer_norm/outer_axis/outer_strided_r128_c129` | none | 0.02× | 0.02× |
| `metal/bf16/layer_norm/outer_axis/outer_strided_r128_c32` | none | 0.05× | 0.05× |
| `metal/bf16/layer_norm/outer_axis/outer_strided_r128_c5` | none | 0.22× | 0.24× |
| `metal/bf16/layer_norm/outer_axis/outer_strided_r128_c65` | none | 0.03× | 0.03× |
| `metal/bf16/layer_norm/outer_axis/outer_strided_r16_c129` | none | 0.13× | 0.15× |
| `metal/bf16/layer_norm/outer_axis/outer_strided_r16_c32` | none | 0.30× | 0.32× |
| `metal/bf16/layer_norm/outer_axis/outer_strided_r16_c5` | none | 0.75× | 0.80× |
| `metal/bf16/layer_norm/outer_axis/outer_strided_r16_c65` | none | 0.22× | 0.21× |
| `metal/bf16/layer_norm/outer_axis/outer_strided_r33_c129` | none | 0.05× | 0.08× |
| `metal/bf16/layer_norm/outer_axis/outer_strided_r33_c32` | none | 0.17× | 0.19× |
| `metal/bf16/layer_norm/outer_axis/outer_strided_r33_c5` | none | 0.54× | 0.57× |
| `metal/bf16/layer_norm/outer_axis/outer_strided_r33_c65` | none | 0.12× | 0.12× |
| `metal/bf16/layer_norm/outer_axis/outer_strided_r3_c129` | none | 0.36× | 0.40× |
| `metal/bf16/layer_norm/outer_axis/outer_strided_r3_c32` | none | 0.78× | 0.84× |
| `metal/bf16/layer_norm/outer_axis/outer_strided_r3_c5` | none | 1.09× | 1.36× |
| `metal/bf16/layer_norm/outer_axis/outer_strided_r3_c65` | none | 0.58× | 0.55× |
| `metal/bf16/layer_norm/strided/inner_offset_r128_c129` | none | 0.57× | 0.61× |
| `metal/bf16/layer_norm/strided/inner_offset_r128_c32` | none | 0.59× | 1.07× |
| `metal/bf16/layer_norm/strided/inner_offset_r128_c5` | none | 0.87× | 1.08× |
| `metal/bf16/layer_norm/strided/inner_offset_r128_c65` | none | 0.63× | 0.82× |
| `metal/bf16/layer_norm/strided/inner_offset_r16_c129` | none | 0.49× | 0.56× |
| `metal/bf16/layer_norm/strided/inner_offset_r16_c32` | none | 0.62× | 0.48× |
| `metal/bf16/layer_norm/strided/inner_offset_r16_c5` | none | 1.01× | 0.78× |
| `metal/bf16/layer_norm/strided/inner_offset_r16_c65` | none | 0.68× | 0.63× |
| `metal/bf16/layer_norm/strided/inner_offset_r33_c129` | none | 0.44× | 0.55× |
| `metal/bf16/layer_norm/strided/inner_offset_r33_c32` | none | 0.82× | 1.02× |
| `metal/bf16/layer_norm/strided/inner_offset_r33_c5` | none | 0.78× | 0.80× |
| `metal/bf16/layer_norm/strided/inner_offset_r33_c65` | none | 0.45× | 0.66× |
| `metal/bf16/layer_norm/strided/inner_offset_r3_c129` | none | 0.51× | 0.55× |
| `metal/bf16/layer_norm/strided/inner_offset_r3_c32` | none | 0.69× | 0.51× |
| `metal/bf16/layer_norm/strided/inner_offset_r3_c5` | none | 0.83× | 0.89× |
| `metal/bf16/layer_norm/strided/inner_offset_r3_c65` | none | 0.72× | 0.71× |
| `metal/bf16/layer_norm/strided/inner_strided_r128_c129` | none | 0.58× | 0.67× |
| `metal/bf16/layer_norm/strided/inner_strided_r128_c32` | none | 0.90× | 1.03× |
| `metal/bf16/layer_norm/strided/inner_strided_r128_c5` | none | 0.79× | 0.89× |
| `metal/bf16/layer_norm/strided/inner_strided_r128_c65` | none | 0.61× | 0.76× |
| `metal/bf16/layer_norm/strided/inner_strided_r16_c129` | none | 0.49× | 0.58× |
| `metal/bf16/layer_norm/strided/inner_strided_r16_c32` | none | 0.87× | 1.08× |
| `metal/bf16/layer_norm/strided/inner_strided_r16_c5` | none | 1.05× | 0.79× |
| `metal/bf16/layer_norm/strided/inner_strided_r16_c65` | none | 0.68× | 0.75× |
| `metal/bf16/layer_norm/strided/inner_strided_r33_c129` | none | 0.57× | 0.53× |
| `metal/bf16/layer_norm/strided/inner_strided_r33_c32` | none | 0.44× | 0.64× |
| `metal/bf16/layer_norm/strided/inner_strided_r33_c5` | none | 0.82× | 1.01× |
| `metal/bf16/layer_norm/strided/inner_strided_r33_c65` | none | 0.68× | 0.75× |
| `metal/bf16/layer_norm/strided/inner_strided_r3_c129` | none | 0.52× | 0.55× |
| `metal/bf16/layer_norm/strided/inner_strided_r3_c32` | none | 0.86× | 0.63× |
| `metal/bf16/layer_norm/strided/inner_strided_r3_c5` | none | 0.72× | 0.88× |
| `metal/bf16/layer_norm/strided/inner_strided_r3_c65` | none | 0.76× | 0.65× |

## Removed cells

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
