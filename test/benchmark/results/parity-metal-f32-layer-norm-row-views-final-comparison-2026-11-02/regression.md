# Parity regression check

Target: `Mac16,1/26.6.2/arm64`

- Cells compared: 64
- Added cells: 0
- Removed cells: 64
- Changed case definitions: 0
- New correctness failures: 0
- Diagnostic timing suspects: 0
- Target matches: yes
- Measurement protocol matches: yes

Timing changes are diagnostic and do not fail this check. Timing is compared only when target and protocol match. No timing baseline is accepted by this comparison.

| Cell | Correctness regression | Operator relative timing | Program relative timing |
| --- | --- | ---: | ---: |
| `metal/f32/layer_norm/outer_axis/outer_offset_r128_c129` | none | 0.86× | 1.04× |
| `metal/f32/layer_norm/outer_axis/outer_offset_r128_c32` | none | 1.27× | 0.97× |
| `metal/f32/layer_norm/outer_axis/outer_offset_r128_c5` | none | 0.97× | 1.02× |
| `metal/f32/layer_norm/outer_axis/outer_offset_r128_c65` | none | 1.03× | 1.00× |
| `metal/f32/layer_norm/outer_axis/outer_offset_r16_c129` | none | 1.01× | 0.92× |
| `metal/f32/layer_norm/outer_axis/outer_offset_r16_c32` | none | 0.73× | 1.05× |
| `metal/f32/layer_norm/outer_axis/outer_offset_r16_c5` | none | 1.09× | 1.03× |
| `metal/f32/layer_norm/outer_axis/outer_offset_r16_c65` | none | 0.96× | 1.00× |
| `metal/f32/layer_norm/outer_axis/outer_offset_r33_c129` | none | 1.39× | 1.08× |
| `metal/f32/layer_norm/outer_axis/outer_offset_r33_c32` | none | 1.03× | 0.96× |
| `metal/f32/layer_norm/outer_axis/outer_offset_r33_c5` | none | 0.94× | 0.96× |
| `metal/f32/layer_norm/outer_axis/outer_offset_r33_c65` | none | 0.92× | 1.06× |
| `metal/f32/layer_norm/outer_axis/outer_offset_r3_c129` | none | 1.04× | 0.98× |
| `metal/f32/layer_norm/outer_axis/outer_offset_r3_c32` | none | 0.98× | 0.92× |
| `metal/f32/layer_norm/outer_axis/outer_offset_r3_c5` | none | 0.98× | 1.01× |
| `metal/f32/layer_norm/outer_axis/outer_offset_r3_c65` | none | 1.03× | 1.30× |
| `metal/f32/layer_norm/outer_axis/outer_strided_r128_c129` | none | 1.20× | 1.09× |
| `metal/f32/layer_norm/outer_axis/outer_strided_r128_c32` | none | 0.56× | 0.74× |
| `metal/f32/layer_norm/outer_axis/outer_strided_r128_c5` | none | 1.04× | 1.04× |
| `metal/f32/layer_norm/outer_axis/outer_strided_r128_c65` | none | 1.00× | 0.98× |
| `metal/f32/layer_norm/outer_axis/outer_strided_r16_c129` | none | 1.09× | 0.95× |
| `metal/f32/layer_norm/outer_axis/outer_strided_r16_c32` | none | 1.25× | 0.87× |
| `metal/f32/layer_norm/outer_axis/outer_strided_r16_c5` | none | 1.02× | 1.00× |
| `metal/f32/layer_norm/outer_axis/outer_strided_r16_c65` | none | 1.00× | 1.35× |
| `metal/f32/layer_norm/outer_axis/outer_strided_r33_c129` | none | 0.90× | 0.64× |
| `metal/f32/layer_norm/outer_axis/outer_strided_r33_c32` | none | 1.00× | 1.05× |
| `metal/f32/layer_norm/outer_axis/outer_strided_r33_c5` | none | 1.10× | 0.99× |
| `metal/f32/layer_norm/outer_axis/outer_strided_r33_c65` | none | 1.07× | 0.85× |
| `metal/f32/layer_norm/outer_axis/outer_strided_r3_c129` | none | 1.08× | 0.78× |
| `metal/f32/layer_norm/outer_axis/outer_strided_r3_c32` | none | 1.23× | 1.23× |
| `metal/f32/layer_norm/outer_axis/outer_strided_r3_c5` | none | 0.86× | 1.03× |
| `metal/f32/layer_norm/outer_axis/outer_strided_r3_c65` | none | 1.00× | 1.25× |
| `metal/f32/layer_norm/strided/inner_offset_r128_c129` | none | 1.02× | 0.74× |
| `metal/f32/layer_norm/strided/inner_offset_r128_c32` | none | 0.95× | 0.77× |
| `metal/f32/layer_norm/strided/inner_offset_r128_c5` | none | 0.78× | 0.81× |
| `metal/f32/layer_norm/strided/inner_offset_r128_c65` | none | 0.95× | 0.74× |
| `metal/f32/layer_norm/strided/inner_offset_r16_c129` | none | 0.99× | 0.97× |
| `metal/f32/layer_norm/strided/inner_offset_r16_c32` | none | 0.77× | 1.02× |
| `metal/f32/layer_norm/strided/inner_offset_r16_c5` | none | 0.79× | 0.99× |
| `metal/f32/layer_norm/strided/inner_offset_r16_c65` | none | 0.98× | 0.74× |
| `metal/f32/layer_norm/strided/inner_offset_r33_c129` | none | 0.98× | 0.85× |
| `metal/f32/layer_norm/strided/inner_offset_r33_c32` | none | 1.03× | 0.91× |
| `metal/f32/layer_norm/strided/inner_offset_r33_c5` | none | 0.77× | 0.84× |
| `metal/f32/layer_norm/strided/inner_offset_r33_c65` | none | 0.60× | 0.68× |
| `metal/f32/layer_norm/strided/inner_offset_r3_c129` | none | 1.05× | 1.04× |
| `metal/f32/layer_norm/strided/inner_offset_r3_c32` | none | 0.86× | 0.77× |
| `metal/f32/layer_norm/strided/inner_offset_r3_c5` | none | 0.65× | 0.98× |
| `metal/f32/layer_norm/strided/inner_offset_r3_c65` | none | 0.85× | 1.00× |
| `metal/f32/layer_norm/strided/inner_strided_r128_c129` | none | 0.75× | 1.17× |
| `metal/f32/layer_norm/strided/inner_strided_r128_c32` | none | 1.00× | 0.78× |
| `metal/f32/layer_norm/strided/inner_strided_r128_c5` | none | 0.79× | 0.85× |
| `metal/f32/layer_norm/strided/inner_strided_r128_c65` | none | 1.02× | 0.92× |
| `metal/f32/layer_norm/strided/inner_strided_r16_c129` | none | 0.92× | 0.91× |
| `metal/f32/layer_norm/strided/inner_strided_r16_c32` | none | 0.49× | 0.96× |
| `metal/f32/layer_norm/strided/inner_strided_r16_c5` | none | 0.76× | 0.83× |
| `metal/f32/layer_norm/strided/inner_strided_r16_c65` | none | 1.01× | 0.79× |
| `metal/f32/layer_norm/strided/inner_strided_r33_c129` | none | 1.15× | 0.99× |
| `metal/f32/layer_norm/strided/inner_strided_r33_c32` | none | 1.00× | 0.51× |
| `metal/f32/layer_norm/strided/inner_strided_r33_c5` | none | 1.06× | 0.75× |
| `metal/f32/layer_norm/strided/inner_strided_r33_c65` | none | 0.57× | 0.91× |
| `metal/f32/layer_norm/strided/inner_strided_r3_c129` | none | 1.06× | 0.79× |
| `metal/f32/layer_norm/strided/inner_strided_r3_c32` | none | 0.85× | 1.02× |
| `metal/f32/layer_norm/strided/inner_strided_r3_c5` | none | 0.81× | 1.04× |
| `metal/f32/layer_norm/strided/inner_strided_r3_c65` | none | 1.05× | 0.92× |

## Removed cells

- `metal/bf16/layer_norm/outer_axis/outer_offset_r128_c129`
- `metal/bf16/layer_norm/outer_axis/outer_offset_r128_c32`
- `metal/bf16/layer_norm/outer_axis/outer_offset_r128_c5`
- `metal/bf16/layer_norm/outer_axis/outer_offset_r128_c65`
- `metal/bf16/layer_norm/outer_axis/outer_offset_r16_c129`
- `metal/bf16/layer_norm/outer_axis/outer_offset_r16_c32`
- `metal/bf16/layer_norm/outer_axis/outer_offset_r16_c5`
- `metal/bf16/layer_norm/outer_axis/outer_offset_r16_c65`
- `metal/bf16/layer_norm/outer_axis/outer_offset_r33_c129`
- `metal/bf16/layer_norm/outer_axis/outer_offset_r33_c32`
- `metal/bf16/layer_norm/outer_axis/outer_offset_r33_c5`
- `metal/bf16/layer_norm/outer_axis/outer_offset_r33_c65`
- `metal/bf16/layer_norm/outer_axis/outer_offset_r3_c129`
- `metal/bf16/layer_norm/outer_axis/outer_offset_r3_c32`
- `metal/bf16/layer_norm/outer_axis/outer_offset_r3_c5`
- `metal/bf16/layer_norm/outer_axis/outer_offset_r3_c65`
- `metal/bf16/layer_norm/outer_axis/outer_strided_r128_c129`
- `metal/bf16/layer_norm/outer_axis/outer_strided_r128_c32`
- `metal/bf16/layer_norm/outer_axis/outer_strided_r128_c5`
- `metal/bf16/layer_norm/outer_axis/outer_strided_r128_c65`
- `metal/bf16/layer_norm/outer_axis/outer_strided_r16_c129`
- `metal/bf16/layer_norm/outer_axis/outer_strided_r16_c32`
- `metal/bf16/layer_norm/outer_axis/outer_strided_r16_c5`
- `metal/bf16/layer_norm/outer_axis/outer_strided_r16_c65`
- `metal/bf16/layer_norm/outer_axis/outer_strided_r33_c129`
- `metal/bf16/layer_norm/outer_axis/outer_strided_r33_c32`
- `metal/bf16/layer_norm/outer_axis/outer_strided_r33_c5`
- `metal/bf16/layer_norm/outer_axis/outer_strided_r33_c65`
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
