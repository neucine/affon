# Parity regression check

Target: `Mac16,1/26.6.2/arm64`

- Cells compared: 64
- Added cells: 0
- Removed cells: 0
- Changed case definitions: 0
- New correctness failures: 0
- Diagnostic timing suspects: 0
- Target matches: yes
- Measurement protocol matches: yes

Timing changes are diagnostic and do not fail this check. Timing is compared only when target and protocol match. No timing baseline is accepted by this comparison.

| Cell | Correctness regression | Operator relative timing | Program relative timing |
| --- | --- | ---: | ---: |
| `metal/f32/layer_norm/outer_axis/outer_offset_r128_c129` | none | 1.04× | 0.93× |
| `metal/f32/layer_norm/outer_axis/outer_offset_r128_c32` | none | 0.77× | 1.10× |
| `metal/f32/layer_norm/outer_axis/outer_offset_r128_c5` | none | 0.88× | 1.03× |
| `metal/f32/layer_norm/outer_axis/outer_offset_r128_c65` | none | 0.96× | 1.01× |
| `metal/f32/layer_norm/outer_axis/outer_offset_r16_c129` | none | 1.16× | 0.98× |
| `metal/f32/layer_norm/outer_axis/outer_offset_r16_c32` | none | 1.31× | 1.00× |
| `metal/f32/layer_norm/outer_axis/outer_offset_r16_c5` | none | 1.02× | 1.01× |
| `metal/f32/layer_norm/outer_axis/outer_offset_r16_c65` | none | 0.82× | 1.07× |
| `metal/f32/layer_norm/outer_axis/outer_offset_r33_c129` | none | 0.71× | 0.91× |
| `metal/f32/layer_norm/outer_axis/outer_offset_r33_c32` | none | 1.07× | 1.01× |
| `metal/f32/layer_norm/outer_axis/outer_offset_r33_c5` | none | 1.08× | 0.85× |
| `metal/f32/layer_norm/outer_axis/outer_offset_r33_c65` | none | 1.07× | 0.79× |
| `metal/f32/layer_norm/outer_axis/outer_offset_r3_c129` | none | 1.01× | 1.02× |
| `metal/f32/layer_norm/outer_axis/outer_offset_r3_c32` | none | 1.02× | 0.90× |
| `metal/f32/layer_norm/outer_axis/outer_offset_r3_c5` | none | 1.04× | 0.95× |
| `metal/f32/layer_norm/outer_axis/outer_offset_r3_c65` | none | 1.05× | 1.04× |
| `metal/f32/layer_norm/outer_axis/outer_strided_r128_c129` | none | 0.90× | 1.00× |
| `metal/f32/layer_norm/outer_axis/outer_strided_r128_c32` | none | 0.99× | 1.00× |
| `metal/f32/layer_norm/outer_axis/outer_strided_r128_c5` | none | 1.04× | 1.07× |
| `metal/f32/layer_norm/outer_axis/outer_strided_r128_c65` | none | 1.00× | 1.01× |
| `metal/f32/layer_norm/outer_axis/outer_strided_r16_c129` | none | 0.72× | 1.01× |
| `metal/f32/layer_norm/outer_axis/outer_strided_r16_c32` | none | 0.99× | 1.21× |
| `metal/f32/layer_norm/outer_axis/outer_strided_r16_c5` | none | 1.06× | 1.02× |
| `metal/f32/layer_norm/outer_axis/outer_strided_r16_c65` | none | 1.03× | 0.79× |
| `metal/f32/layer_norm/outer_axis/outer_strided_r33_c129` | none | 1.01× | 0.86× |
| `metal/f32/layer_norm/outer_axis/outer_strided_r33_c32` | none | 1.00× | 0.75× |
| `metal/f32/layer_norm/outer_axis/outer_strided_r33_c5` | none | 0.98× | 1.01× |
| `metal/f32/layer_norm/outer_axis/outer_strided_r33_c65` | none | 0.80× | 0.79× |
| `metal/f32/layer_norm/outer_axis/outer_strided_r3_c129` | none | 0.85× | 1.34× |
| `metal/f32/layer_norm/outer_axis/outer_strided_r3_c32` | none | 0.95× | 1.01× |
| `metal/f32/layer_norm/outer_axis/outer_strided_r3_c5` | none | 0.92× | 1.08× |
| `metal/f32/layer_norm/outer_axis/outer_strided_r3_c65` | none | 1.07× | 1.01× |
| `metal/f32/layer_norm/strided/inner_offset_r128_c129` | none | 0.83× | 1.08× |
| `metal/f32/layer_norm/strided/inner_offset_r128_c32` | none | 0.96× | 1.01× |
| `metal/f32/layer_norm/strided/inner_offset_r128_c5` | none | 1.06× | 0.99× |
| `metal/f32/layer_norm/strided/inner_offset_r128_c65` | none | 0.94× | 0.98× |
| `metal/f32/layer_norm/strided/inner_offset_r16_c129` | none | 0.75× | 0.81× |
| `metal/f32/layer_norm/strided/inner_offset_r16_c32` | none | 1.25× | 0.89× |
| `metal/f32/layer_norm/strided/inner_offset_r16_c5` | none | 0.82× | 1.11× |
| `metal/f32/layer_norm/strided/inner_offset_r16_c65` | none | 1.02× | 1.02× |
| `metal/f32/layer_norm/strided/inner_offset_r33_c129` | none | 0.82× | 1.04× |
| `metal/f32/layer_norm/strided/inner_offset_r33_c32` | none | 0.91× | 0.99× |
| `metal/f32/layer_norm/strided/inner_offset_r33_c5` | none | 0.97× | 1.03× |
| `metal/f32/layer_norm/strided/inner_offset_r33_c65` | none | 1.07× | 0.96× |
| `metal/f32/layer_norm/strided/inner_offset_r3_c129` | none | 0.91× | 0.94× |
| `metal/f32/layer_norm/strided/inner_offset_r3_c32` | none | 1.00× | 1.34× |
| `metal/f32/layer_norm/strided/inner_offset_r3_c5` | none | 1.08× | 0.99× |
| `metal/f32/layer_norm/strided/inner_offset_r3_c65` | none | 1.12× | 0.88× |
| `metal/f32/layer_norm/strided/inner_strided_r128_c129` | none | 0.96× | 0.78× |
| `metal/f32/layer_norm/strided/inner_strided_r128_c32` | none | 1.00× | 1.05× |
| `metal/f32/layer_norm/strided/inner_strided_r128_c5` | none | 1.00× | 0.99× |
| `metal/f32/layer_norm/strided/inner_strided_r128_c65` | none | 0.87× | 1.08× |
| `metal/f32/layer_norm/strided/inner_strided_r16_c129` | none | 0.84× | 1.01× |
| `metal/f32/layer_norm/strided/inner_strided_r16_c32` | none | 0.96× | 1.03× |
| `metal/f32/layer_norm/strided/inner_strided_r16_c5` | none | 1.01× | 0.82× |
| `metal/f32/layer_norm/strided/inner_strided_r16_c65` | none | 0.99× | 1.06× |
| `metal/f32/layer_norm/strided/inner_strided_r33_c129` | none | 0.91× | 0.89× |
| `metal/f32/layer_norm/strided/inner_strided_r33_c32` | none | 1.01× | 0.96× |
| `metal/f32/layer_norm/strided/inner_strided_r33_c5` | none | 0.86× | 1.05× |
| `metal/f32/layer_norm/strided/inner_strided_r33_c65` | none | 1.39× | 0.97× |
| `metal/f32/layer_norm/strided/inner_strided_r3_c129` | none | 0.92× | 1.29× |
| `metal/f32/layer_norm/strided/inner_strided_r3_c32` | none | 1.00× | 1.00× |
| `metal/f32/layer_norm/strided/inner_strided_r3_c5` | none | 0.99× | 0.84× |
| `metal/f32/layer_norm/strided/inner_strided_r3_c65` | none | 1.08× | 1.03× |
