# Parity regression check

Target: `Mac16,1/26.6.2/arm64`

- Cells compared: 64
- Added cells: 0
- Removed cells: 0
- Changed case definitions: 0
- New correctness failures: 0
- Diagnostic timing suspects: 1
- Target matches: yes
- Measurement protocol matches: yes

Timing changes are diagnostic and do not fail this check. Timing is compared only when target and protocol match. No timing baseline is accepted by this comparison.

| Cell | Correctness regression | Operator relative timing | Program relative timing |
| --- | --- | ---: | ---: |
| `metal/f32/layer_norm/outer_axis/outer_offset_r129_c127` | none | 1.07× | 0.95× |
| `metal/f32/layer_norm/outer_axis/outer_offset_r129_c17` | none | 0.74× | 1.06× |
| `metal/f32/layer_norm/outer_axis/outer_offset_r129_c257` | none | 0.95× | 0.95× |
| `metal/f32/layer_norm/outer_axis/outer_offset_r129_c63` | none | 1.03× | 0.91× |
| `metal/f32/layer_norm/outer_axis/outer_offset_r2_c127` | none | 0.98× | 0.88× |
| `metal/f32/layer_norm/outer_axis/outer_offset_r2_c17` | none | 1.32× | 0.57× |
| `metal/f32/layer_norm/outer_axis/outer_offset_r2_c257` | none | 1.00× | 1.00× |
| `metal/f32/layer_norm/outer_axis/outer_offset_r2_c63` | none | 1.17× | 1.00× |
| `metal/f32/layer_norm/outer_axis/outer_offset_r64_c127` | none | 0.79× | 1.04× |
| `metal/f32/layer_norm/outer_axis/outer_offset_r64_c17` | none | 1.14× | 0.82× |
| `metal/f32/layer_norm/outer_axis/outer_offset_r64_c257` | none | 0.92× | 0.86× |
| `metal/f32/layer_norm/outer_axis/outer_offset_r64_c63` | none | 0.84× | 0.55× |
| `metal/f32/layer_norm/outer_axis/outer_offset_r7_c127` | none | 0.73× | 1.13× |
| `metal/f32/layer_norm/outer_axis/outer_offset_r7_c17` | none | 1.21× | 1.15× |
| `metal/f32/layer_norm/outer_axis/outer_offset_r7_c257` | none | 1.01× | 0.89× |
| `metal/f32/layer_norm/outer_axis/outer_offset_r7_c63` | none | 0.79× | 1.04× |
| `metal/f32/layer_norm/outer_axis/outer_strided_r129_c127` | none | 0.91× | 1.00× |
| `metal/f32/layer_norm/outer_axis/outer_strided_r129_c17` | none | 0.83× | 0.88× |
| `metal/f32/layer_norm/outer_axis/outer_strided_r129_c257` | none | 1.00× | 0.96× |
| `metal/f32/layer_norm/outer_axis/outer_strided_r129_c63` | none | 0.91× | 1.00× |
| `metal/f32/layer_norm/outer_axis/outer_strided_r2_c127` | none | 0.88× | 1.38× |
| `metal/f32/layer_norm/outer_axis/outer_strided_r2_c17` | none | 0.86× | 1.02× |
| `metal/f32/layer_norm/outer_axis/outer_strided_r2_c257` | none | 1.18× | 1.13× |
| `metal/f32/layer_norm/outer_axis/outer_strided_r2_c63` | none | 1.07× | 0.80× |
| `metal/f32/layer_norm/outer_axis/outer_strided_r64_c127` | none | 1.09× | 1.06× |
| `metal/f32/layer_norm/outer_axis/outer_strided_r64_c17` | none | 0.84× | 0.82× |
| `metal/f32/layer_norm/outer_axis/outer_strided_r64_c257` | none | 0.99× | 1.33× |
| `metal/f32/layer_norm/outer_axis/outer_strided_r64_c63` | none | 0.60× | 1.19× |
| `metal/f32/layer_norm/outer_axis/outer_strided_r7_c127` | none | 1.01× | 1.01× |
| `metal/f32/layer_norm/outer_axis/outer_strided_r7_c17` | none | 0.95× | 0.92× |
| `metal/f32/layer_norm/outer_axis/outer_strided_r7_c257` | none | 0.94× | 1.53× |
| `metal/f32/layer_norm/outer_axis/outer_strided_r7_c63` | none | 0.98× | 0.97× |
| `metal/f32/layer_norm/strided/inner_offset_r129_c127` | none | 0.95× | 0.73× |
| `metal/f32/layer_norm/strided/inner_offset_r129_c17` | none | 1.06× | 1.10× |
| `metal/f32/layer_norm/strided/inner_offset_r129_c257` | none | 0.74× | 0.71× |
| `metal/f32/layer_norm/strided/inner_offset_r129_c63` | none | 0.84× | 0.75× |
| `metal/f32/layer_norm/strided/inner_offset_r2_c127` | none | 0.79× | 0.85× |
| `metal/f32/layer_norm/strided/inner_offset_r2_c17` | none | 0.78× | 0.99× |
| `metal/f32/layer_norm/strided/inner_offset_r2_c257` | none | 1.02× | 1.01× |
| `metal/f32/layer_norm/strided/inner_offset_r2_c63` | none | 1.01× | 0.90× |
| `metal/f32/layer_norm/strided/inner_offset_r64_c127` | none | 0.81× | 0.80× |
| `metal/f32/layer_norm/strided/inner_offset_r64_c17` | none | 0.87× | 0.99× |
| `metal/f32/layer_norm/strided/inner_offset_r64_c257` | none | 0.74× | 0.75× |
| `metal/f32/layer_norm/strided/inner_offset_r64_c63` | none | 0.87× | 0.89× |
| `metal/f32/layer_norm/strided/inner_offset_r7_c127` | none | 1.28× | 0.84× |
| `metal/f32/layer_norm/strided/inner_offset_r7_c17` | none | 1.27× | 1.14× |
| `metal/f32/layer_norm/strided/inner_offset_r7_c257` | none | 0.98× | 1.14× |
| `metal/f32/layer_norm/strided/inner_offset_r7_c63` | none | 1.24× | 1.12× |
| `metal/f32/layer_norm/strided/inner_strided_r129_c127` | none | 0.80× | 0.85× |
| `metal/f32/layer_norm/strided/inner_strided_r129_c17` | none | 1.17× | 1.03× |
| `metal/f32/layer_norm/strided/inner_strided_r129_c257` | none | 0.72× | 0.80× |
| `metal/f32/layer_norm/strided/inner_strided_r129_c63` | none | 0.76× | 0.92× |
| `metal/f32/layer_norm/strided/inner_strided_r2_c127` | none | 0.80× | 0.96× |
| `metal/f32/layer_norm/strided/inner_strided_r2_c17` | none | 0.99× | 0.76× |
| `metal/f32/layer_norm/strided/inner_strided_r2_c257` | none | 1.00× | 1.10× |
| `metal/f32/layer_norm/strided/inner_strided_r2_c63` | none | 0.90× | 0.82× |
| `metal/f32/layer_norm/strided/inner_strided_r64_c127` | none | 0.97× | 0.86× |
| `metal/f32/layer_norm/strided/inner_strided_r64_c17` | none | 1.04× | 1.21× |
| `metal/f32/layer_norm/strided/inner_strided_r64_c257` | none | 0.85× | 0.80× |
| `metal/f32/layer_norm/strided/inner_strided_r64_c63` | none | 0.79× | 0.92× |
| `metal/f32/layer_norm/strided/inner_strided_r7_c127` | none | 0.83× | 1.06× |
| `metal/f32/layer_norm/strided/inner_strided_r7_c17` | none | 0.83× | 0.87× |
| `metal/f32/layer_norm/strided/inner_strided_r7_c257` | none | 0.88× | 1.22× |
| `metal/f32/layer_norm/strided/inner_strided_r7_c63` | none | 0.89× | 0.89× |
