# Parity regression check

Target: `Mac16,1/26.6.2/arm64`

- Cells compared: 64
- Added cells: 0
- Removed cells: 0
- Changed case definitions: 0
- New correctness failures: 0
- Diagnostic timing suspects: 2
- Target matches: yes
- Measurement protocol matches: yes

Timing changes are diagnostic and do not fail this check. Timing is compared only when target and protocol match. No timing baseline is accepted by this comparison.

| Cell | Correctness regression | Operator relative timing | Program relative timing |
| --- | --- | ---: | ---: |
| `metal/f32/layer_norm/outer_axis/outer_offset_r129_c127` | none | 1.03× | 0.85× |
| `metal/f32/layer_norm/outer_axis/outer_offset_r129_c17` | none | 0.17× | 0.52× |
| `metal/f32/layer_norm/outer_axis/outer_offset_r129_c257` | none | 0.89× | 1.19× |
| `metal/f32/layer_norm/outer_axis/outer_offset_r129_c63` | none | 1.68× | 1.63× |
| `metal/f32/layer_norm/outer_axis/outer_offset_r2_c127` | none | 0.04× | 0.55× |
| `metal/f32/layer_norm/outer_axis/outer_offset_r2_c17` | none | 0.02× | 0.28× |
| `metal/f32/layer_norm/outer_axis/outer_offset_r2_c257` | none | 0.05× | 0.63× |
| `metal/f32/layer_norm/outer_axis/outer_offset_r2_c63` | none | 0.03× | 0.64× |
| `metal/f32/layer_norm/outer_axis/outer_offset_r64_c127` | none | 0.67× | 1.00× |
| `metal/f32/layer_norm/outer_axis/outer_offset_r64_c17` | none | 0.10× | 0.44× |
| `metal/f32/layer_norm/outer_axis/outer_offset_r64_c257` | none | 0.61× | 1.01× |
| `metal/f32/layer_norm/outer_axis/outer_offset_r64_c63` | none | 1.01× | 0.83× |
| `metal/f32/layer_norm/outer_axis/outer_offset_r7_c127` | none | 0.04× | 0.60× |
| `metal/f32/layer_norm/outer_axis/outer_offset_r7_c17` | none | 0.02× | 0.82× |
| `metal/f32/layer_norm/outer_axis/outer_offset_r7_c257` | none | 0.08× | 0.50× |
| `metal/f32/layer_norm/outer_axis/outer_offset_r7_c63` | none | 0.04× | 0.56× |
| `metal/f32/layer_norm/outer_axis/outer_strided_r129_c127` | none | 0.88× | 1.02× |
| `metal/f32/layer_norm/outer_axis/outer_strided_r129_c17` | none | 0.13× | 0.42× |
| `metal/f32/layer_norm/outer_axis/outer_strided_r129_c257` | none | 1.32× | 1.02× |
| `metal/f32/layer_norm/outer_axis/outer_strided_r129_c63` | none | 1.35× | 1.32× |
| `metal/f32/layer_norm/outer_axis/outer_strided_r2_c127` | none | 0.04× | 0.96× |
| `metal/f32/layer_norm/outer_axis/outer_strided_r2_c17` | none | 0.02× | 0.37× |
| `metal/f32/layer_norm/outer_axis/outer_strided_r2_c257` | none | 0.05× | 0.65× |
| `metal/f32/layer_norm/outer_axis/outer_strided_r2_c63` | none | 0.05× | 0.53× |
| `metal/f32/layer_norm/outer_axis/outer_strided_r64_c127` | none | 0.63× | 0.76× |
| `metal/f32/layer_norm/outer_axis/outer_strided_r64_c17` | none | 0.08× | 0.48× |
| `metal/f32/layer_norm/outer_axis/outer_strided_r64_c257` | none | 1.75× | 1.05× |
| `metal/f32/layer_norm/outer_axis/outer_strided_r64_c63` | none | 0.68× | 0.53× |
| `metal/f32/layer_norm/outer_axis/outer_strided_r7_c127` | none | 0.05× | 0.57× |
| `metal/f32/layer_norm/outer_axis/outer_strided_r7_c17` | none | 0.03× | 0.50× |
| `metal/f32/layer_norm/outer_axis/outer_strided_r7_c257` | none | 0.06× | 1.09× |
| `metal/f32/layer_norm/outer_axis/outer_strided_r7_c63` | none | 0.04× | 0.73× |
| `metal/f32/layer_norm/strided/inner_offset_r129_c127` | none | 0.88× | 0.73× |
| `metal/f32/layer_norm/strided/inner_offset_r129_c17` | none | 0.23× | 0.89× |
| `metal/f32/layer_norm/strided/inner_offset_r129_c257` | none | 0.74× | 0.72× |
| `metal/f32/layer_norm/strided/inner_offset_r129_c63` | none | 1.02× | 0.74× |
| `metal/f32/layer_norm/strided/inner_offset_r2_c127` | none | 0.03× | 0.92× |
| `metal/f32/layer_norm/strided/inner_offset_r2_c17` | none | 0.03× | 0.64× |
| `metal/f32/layer_norm/strided/inner_offset_r2_c257` | none | 0.06× | 0.69× |
| `metal/f32/layer_norm/strided/inner_offset_r2_c63` | none | 0.03× | 1.13× |
| `metal/f32/layer_norm/strided/inner_offset_r64_c127` | none | 0.79× | 0.73× |
| `metal/f32/layer_norm/strided/inner_offset_r64_c17` | none | 0.12× | 0.90× |
| `metal/f32/layer_norm/strided/inner_offset_r64_c257` | none | 0.82× | 0.73× |
| `metal/f32/layer_norm/strided/inner_offset_r64_c63` | none | 0.24× | 0.86× |
| `metal/f32/layer_norm/strided/inner_offset_r7_c127` | none | 0.08× | 0.71× |
| `metal/f32/layer_norm/strided/inner_offset_r7_c17` | none | 0.04× | 0.64× |
| `metal/f32/layer_norm/strided/inner_offset_r7_c257` | none | 0.07× | 0.83× |
| `metal/f32/layer_norm/strided/inner_offset_r7_c63` | none | 0.05× | 0.69× |
| `metal/f32/layer_norm/strided/inner_strided_r129_c127` | none | 0.81× | 0.84× |
| `metal/f32/layer_norm/strided/inner_strided_r129_c17` | none | 0.27× | 0.87× |
| `metal/f32/layer_norm/strided/inner_strided_r129_c257` | none | 0.67× | 0.76× |
| `metal/f32/layer_norm/strided/inner_strided_r129_c63` | none | 0.84× | 1.35× |
| `metal/f32/layer_norm/strided/inner_strided_r2_c127` | none | 0.03× | 0.48× |
| `metal/f32/layer_norm/strided/inner_strided_r2_c17` | none | 0.03× | 0.57× |
| `metal/f32/layer_norm/strided/inner_strided_r2_c257` | none | 0.05× | 0.72× |
| `metal/f32/layer_norm/strided/inner_strided_r2_c63` | none | 0.02× | 0.52× |
| `metal/f32/layer_norm/strided/inner_strided_r64_c127` | none | 0.82× | 0.84× |
| `metal/f32/layer_norm/strided/inner_strided_r64_c17` | none | 0.13× | 0.78× |
| `metal/f32/layer_norm/strided/inner_strided_r64_c257` | none | 0.78× | 0.77× |
| `metal/f32/layer_norm/strided/inner_strided_r64_c63` | none | 0.21× | 0.92× |
| `metal/f32/layer_norm/strided/inner_strided_r7_c127` | none | 0.06× | 0.84× |
| `metal/f32/layer_norm/strided/inner_strided_r7_c17` | none | 0.04× | 0.64× |
| `metal/f32/layer_norm/strided/inner_strided_r7_c257` | none | 0.09× | 0.75× |
| `metal/f32/layer_norm/strided/inner_strided_r7_c63` | none | 0.07× | 0.60× |
