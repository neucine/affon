# Parity regression check

Target: `Mac16,1/26.6.2/arm64`

- Cells compared: 64
- Added cells: 0
- Removed cells: 0
- Changed case definitions: 0
- New correctness failures: 1
- Diagnostic timing suspects: 9
- Target matches: yes
- Measurement protocol matches: yes

Timing changes are diagnostic and do not fail this check. Timing is compared only when target and protocol match. No timing baseline is accepted by this comparison.

| Cell | Correctness regression | Operator relative timing | Program relative timing |
| --- | --- | ---: | ---: |
| `metal/f32/layer_norm/outer_axis/outer_offset_r129_c127` | none | 2.39× | 2.35× |
| `metal/f32/layer_norm/outer_axis/outer_offset_r129_c17` | none | 0.19× | 0.43× |
| `metal/f32/layer_norm/outer_axis/outer_offset_r129_c257` | none | 1.22× | 2.51× |
| `metal/f32/layer_norm/outer_axis/outer_offset_r129_c63` | none | 1.62× | 1.56× |
| `metal/f32/layer_norm/outer_axis/outer_offset_r2_c127` | none | 0.03× | 0.41× |
| `metal/f32/layer_norm/outer_axis/outer_offset_r2_c17` | none | 0.02× | 0.33× |
| `metal/f32/layer_norm/outer_axis/outer_offset_r2_c257` | none | 0.04× | 0.62× |
| `metal/f32/layer_norm/outer_axis/outer_offset_r2_c63` | none | 0.03× | 0.73× |
| `metal/f32/layer_norm/outer_axis/outer_offset_r64_c127` | none | 1.21× | 1.68× |
| `metal/f32/layer_norm/outer_axis/outer_offset_r64_c17` | none | 0.10× | 0.47× |
| `metal/f32/layer_norm/outer_axis/outer_offset_r64_c257` | none | 1.62× | 2.06× |
| `metal/f32/layer_norm/outer_axis/outer_offset_r64_c63` | none | 0.16× | 0.36× |
| `metal/f32/layer_norm/outer_axis/outer_offset_r7_c127` | none | 0.05× | 0.50× |
| `metal/f32/layer_norm/outer_axis/outer_offset_r7_c17` | none | 0.04× | 0.59× |
| `metal/f32/layer_norm/outer_axis/outer_offset_r7_c257` | none | 0.07× | 0.48× |
| `metal/f32/layer_norm/outer_axis/outer_offset_r7_c63` | none | 0.05× | 0.52× |
| `metal/f32/layer_norm/outer_axis/outer_strided_r129_c127` | none | 2.19× | 2.19× |
| `metal/f32/layer_norm/outer_axis/outer_strided_r129_c17` | none | 0.19× | 0.40× |
| `metal/f32/layer_norm/outer_axis/outer_strided_r129_c257` | none | 2.36× | 1.61× |
| `metal/f32/layer_norm/outer_axis/outer_strided_r129_c63` | none | 1.44× | 1.34× |
| `metal/f32/layer_norm/outer_axis/outer_strided_r2_c127` | none | 0.03× | 0.80× |
| `metal/f32/layer_norm/outer_axis/outer_strided_r2_c17` | none | 0.02× | 0.34× |
| `metal/f32/layer_norm/outer_axis/outer_strided_r2_c257` | none | 0.03× | 0.59× |
| `metal/f32/layer_norm/outer_axis/outer_strided_r2_c63` | none | 0.03× | 0.67× |
| `metal/f32/layer_norm/outer_axis/outer_strided_r64_c127` | none | 1.21× | 1.67× |
| `metal/f32/layer_norm/outer_axis/outer_strided_r64_c17` | none | 0.08× | 0.46× |
| `metal/f32/layer_norm/outer_axis/outer_strided_r64_c257` | none | 2.36× | 2.29× |
| `metal/f32/layer_norm/outer_axis/outer_strided_r64_c63` | probe_status, operator_peer, program_peer | — | — |
| `metal/f32/layer_norm/outer_axis/outer_strided_r7_c127` | none | 0.05× | 0.50× |
| `metal/f32/layer_norm/outer_axis/outer_strided_r7_c17` | none | 0.04× | 0.44× |
| `metal/f32/layer_norm/outer_axis/outer_strided_r7_c257` | none | 0.06× | 0.70× |
| `metal/f32/layer_norm/outer_axis/outer_strided_r7_c63` | none | 0.04× | 0.68× |
| `metal/f32/layer_norm/strided/inner_offset_r129_c127` | none | 0.90× | 0.73× |
| `metal/f32/layer_norm/strided/inner_offset_r129_c17` | none | 0.22× | 0.66× |
| `metal/f32/layer_norm/strided/inner_offset_r129_c257` | none | 0.71× | 0.72× |
| `metal/f32/layer_norm/strided/inner_offset_r129_c63` | none | 0.85× | 0.76× |
| `metal/f32/layer_norm/strided/inner_offset_r2_c127` | none | 0.03× | 0.70× |
| `metal/f32/layer_norm/strided/inner_offset_r2_c17` | none | 0.03× | 0.71× |
| `metal/f32/layer_norm/strided/inner_offset_r2_c257` | none | 0.04× | 0.63× |
| `metal/f32/layer_norm/strided/inner_offset_r2_c63` | none | 0.03× | 0.69× |
| `metal/f32/layer_norm/strided/inner_offset_r64_c127` | none | 0.70× | 0.70× |
| `metal/f32/layer_norm/strided/inner_offset_r64_c17` | none | 0.11× | 0.74× |
| `metal/f32/layer_norm/strided/inner_offset_r64_c257` | none | 0.77× | 0.81× |
| `metal/f32/layer_norm/strided/inner_offset_r64_c63` | none | 0.21× | 0.68× |
| `metal/f32/layer_norm/strided/inner_offset_r7_c127` | none | 0.08× | 0.44× |
| `metal/f32/layer_norm/strided/inner_offset_r7_c17` | none | 0.06× | 0.54× |
| `metal/f32/layer_norm/strided/inner_offset_r7_c257` | none | 0.10× | 0.75× |
| `metal/f32/layer_norm/strided/inner_offset_r7_c63` | none | 0.05× | 0.79× |
| `metal/f32/layer_norm/strided/inner_strided_r129_c127` | none | 0.88× | 0.88× |
| `metal/f32/layer_norm/strided/inner_strided_r129_c17` | none | 0.23× | 0.85× |
| `metal/f32/layer_norm/strided/inner_strided_r129_c257` | none | 0.71× | 0.77× |
| `metal/f32/layer_norm/strided/inner_strided_r129_c63` | none | 0.71× | 1.03× |
| `metal/f32/layer_norm/strided/inner_strided_r2_c127` | none | 0.04× | 0.52× |
| `metal/f32/layer_norm/strided/inner_strided_r2_c17` | none | 0.03× | 1.05× |
| `metal/f32/layer_norm/strided/inner_strided_r2_c257` | none | 0.06× | 0.65× |
| `metal/f32/layer_norm/strided/inner_strided_r2_c63` | none | 0.03× | 0.87× |
| `metal/f32/layer_norm/strided/inner_strided_r64_c127` | none | 0.87× | 0.72× |
| `metal/f32/layer_norm/strided/inner_strided_r64_c17` | none | 0.13× | 0.60× |
| `metal/f32/layer_norm/strided/inner_strided_r64_c257` | none | 0.82× | 0.76× |
| `metal/f32/layer_norm/strided/inner_strided_r64_c63` | none | 0.18× | 0.71× |
| `metal/f32/layer_norm/strided/inner_strided_r7_c127` | none | 0.06× | 0.72× |
| `metal/f32/layer_norm/strided/inner_strided_r7_c17` | none | 0.04× | 0.47× |
| `metal/f32/layer_norm/strided/inner_strided_r7_c257` | none | 0.08× | 0.56× |
| `metal/f32/layer_norm/strided/inner_strided_r7_c63` | none | 0.06× | 0.65× |
