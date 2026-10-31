# Parity regression check

Target: `Mac16,1/26.6.2/arm64`

- Cells compared: 28
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
| `metal/f32/max_all/long_axis` | none | 0.12× | 0.10× |
| `metal/f32/max_all/outer_axis` | none | 0.05× | 0.05× |
| `metal/f32/max_all/ragged` | none | 0.27× | 0.28× |
| `metal/f32/max_all/ragged/r33_c129` | none | 0.07× | 0.07× |
| `metal/f32/max_all/strided/r64_c127` | none | 0.06× | 0.04× |
| `metal/f32/mean_all/long_axis` | none | 0.09× | 0.13× |
| `metal/f32/mean_all/outer_axis` | none | 0.04× | 0.04× |
| `metal/f32/mean_all/ragged` | none | 0.24× | 0.24× |
| `metal/f32/mean_all/ragged/r33_c129` | none | 0.07× | 0.09× |
| `metal/f32/mean_all/strided/r64_c127` | none | 0.04× | 0.04× |
| `metal/f32/min_all/long_axis` | none | 0.11× | 0.09× |
| `metal/f32/min_all/outer_axis` | none | 0.04× | 0.05× |
| `metal/f32/min_all/ragged` | none | 0.28× | 0.28× |
| `metal/f32/min_all/ragged/r33_c129` | none | 0.11× | 0.11× |
| `metal/f32/min_all/strided/r64_c127` | none | 0.04× | 0.04× |
| `metal/f32/std_all/long_axis` | none | 1.55× | 0.98× |
| `metal/f32/std_all/outer_axis` | none | 0.02× | 0.03× |
| `metal/f32/std_all/ragged` | none | 0.16× | 0.19× |
| `metal/f32/std_all/strided/r64_c127` | none | 1.13× | 1.02× |
| `metal/f32/sum_all/long_axis` | none | 0.12× | 0.12× |
| `metal/f32/sum_all/outer_axis` | none | 0.05× | 0.04× |
| `metal/f32/sum_all/ragged` | none | 0.25× | 0.27× |
| `metal/f32/sum_all/ragged/r33_c129` | none | 0.10× | 0.07× |
| `metal/f32/sum_all/strided/r64_c127` | none | 0.04× | 0.04× |
| `metal/f32/variance_all/long_axis` | none | 0.82× | 0.74× |
| `metal/f32/variance_all/outer_axis` | none | 0.02× | 0.02× |
| `metal/f32/variance_all/ragged` | none | 0.17× | 0.17× |
| `metal/f32/variance_all/strided/r64_c127` | none | 1.07× | 1.07× |
