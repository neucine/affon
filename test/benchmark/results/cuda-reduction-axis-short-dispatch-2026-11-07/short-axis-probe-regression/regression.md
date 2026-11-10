# Parity regression check

Target: `NVIDIA GeForce RTX 3090/7.0.0-38-generic/x86_64`

- Cells compared: 24
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
| `cuda/f32/argmax_axis/outer_axis/offset_r3_c5` | none | 0.69× | 0.75× |
| `cuda/f32/argmax_axis/outer_axis/offset_r7_c63` | none | 0.13× | 0.15× |
| `cuda/f32/argmax_axis/strided/strided_r129_c63` | none | 0.04× | 0.04× |
| `cuda/f32/argmin_axis/outer_axis/offset_r3_c5` | none | 0.70× | 0.80× |
| `cuda/f32/argmin_axis/outer_axis/offset_r7_c63` | none | 0.13× | 0.16× |
| `cuda/f32/argmin_axis/strided/strided_r129_c63` | none | 0.04× | 0.04× |
| `cuda/f32/max_axis/outer_axis/offset_r3_c129` | none | 0.07× | 0.09× |
| `cuda/f32/max_axis/outer_axis/offset_r3_c5` | none | 0.70× | 0.76× |
| `cuda/f32/max_axis/strided/inner_strided_r129_c63` | none | 0.04× | 0.04× |
| `cuda/f32/mean_axis/outer_axis/offset_r3_c129` | none | 0.07× | 0.09× |
| `cuda/f32/mean_axis/outer_axis/offset_r3_c5` | none | 0.71× | 0.74× |
| `cuda/f32/mean_axis/strided/inner_strided_r129_c63` | none | 0.07× | 0.09× |
| `cuda/f32/min_axis/outer_axis/offset_r3_c129` | none | 0.07× | 0.09× |
| `cuda/f32/min_axis/outer_axis/offset_r3_c5` | none | 0.71× | 0.76× |
| `cuda/f32/min_axis/strided/inner_strided_r129_c63` | none | 0.03× | 0.04× |
| `cuda/f32/std_axis/outer_axis/offset_r3_c5` | none | 0.66× | 0.74× |
| `cuda/f32/std_axis/outer_axis/offset_r7_c63` | none | 0.11× | 0.14× |
| `cuda/f32/std_axis/strided/strided_r129_c63` | none | 0.03× | 0.04× |
| `cuda/f32/sum_axis/outer_axis/offset_r3_c129` | none | 0.07× | 0.09× |
| `cuda/f32/sum_axis/outer_axis/offset_r3_c5` | none | 0.70× | 0.72× |
| `cuda/f32/sum_axis/strided/inner_strided_r129_c63` | none | 0.05× | 0.06× |
| `cuda/f32/variance_axis/outer_axis/offset_r3_c5` | none | 0.66× | 0.70× |
| `cuda/f32/variance_axis/outer_axis/offset_r7_c63` | none | 0.11× | 0.13× |
| `cuda/f32/variance_axis/strided/strided_r129_c63` | none | 0.03× | 0.04× |
