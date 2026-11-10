# Parity regression check

Target: `NVIDIA GeForce RTX 3090/7.0.0-42-generic/x86_64`

- Cells compared: 12
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
| `cuda/bf16/argmax_axis/strided/strided_r64_c63` | none | 0.54× | 0.56× |
| `cuda/bf16/argmin_axis/strided/strided_r64_c63` | none | 0.56× | 0.56× |
| `cuda/bf16/max_axis/strided/inner_strided_r64_c63` | none | 0.53× | 0.48× |
| `cuda/bf16/mean_axis/strided/inner_strided_r64_c63` | none | 0.54× | 0.55× |
| `cuda/bf16/min_axis/strided/inner_strided_r64_c63` | none | 0.69× | 0.54× |
| `cuda/bf16/sum_axis/strided/inner_strided_r64_c63` | none | 0.53× | 0.56× |
| `cuda/f32/argmax_axis/strided/strided_r129_c63` | none | 0.83× | 0.83× |
| `cuda/f32/argmin_axis/strided/strided_r64_c63` | none | 0.79× | 0.88× |
| `cuda/f32/max_axis/strided/inner_strided_r64_c63` | none | 0.61× | 0.74× |
| `cuda/f32/mean_axis/strided/inner_strided_r64_c63` | none | 0.61× | 0.66× |
| `cuda/f32/min_axis/strided/inner_strided_r64_c63` | none | 0.62× | 0.66× |
| `cuda/f32/sum_axis/strided/inner_offset_r64_c63` | none | 0.68× | 0.78× |
