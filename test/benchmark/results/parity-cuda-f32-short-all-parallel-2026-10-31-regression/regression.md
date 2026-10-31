# Parity regression check

Target: `NVIDIA GeForce RTX 3090/7.0.0-38-generic/x86_64`

- Cells compared: 7
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
| `cuda/f32/argmax_all/short_axis` | none | 0.16× | 0.20× |
| `cuda/f32/argmin_all/short_axis` | none | 0.16× | 0.21× |
| `cuda/f32/max_all/short_axis` | none | 0.17× | 0.20× |
| `cuda/f32/mean_all/short_axis` | none | 0.16× | 0.20× |
| `cuda/f32/min_all/short_axis` | none | 0.16× | 0.20× |
| `cuda/f32/std_all/short_axis` | none | 0.16× | 0.21× |
| `cuda/f32/variance_all/short_axis` | none | 0.16× | 0.19× |
