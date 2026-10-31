# Parity regression check

Target: `NVIDIA GeForce RTX 3090/7.0.0-38-generic/x86_64`

- Cells compared: 16
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
| `cuda/f32/argmax_axis/long_axis` | none | 0.08× | 0.12× |
| `cuda/f32/argmax_axis/short_axis` | none | 0.98× | 1.03× |
| `cuda/f32/argmin_axis/long_axis` | none | 0.08× | 0.13× |
| `cuda/f32/argmin_axis/short_axis` | none | 0.98× | 0.99× |
| `cuda/f32/max_axis/long_axis` | none | 0.00× | 0.00× |
| `cuda/f32/max_axis/short_axis` | none | 0.96× | 1.00× |
| `cuda/f32/mean_axis/long_axis` | none | 0.00× | 0.00× |
| `cuda/f32/mean_axis/short_axis` | none | 0.97× | 0.99× |
| `cuda/f32/min_axis/long_axis` | none | 0.00× | 0.00× |
| `cuda/f32/min_axis/short_axis` | none | 0.98× | 1.00× |
| `cuda/f32/std_axis/long_axis` | none | 0.00× | 0.01× |
| `cuda/f32/std_axis/short_axis` | none | 0.98× | 0.92× |
| `cuda/f32/sum_axis/long_axis` | none | 0.01× | 0.01× |
| `cuda/f32/sum_axis/short_axis` | none | 0.99× | 1.03× |
| `cuda/f32/variance_axis/long_axis` | none | 0.00× | 0.01× |
| `cuda/f32/variance_axis/short_axis` | none | 0.97× | 0.99× |
