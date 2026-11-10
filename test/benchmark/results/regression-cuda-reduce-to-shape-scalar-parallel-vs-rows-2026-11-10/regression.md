# Parity regression check

Target: `NVIDIA GeForce RTX 3090/7.0.0-42-generic/x86_64`

- Cells compared: 134
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
| `cuda/f32/reduce_to_shape/long_axis` | none | 0.97× | 0.98× |
| `cuda/f32/reduce_to_shape/long_axis/dense_r128_c129` | none | 1.01× | 1.01× |
| `cuda/f32/reduce_to_shape/long_axis/dense_r128_c32` | none | 0.99× | 1.01× |
| `cuda/f32/reduce_to_shape/long_axis/dense_r128_c5` | none | 1.00× | 1.00× |
| `cuda/f32/reduce_to_shape/long_axis/dense_r128_c65` | none | 0.99× | 1.03× |
| `cuda/f32/reduce_to_shape/long_axis/dense_r16_c129` | none | 0.98× | 0.99× |
| `cuda/f32/reduce_to_shape/long_axis/dense_r16_c32` | none | 0.98× | 0.97× |
| `cuda/f32/reduce_to_shape/long_axis/dense_r16_c5` | none | 1.00× | 0.99× |
| `cuda/f32/reduce_to_shape/long_axis/dense_r16_c65` | none | 1.02× | 0.99× |
| `cuda/f32/reduce_to_shape/long_axis/dense_r33_c129` | none | 1.01× | 1.02× |
| `cuda/f32/reduce_to_shape/long_axis/dense_r33_c32` | none | 1.00× | 1.00× |
| `cuda/f32/reduce_to_shape/long_axis/dense_r33_c5` | none | 0.98× | 0.98× |
| `cuda/f32/reduce_to_shape/long_axis/dense_r33_c65` | none | 1.02× | 1.00× |
| `cuda/f32/reduce_to_shape/long_axis/dense_r3_c129` | none | 0.99× | 1.00× |
| `cuda/f32/reduce_to_shape/long_axis/dense_r3_c32` | none | 0.98× | 0.99× |
| `cuda/f32/reduce_to_shape/long_axis/dense_r3_c5` | none | 1.00× | 1.02× |
| `cuda/f32/reduce_to_shape/long_axis/dense_r3_c65` | none | 0.98× | 0.98× |
| `cuda/f32/reduce_to_shape/long_axis/offset_r128_c129` | none | 1.00× | 1.03× |
| `cuda/f32/reduce_to_shape/long_axis/offset_r128_c32` | none | 1.03× | 1.01× |
| `cuda/f32/reduce_to_shape/long_axis/offset_r128_c5` | none | 1.00× | 1.01× |
| `cuda/f32/reduce_to_shape/long_axis/offset_r128_c65` | none | 1.00× | 1.02× |
| `cuda/f32/reduce_to_shape/long_axis/offset_r16_c129` | none | 0.99× | 0.98× |
| `cuda/f32/reduce_to_shape/long_axis/offset_r16_c32` | none | 0.99× | 1.01× |
| `cuda/f32/reduce_to_shape/long_axis/offset_r16_c5` | none | 0.99× | 0.97× |
| `cuda/f32/reduce_to_shape/long_axis/offset_r16_c65` | none | 1.03× | 0.97× |
| `cuda/f32/reduce_to_shape/long_axis/offset_r33_c129` | none | 1.04× | 0.98× |
| `cuda/f32/reduce_to_shape/long_axis/offset_r33_c32` | none | 0.97× | 0.97× |
| `cuda/f32/reduce_to_shape/long_axis/offset_r33_c5` | none | 1.02× | 0.97× |
| `cuda/f32/reduce_to_shape/long_axis/offset_r33_c65` | none | 1.00× | 1.03× |
| `cuda/f32/reduce_to_shape/long_axis/offset_r3_c129` | none | 0.96× | 1.01× |
| `cuda/f32/reduce_to_shape/long_axis/offset_r3_c32` | none | 1.01× | 1.02× |
| `cuda/f32/reduce_to_shape/long_axis/offset_r3_c5` | none | 1.01× | 0.97× |
| `cuda/f32/reduce_to_shape/long_axis/offset_r3_c65` | none | 0.97× | 0.99× |
| `cuda/f32/reduce_to_shape/outer_axis` | none | 1.00× | 1.03× |
| `cuda/f32/reduce_to_shape/outer_axis/dense_r128_c129` | none | 0.38× | 0.43× |
| `cuda/f32/reduce_to_shape/outer_axis/dense_r128_c32` | none | 0.35× | 0.42× |
| `cuda/f32/reduce_to_shape/outer_axis/dense_r128_c5` | none | 0.41× | 0.46× |
| `cuda/f32/reduce_to_shape/outer_axis/dense_r128_c65` | none | 0.36× | 0.41× |
| `cuda/f32/reduce_to_shape/outer_axis/dense_r16_c129` | none | 0.83× | 0.86× |
| `cuda/f32/reduce_to_shape/outer_axis/dense_r16_c32` | none | 0.78× | 0.86× |
| `cuda/f32/reduce_to_shape/outer_axis/dense_r16_c5` | none | 0.86× | 0.89× |
| `cuda/f32/reduce_to_shape/outer_axis/dense_r16_c65` | none | 0.78× | 0.87× |
| `cuda/f32/reduce_to_shape/outer_axis/dense_r33_c129` | none | 0.76× | 0.78× |
| `cuda/f32/reduce_to_shape/outer_axis/dense_r33_c32` | none | 0.69× | 0.76× |
| `cuda/f32/reduce_to_shape/outer_axis/dense_r33_c5` | none | 0.77× | 0.80× |
| `cuda/f32/reduce_to_shape/outer_axis/dense_r33_c65` | none | 0.72× | 0.75× |
| `cuda/f32/reduce_to_shape/outer_axis/dense_r3_c129` | none | 0.98× | 1.03× |
| `cuda/f32/reduce_to_shape/outer_axis/dense_r3_c32` | none | 1.03× | 0.99× |
| `cuda/f32/reduce_to_shape/outer_axis/dense_r3_c5` | none | 1.00× | 0.99× |
| `cuda/f32/reduce_to_shape/outer_axis/dense_r3_c65` | none | 1.00× | 0.99× |
| `cuda/f32/reduce_to_shape/outer_axis/offset_r128_c129` | none | 0.39× | 0.45× |
| `cuda/f32/reduce_to_shape/outer_axis/offset_r128_c32` | none | 0.36× | 0.41× |
| `cuda/f32/reduce_to_shape/outer_axis/offset_r128_c5` | none | 0.40× | 0.44× |
| `cuda/f32/reduce_to_shape/outer_axis/offset_r128_c65` | none | 0.37× | 0.42× |
| `cuda/f32/reduce_to_shape/outer_axis/offset_r16_c129` | none | 0.85× | 0.86× |
| `cuda/f32/reduce_to_shape/outer_axis/offset_r16_c32` | none | 0.84× | 0.85× |
| `cuda/f32/reduce_to_shape/outer_axis/offset_r16_c5` | none | 0.84× | 0.89× |
| `cuda/f32/reduce_to_shape/outer_axis/offset_r16_c65` | none | 0.83× | 0.88× |
| `cuda/f32/reduce_to_shape/outer_axis/offset_r33_c129` | none | 0.74× | 0.79× |
| `cuda/f32/reduce_to_shape/outer_axis/offset_r33_c32` | none | 0.74× | 0.75× |
| `cuda/f32/reduce_to_shape/outer_axis/offset_r33_c5` | none | 0.74× | 0.80× |
| `cuda/f32/reduce_to_shape/outer_axis/offset_r33_c65` | none | 0.72× | 0.76× |
| `cuda/f32/reduce_to_shape/outer_axis/offset_r3_c129` | none | 1.00× | 0.98× |
| `cuda/f32/reduce_to_shape/outer_axis/offset_r3_c32` | none | 1.00× | 1.02× |
| `cuda/f32/reduce_to_shape/outer_axis/offset_r3_c5` | none | 1.01× | 1.01× |
| `cuda/f32/reduce_to_shape/outer_axis/offset_r3_c65` | none | 1.01× | 0.99× |
| `cuda/f32/reduce_to_shape/ragged` | none | 0.99× | 0.98× |
| `cuda/f32/reduce_to_shape/ragged/dense_r128_c129` | none | 0.05× | 0.05× |
| `cuda/f32/reduce_to_shape/ragged/dense_r128_c32` | none | 0.10× | 0.12× |
| `cuda/f32/reduce_to_shape/ragged/dense_r128_c5` | none | 0.31× | 0.36× |
| `cuda/f32/reduce_to_shape/ragged/dense_r128_c65` | none | 0.07× | 0.08× |
| `cuda/f32/reduce_to_shape/ragged/dense_r16_c129` | none | 0.18× | 0.21× |
| `cuda/f32/reduce_to_shape/ragged/dense_r16_c32` | none | 0.43× | 0.48× |
| `cuda/f32/reduce_to_shape/ragged/dense_r16_c5` | none | 1.00× | 0.99× |
| `cuda/f32/reduce_to_shape/ragged/dense_r16_c65` | none | 0.29× | 0.34× |
| `cuda/f32/reduce_to_shape/ragged/dense_r33_c129` | none | 0.10× | 0.13× |
| `cuda/f32/reduce_to_shape/ragged/dense_r33_c32` | none | 0.28× | 0.32× |
| `cuda/f32/reduce_to_shape/ragged/dense_r33_c5` | none | 0.99× | 1.00× |
| `cuda/f32/reduce_to_shape/ragged/dense_r33_c65` | none | 0.17× | 0.20× |
| `cuda/f32/reduce_to_shape/ragged/dense_r3_c129` | none | 0.51× | 0.54× |
| `cuda/f32/reduce_to_shape/ragged/dense_r3_c32` | none | 1.00× | 0.97× |
| `cuda/f32/reduce_to_shape/ragged/dense_r3_c5` | none | 1.00× | 1.00× |
| `cuda/f32/reduce_to_shape/ragged/dense_r3_c65` | none | 0.99× | 0.97× |
| `cuda/f32/reduce_to_shape/ragged/offset_r128_c129` | none | 0.05× | 0.06× |
| `cuda/f32/reduce_to_shape/ragged/offset_r128_c32` | none | 0.10× | 0.12× |
| `cuda/f32/reduce_to_shape/ragged/offset_r128_c5` | none | 0.31× | 0.35× |
| `cuda/f32/reduce_to_shape/ragged/offset_r128_c65` | none | 0.07× | 0.08× |
| `cuda/f32/reduce_to_shape/ragged/offset_r16_c129` | none | 0.18× | 0.21× |
| `cuda/f32/reduce_to_shape/ragged/offset_r16_c32` | none | 0.43× | 0.50× |
| `cuda/f32/reduce_to_shape/ragged/offset_r16_c5` | none | 0.99× | 1.01× |
| `cuda/f32/reduce_to_shape/ragged/offset_r16_c65` | none | 0.28× | 0.35× |
| `cuda/f32/reduce_to_shape/ragged/offset_r33_c129` | none | 0.11× | 0.13× |
| `cuda/f32/reduce_to_shape/ragged/offset_r33_c32` | none | 0.28× | 0.33× |
| `cuda/f32/reduce_to_shape/ragged/offset_r33_c5` | none | 1.00× | 1.01× |
| `cuda/f32/reduce_to_shape/ragged/offset_r33_c65` | none | 0.17× | 0.21× |
| `cuda/f32/reduce_to_shape/ragged/offset_r3_c129` | none | 0.51× | 0.60× |
| `cuda/f32/reduce_to_shape/ragged/offset_r3_c32` | none | 1.02× | 0.99× |
| `cuda/f32/reduce_to_shape/ragged/offset_r3_c5` | none | 1.01× | 1.03× |
| `cuda/f32/reduce_to_shape/ragged/offset_r3_c65` | none | 0.99× | 1.03× |
| `cuda/f32/reduce_to_shape/scalar` | none | 0.27× | 0.32× |
| `cuda/f32/reduce_to_shape/short_axis` | none | 1.00× | 1.01× |
| `cuda/f32/reduce_to_shape/short_axis/dense_r128_c129` | none | 0.37× | 0.42× |
| `cuda/f32/reduce_to_shape/short_axis/dense_r128_c32` | none | 0.35× | 0.41× |
| `cuda/f32/reduce_to_shape/short_axis/dense_r128_c5` | none | 0.40× | 0.48× |
| `cuda/f32/reduce_to_shape/short_axis/dense_r128_c65` | none | 0.36× | 0.42× |
| `cuda/f32/reduce_to_shape/short_axis/dense_r16_c129` | none | 0.81× | 0.86× |
| `cuda/f32/reduce_to_shape/short_axis/dense_r16_c32` | none | 0.80× | 0.86× |
| `cuda/f32/reduce_to_shape/short_axis/dense_r16_c5` | none | 0.85× | 0.87× |
| `cuda/f32/reduce_to_shape/short_axis/dense_r16_c65` | none | 2.37× | 0.86× |
| `cuda/f32/reduce_to_shape/short_axis/dense_r33_c129` | none | 0.75× | 0.78× |
| `cuda/f32/reduce_to_shape/short_axis/dense_r33_c32` | none | 0.72× | 0.77× |
| `cuda/f32/reduce_to_shape/short_axis/dense_r33_c5` | none | 0.72× | 0.79× |
| `cuda/f32/reduce_to_shape/short_axis/dense_r33_c65` | none | 0.74× | 0.75× |
| `cuda/f32/reduce_to_shape/short_axis/dense_r3_c129` | none | 0.98× | 0.99× |
| `cuda/f32/reduce_to_shape/short_axis/dense_r3_c32` | none | 0.96× | 1.01× |
| `cuda/f32/reduce_to_shape/short_axis/dense_r3_c5` | none | 0.97× | 1.01× |
| `cuda/f32/reduce_to_shape/short_axis/dense_r3_c65` | none | 0.98× | 1.01× |
| `cuda/f32/reduce_to_shape/short_axis/offset_r128_c129` | none | 0.37× | 0.44× |
| `cuda/f32/reduce_to_shape/short_axis/offset_r128_c32` | none | 0.36× | 0.41× |
| `cuda/f32/reduce_to_shape/short_axis/offset_r128_c5` | none | 0.38× | 0.45× |
| `cuda/f32/reduce_to_shape/short_axis/offset_r128_c65` | none | 0.36× | 0.43× |
| `cuda/f32/reduce_to_shape/short_axis/offset_r16_c129` | none | 0.81× | 0.85× |
| `cuda/f32/reduce_to_shape/short_axis/offset_r16_c32` | none | 0.80× | 0.87× |
| `cuda/f32/reduce_to_shape/short_axis/offset_r16_c5` | none | 0.82× | 0.88× |
| `cuda/f32/reduce_to_shape/short_axis/offset_r16_c65` | none | 0.83× | 0.84× |
| `cuda/f32/reduce_to_shape/short_axis/offset_r33_c129` | none | 0.72× | 0.77× |
| `cuda/f32/reduce_to_shape/short_axis/offset_r33_c32` | none | 0.70× | 0.76× |
| `cuda/f32/reduce_to_shape/short_axis/offset_r33_c5` | none | 0.73× | 0.78× |
| `cuda/f32/reduce_to_shape/short_axis/offset_r33_c65` | none | 0.71× | 0.77× |
| `cuda/f32/reduce_to_shape/short_axis/offset_r3_c129` | none | 1.01× | 0.98× |
| `cuda/f32/reduce_to_shape/short_axis/offset_r3_c32` | none | 0.99× | 1.01× |
| `cuda/f32/reduce_to_shape/short_axis/offset_r3_c5` | none | 0.98× | 0.99× |
| `cuda/f32/reduce_to_shape/short_axis/offset_r3_c65` | none | 1.01× | 1.01× |
| `cuda/f32/reduce_to_shape/strided` | none | 0.70× | 0.78× |
