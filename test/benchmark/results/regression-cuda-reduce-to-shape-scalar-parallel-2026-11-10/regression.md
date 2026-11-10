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
| `cuda/f32/reduce_to_shape/long_axis` | none | 0.99× | 1.01× |
| `cuda/f32/reduce_to_shape/long_axis/dense_r128_c129` | none | 0.74× | 0.75× |
| `cuda/f32/reduce_to_shape/long_axis/dense_r128_c32` | none | 0.92× | 0.93× |
| `cuda/f32/reduce_to_shape/long_axis/dense_r128_c5` | none | 1.00× | 1.04× |
| `cuda/f32/reduce_to_shape/long_axis/dense_r128_c65` | none | 0.87× | 0.90× |
| `cuda/f32/reduce_to_shape/long_axis/dense_r16_c129` | none | 0.64× | 0.71× |
| `cuda/f32/reduce_to_shape/long_axis/dense_r16_c32` | none | 0.93× | 0.94× |
| `cuda/f32/reduce_to_shape/long_axis/dense_r16_c5` | none | 1.01× | 1.01× |
| `cuda/f32/reduce_to_shape/long_axis/dense_r16_c65` | none | 0.77× | 0.81× |
| `cuda/f32/reduce_to_shape/long_axis/dense_r33_c129` | none | 0.66× | 0.70× |
| `cuda/f32/reduce_to_shape/long_axis/dense_r33_c32` | none | 0.89× | 0.90× |
| `cuda/f32/reduce_to_shape/long_axis/dense_r33_c5` | none | 1.01× | 1.00× |
| `cuda/f32/reduce_to_shape/long_axis/dense_r33_c65` | none | 0.80× | 0.82× |
| `cuda/f32/reduce_to_shape/long_axis/dense_r3_c129` | none | 0.68× | 0.75× |
| `cuda/f32/reduce_to_shape/long_axis/dense_r3_c32` | none | 0.93× | 0.92× |
| `cuda/f32/reduce_to_shape/long_axis/dense_r3_c5` | none | 1.04× | 0.97× |
| `cuda/f32/reduce_to_shape/long_axis/dense_r3_c65` | none | 0.83× | 0.85× |
| `cuda/f32/reduce_to_shape/long_axis/offset_r128_c129` | none | 0.84× | 0.82× |
| `cuda/f32/reduce_to_shape/long_axis/offset_r128_c32` | none | 0.94× | 0.96× |
| `cuda/f32/reduce_to_shape/long_axis/offset_r128_c5` | none | 0.99× | 1.00× |
| `cuda/f32/reduce_to_shape/long_axis/offset_r128_c65` | none | 0.83× | 0.90× |
| `cuda/f32/reduce_to_shape/long_axis/offset_r16_c129` | none | 0.70× | 0.76× |
| `cuda/f32/reduce_to_shape/long_axis/offset_r16_c32` | none | 0.89× | 0.90× |
| `cuda/f32/reduce_to_shape/long_axis/offset_r16_c5` | none | 0.99× | 1.02× |
| `cuda/f32/reduce_to_shape/long_axis/offset_r16_c65` | none | 0.83× | 0.87× |
| `cuda/f32/reduce_to_shape/long_axis/offset_r33_c129` | none | 0.71× | 0.75× |
| `cuda/f32/reduce_to_shape/long_axis/offset_r33_c32` | none | 0.89× | 2.94× |
| `cuda/f32/reduce_to_shape/long_axis/offset_r33_c5` | none | 0.96× | 1.00× |
| `cuda/f32/reduce_to_shape/long_axis/offset_r33_c65` | none | 0.81× | 0.87× |
| `cuda/f32/reduce_to_shape/long_axis/offset_r3_c129` | none | 0.74× | 0.77× |
| `cuda/f32/reduce_to_shape/long_axis/offset_r3_c32` | none | 0.85× | 0.94× |
| `cuda/f32/reduce_to_shape/long_axis/offset_r3_c5` | none | 1.02× | 1.01× |
| `cuda/f32/reduce_to_shape/long_axis/offset_r3_c65` | none | 0.85× | 0.86× |
| `cuda/f32/reduce_to_shape/outer_axis` | none | 1.03× | 1.01× |
| `cuda/f32/reduce_to_shape/outer_axis/dense_r128_c129` | none | 0.36× | 0.43× |
| `cuda/f32/reduce_to_shape/outer_axis/dense_r128_c32` | none | 0.34× | 0.40× |
| `cuda/f32/reduce_to_shape/outer_axis/dense_r128_c5` | none | 0.39× | 0.45× |
| `cuda/f32/reduce_to_shape/outer_axis/dense_r128_c65` | none | 0.35× | 0.41× |
| `cuda/f32/reduce_to_shape/outer_axis/dense_r16_c129` | none | 1.02× | 1.02× |
| `cuda/f32/reduce_to_shape/outer_axis/dense_r16_c32` | none | 1.00× | 1.04× |
| `cuda/f32/reduce_to_shape/outer_axis/dense_r16_c5` | none | 0.98× | 0.99× |
| `cuda/f32/reduce_to_shape/outer_axis/dense_r16_c65` | none | 0.98× | 0.98× |
| `cuda/f32/reduce_to_shape/outer_axis/dense_r33_c129` | none | 0.71× | 0.75× |
| `cuda/f32/reduce_to_shape/outer_axis/dense_r33_c32` | none | 0.68× | 0.77× |
| `cuda/f32/reduce_to_shape/outer_axis/dense_r33_c5` | none | 0.72× | 0.78× |
| `cuda/f32/reduce_to_shape/outer_axis/dense_r33_c65` | none | 0.71× | 0.75× |
| `cuda/f32/reduce_to_shape/outer_axis/dense_r3_c129` | none | 1.01× | 1.01× |
| `cuda/f32/reduce_to_shape/outer_axis/dense_r3_c32` | none | 0.99× | 1.01× |
| `cuda/f32/reduce_to_shape/outer_axis/dense_r3_c5` | none | 1.02× | 1.02× |
| `cuda/f32/reduce_to_shape/outer_axis/dense_r3_c65` | none | 0.99× | 1.01× |
| `cuda/f32/reduce_to_shape/outer_axis/offset_r128_c129` | none | 0.35× | 0.43× |
| `cuda/f32/reduce_to_shape/outer_axis/offset_r128_c32` | none | 0.34× | 0.40× |
| `cuda/f32/reduce_to_shape/outer_axis/offset_r128_c5` | none | 0.37× | 0.44× |
| `cuda/f32/reduce_to_shape/outer_axis/offset_r128_c65` | none | 0.35× | 0.41× |
| `cuda/f32/reduce_to_shape/outer_axis/offset_r16_c129` | none | 1.00× | 1.01× |
| `cuda/f32/reduce_to_shape/outer_axis/offset_r16_c32` | none | 0.99× | 1.00× |
| `cuda/f32/reduce_to_shape/outer_axis/offset_r16_c5` | none | 1.04× | 0.99× |
| `cuda/f32/reduce_to_shape/outer_axis/offset_r16_c65` | none | 0.98× | 1.00× |
| `cuda/f32/reduce_to_shape/outer_axis/offset_r33_c129` | none | 0.69× | 0.74× |
| `cuda/f32/reduce_to_shape/outer_axis/offset_r33_c32` | none | 0.67× | 0.72× |
| `cuda/f32/reduce_to_shape/outer_axis/offset_r33_c5` | none | 0.72× | 0.77× |
| `cuda/f32/reduce_to_shape/outer_axis/offset_r33_c65` | none | 0.67× | 0.75× |
| `cuda/f32/reduce_to_shape/outer_axis/offset_r3_c129` | none | 0.98× | 0.99× |
| `cuda/f32/reduce_to_shape/outer_axis/offset_r3_c32` | none | 1.01× | 1.00× |
| `cuda/f32/reduce_to_shape/outer_axis/offset_r3_c5` | none | 0.97× | 1.00× |
| `cuda/f32/reduce_to_shape/outer_axis/offset_r3_c65` | none | 0.97× | 0.97× |
| `cuda/f32/reduce_to_shape/ragged` | none | 0.97× | 0.98× |
| `cuda/f32/reduce_to_shape/ragged/dense_r128_c129` | none | 0.05× | 0.05× |
| `cuda/f32/reduce_to_shape/ragged/dense_r128_c32` | none | 0.09× | 0.12× |
| `cuda/f32/reduce_to_shape/ragged/dense_r128_c5` | none | 0.29× | 0.35× |
| `cuda/f32/reduce_to_shape/ragged/dense_r128_c65` | none | 0.06× | 0.08× |
| `cuda/f32/reduce_to_shape/ragged/dense_r16_c129` | none | 0.17× | 0.21× |
| `cuda/f32/reduce_to_shape/ragged/dense_r16_c32` | none | 0.41× | 0.49× |
| `cuda/f32/reduce_to_shape/ragged/dense_r16_c5` | none | 0.97× | 1.04× |
| `cuda/f32/reduce_to_shape/ragged/dense_r16_c65` | none | 0.27× | 0.33× |
| `cuda/f32/reduce_to_shape/ragged/dense_r33_c129` | none | 0.10× | 0.12× |
| `cuda/f32/reduce_to_shape/ragged/dense_r33_c32` | none | 0.25× | 0.31× |
| `cuda/f32/reduce_to_shape/ragged/dense_r33_c5` | none | 1.01× | 1.00× |
| `cuda/f32/reduce_to_shape/ragged/dense_r33_c65` | none | 0.16× | 0.20× |
| `cuda/f32/reduce_to_shape/ragged/dense_r3_c129` | none | 0.51× | 0.56× |
| `cuda/f32/reduce_to_shape/ragged/dense_r3_c32` | none | 1.01× | 1.01× |
| `cuda/f32/reduce_to_shape/ragged/dense_r3_c5` | none | 1.02× | 1.03× |
| `cuda/f32/reduce_to_shape/ragged/dense_r3_c65` | none | 1.01× | 1.01× |
| `cuda/f32/reduce_to_shape/ragged/offset_r128_c129` | none | 0.05× | 0.05× |
| `cuda/f32/reduce_to_shape/ragged/offset_r128_c32` | none | 0.10× | 0.12× |
| `cuda/f32/reduce_to_shape/ragged/offset_r128_c5` | none | 0.28× | 0.34× |
| `cuda/f32/reduce_to_shape/ragged/offset_r128_c65` | none | 0.07× | 0.07× |
| `cuda/f32/reduce_to_shape/ragged/offset_r16_c129` | none | 0.17× | 0.21× |
| `cuda/f32/reduce_to_shape/ragged/offset_r16_c32` | none | 0.41× | 0.46× |
| `cuda/f32/reduce_to_shape/ragged/offset_r16_c5` | none | 0.99× | 1.01× |
| `cuda/f32/reduce_to_shape/ragged/offset_r16_c65` | none | 0.28× | 0.32× |
| `cuda/f32/reduce_to_shape/ragged/offset_r33_c129` | none | 0.10× | 0.12× |
| `cuda/f32/reduce_to_shape/ragged/offset_r33_c32` | none | 0.27× | 0.32× |
| `cuda/f32/reduce_to_shape/ragged/offset_r33_c5` | none | 0.94× | 1.03× |
| `cuda/f32/reduce_to_shape/ragged/offset_r33_c65` | none | 0.16× | 0.19× |
| `cuda/f32/reduce_to_shape/ragged/offset_r3_c129` | none | 0.49× | 0.58× |
| `cuda/f32/reduce_to_shape/ragged/offset_r3_c32` | none | 1.03× | 0.98× |
| `cuda/f32/reduce_to_shape/ragged/offset_r3_c5` | none | 0.97× | 1.02× |
| `cuda/f32/reduce_to_shape/ragged/offset_r3_c65` | none | 0.98× | 1.03× |
| `cuda/f32/reduce_to_shape/scalar` | none | 0.97× | 1.01× |
| `cuda/f32/reduce_to_shape/short_axis` | none | 1.02× | 0.99× |
| `cuda/f32/reduce_to_shape/short_axis/dense_r128_c129` | none | 0.35× | 0.42× |
| `cuda/f32/reduce_to_shape/short_axis/dense_r128_c32` | none | 0.34× | 0.41× |
| `cuda/f32/reduce_to_shape/short_axis/dense_r128_c5` | none | 0.38× | 0.47× |
| `cuda/f32/reduce_to_shape/short_axis/dense_r128_c65` | none | 0.35× | 0.41× |
| `cuda/f32/reduce_to_shape/short_axis/dense_r16_c129` | none | 0.97× | 1.02× |
| `cuda/f32/reduce_to_shape/short_axis/dense_r16_c32` | none | 0.98× | 1.02× |
| `cuda/f32/reduce_to_shape/short_axis/dense_r16_c5` | none | 0.99× | 1.02× |
| `cuda/f32/reduce_to_shape/short_axis/dense_r16_c65` | none | 0.99× | 1.03× |
| `cuda/f32/reduce_to_shape/short_axis/dense_r33_c129` | none | 0.67× | 0.77× |
| `cuda/f32/reduce_to_shape/short_axis/dense_r33_c32` | none | 0.72× | 0.74× |
| `cuda/f32/reduce_to_shape/short_axis/dense_r33_c5` | none | 0.75× | 0.78× |
| `cuda/f32/reduce_to_shape/short_axis/dense_r33_c65` | none | 0.70× | 0.76× |
| `cuda/f32/reduce_to_shape/short_axis/dense_r3_c129` | none | 0.99× | 1.00× |
| `cuda/f32/reduce_to_shape/short_axis/dense_r3_c32` | none | 0.97× | 0.98× |
| `cuda/f32/reduce_to_shape/short_axis/dense_r3_c5` | none | 0.99× | 1.01× |
| `cuda/f32/reduce_to_shape/short_axis/dense_r3_c65` | none | 0.97× | 1.02× |
| `cuda/f32/reduce_to_shape/short_axis/offset_r128_c129` | none | 0.35× | 0.43× |
| `cuda/f32/reduce_to_shape/short_axis/offset_r128_c32` | none | 0.34× | 0.41× |
| `cuda/f32/reduce_to_shape/short_axis/offset_r128_c5` | none | 0.38× | 0.44× |
| `cuda/f32/reduce_to_shape/short_axis/offset_r128_c65` | none | 0.35× | 0.41× |
| `cuda/f32/reduce_to_shape/short_axis/offset_r16_c129` | none | 0.99× | 1.01× |
| `cuda/f32/reduce_to_shape/short_axis/offset_r16_c32` | none | 0.99× | 0.99× |
| `cuda/f32/reduce_to_shape/short_axis/offset_r16_c5` | none | 0.96× | 1.01× |
| `cuda/f32/reduce_to_shape/short_axis/offset_r16_c65` | none | 0.98× | 0.98× |
| `cuda/f32/reduce_to_shape/short_axis/offset_r33_c129` | none | 0.69× | 0.74× |
| `cuda/f32/reduce_to_shape/short_axis/offset_r33_c32` | none | 0.66× | 0.73× |
| `cuda/f32/reduce_to_shape/short_axis/offset_r33_c5` | none | 0.72× | 0.75× |
| `cuda/f32/reduce_to_shape/short_axis/offset_r33_c65` | none | 0.67× | 0.74× |
| `cuda/f32/reduce_to_shape/short_axis/offset_r3_c129` | none | 1.04× | 1.02× |
| `cuda/f32/reduce_to_shape/short_axis/offset_r3_c32` | none | 1.00× | 1.00× |
| `cuda/f32/reduce_to_shape/short_axis/offset_r3_c5` | none | 1.01× | 1.02× |
| `cuda/f32/reduce_to_shape/short_axis/offset_r3_c65` | none | 1.00× | 0.96× |
| `cuda/f32/reduce_to_shape/strided` | none | 0.68× | 0.76× |
