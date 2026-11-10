# Parity regression check

Target: `NVIDIA GeForce RTX 3090/7.0.0-38-generic/x86_64`

- Cells compared: 134
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
| `cuda/f32/reduce_to_shape/long_axis` | none | 0.72× | 0.76× |
| `cuda/f32/reduce_to_shape/long_axis/dense_r128_c129` | none | 0.01× | 0.01× |
| `cuda/f32/reduce_to_shape/long_axis/dense_r128_c32` | none | 0.02× | 0.02× |
| `cuda/f32/reduce_to_shape/long_axis/dense_r128_c5` | none | 0.10× | 0.12× |
| `cuda/f32/reduce_to_shape/long_axis/dense_r128_c65` | none | 0.01× | 0.01× |
| `cuda/f32/reduce_to_shape/long_axis/dense_r16_c129` | none | 0.03× | 0.03× |
| `cuda/f32/reduce_to_shape/long_axis/dense_r16_c32` | none | 0.07× | 0.09× |
| `cuda/f32/reduce_to_shape/long_axis/dense_r16_c5` | none | 0.32× | 0.38× |
| `cuda/f32/reduce_to_shape/long_axis/dense_r16_c65` | none | 0.04× | 0.05× |
| `cuda/f32/reduce_to_shape/long_axis/dense_r33_c129` | none | 0.02× | 0.02× |
| `cuda/f32/reduce_to_shape/long_axis/dense_r33_c32` | none | 0.04× | 0.05× |
| `cuda/f32/reduce_to_shape/long_axis/dense_r33_c5` | none | 0.21× | 0.24× |
| `cuda/f32/reduce_to_shape/long_axis/dense_r33_c65` | none | 0.02× | 0.03× |
| `cuda/f32/reduce_to_shape/long_axis/dense_r3_c129` | none | 0.12× | 0.13× |
| `cuda/f32/reduce_to_shape/long_axis/dense_r3_c32` | none | 0.28× | 0.33× |
| `cuda/f32/reduce_to_shape/long_axis/dense_r3_c5` | none | 0.73× | 0.77× |
| `cuda/f32/reduce_to_shape/long_axis/dense_r3_c65` | none | 0.18× | 0.21× |
| `cuda/f32/reduce_to_shape/long_axis/offset_r128_c129` | none | 0.01× | 0.01× |
| `cuda/f32/reduce_to_shape/long_axis/offset_r128_c32` | none | 0.02× | 0.02× |
| `cuda/f32/reduce_to_shape/long_axis/offset_r128_c5` | none | 0.10× | 0.12× |
| `cuda/f32/reduce_to_shape/long_axis/offset_r128_c65` | none | 0.01× | 0.01× |
| `cuda/f32/reduce_to_shape/long_axis/offset_r16_c129` | none | 0.03× | 0.03× |
| `cuda/f32/reduce_to_shape/long_axis/offset_r16_c32` | none | 0.08× | 0.09× |
| `cuda/f32/reduce_to_shape/long_axis/offset_r16_c5` | none | 0.31× | 0.36× |
| `cuda/f32/reduce_to_shape/long_axis/offset_r16_c65` | none | 0.04× | 0.05× |
| `cuda/f32/reduce_to_shape/long_axis/offset_r33_c129` | none | 0.01× | 0.02× |
| `cuda/f32/reduce_to_shape/long_axis/offset_r33_c32` | none | 0.04× | 0.05× |
| `cuda/f32/reduce_to_shape/long_axis/offset_r33_c5` | none | 0.20× | 0.23× |
| `cuda/f32/reduce_to_shape/long_axis/offset_r33_c65` | none | 0.02× | 0.03× |
| `cuda/f32/reduce_to_shape/long_axis/offset_r3_c129` | none | 0.12× | 0.14× |
| `cuda/f32/reduce_to_shape/long_axis/offset_r3_c32` | none | 0.29× | 0.33× |
| `cuda/f32/reduce_to_shape/long_axis/offset_r3_c5` | none | 0.73× | 0.74× |
| `cuda/f32/reduce_to_shape/long_axis/offset_r3_c65` | none | 0.18× | 0.21× |
| `cuda/f32/reduce_to_shape/outer_axis` | none | 0.73× | 0.76× |
| `cuda/f32/reduce_to_shape/outer_axis/dense_r128_c129` | none | 0.01× | 0.01× |
| `cuda/f32/reduce_to_shape/outer_axis/dense_r128_c32` | none | 0.03× | 0.03× |
| `cuda/f32/reduce_to_shape/outer_axis/dense_r128_c5` | none | 0.14× | 0.15× |
| `cuda/f32/reduce_to_shape/outer_axis/dense_r128_c65` | none | 0.01× | 0.01× |
| `cuda/f32/reduce_to_shape/outer_axis/dense_r16_c129` | none | 0.02× | 0.03× |
| `cuda/f32/reduce_to_shape/outer_axis/dense_r16_c32` | none | 0.08× | 0.09× |
| `cuda/f32/reduce_to_shape/outer_axis/dense_r16_c5` | none | 0.35× | 0.40× |
| `cuda/f32/reduce_to_shape/outer_axis/dense_r16_c65` | none | 0.04× | 0.05× |
| `cuda/f32/reduce_to_shape/outer_axis/dense_r33_c129` | none | 0.01× | 0.02× |
| `cuda/f32/reduce_to_shape/outer_axis/dense_r33_c32` | none | 0.05× | 0.06× |
| `cuda/f32/reduce_to_shape/outer_axis/dense_r33_c5` | none | 0.24× | 0.27× |
| `cuda/f32/reduce_to_shape/outer_axis/dense_r33_c65` | none | 0.03× | 0.03× |
| `cuda/f32/reduce_to_shape/outer_axis/dense_r3_c129` | none | 0.09× | 0.12× |
| `cuda/f32/reduce_to_shape/outer_axis/dense_r3_c32` | none | 0.27× | 0.31× |
| `cuda/f32/reduce_to_shape/outer_axis/dense_r3_c5` | none | 0.73× | 0.75× |
| `cuda/f32/reduce_to_shape/outer_axis/dense_r3_c65` | none | 0.17× | 0.20× |
| `cuda/f32/reduce_to_shape/outer_axis/offset_r128_c129` | none | 0.01× | 0.01× |
| `cuda/f32/reduce_to_shape/outer_axis/offset_r128_c32` | none | 0.03× | 0.03× |
| `cuda/f32/reduce_to_shape/outer_axis/offset_r128_c5` | none | 0.14× | 0.15× |
| `cuda/f32/reduce_to_shape/outer_axis/offset_r128_c65` | none | 0.01× | 0.01× |
| `cuda/f32/reduce_to_shape/outer_axis/offset_r16_c129` | none | 0.02× | 0.03× |
| `cuda/f32/reduce_to_shape/outer_axis/offset_r16_c32` | none | 0.07× | 0.10× |
| `cuda/f32/reduce_to_shape/outer_axis/offset_r16_c5` | none | 0.35× | 0.40× |
| `cuda/f32/reduce_to_shape/outer_axis/offset_r16_c65` | none | 0.04× | 0.05× |
| `cuda/f32/reduce_to_shape/outer_axis/offset_r33_c129` | none | 0.01× | 0.02× |
| `cuda/f32/reduce_to_shape/outer_axis/offset_r33_c32` | none | 0.05× | 0.06× |
| `cuda/f32/reduce_to_shape/outer_axis/offset_r33_c5` | none | 0.23× | 0.27× |
| `cuda/f32/reduce_to_shape/outer_axis/offset_r33_c65` | none | 0.03× | 0.03× |
| `cuda/f32/reduce_to_shape/outer_axis/offset_r3_c129` | none | 0.09× | 0.12× |
| `cuda/f32/reduce_to_shape/outer_axis/offset_r3_c32` | none | 0.26× | 0.32× |
| `cuda/f32/reduce_to_shape/outer_axis/offset_r3_c5` | none | 0.71× | 0.76× |
| `cuda/f32/reduce_to_shape/outer_axis/offset_r3_c65` | none | 0.17× | 0.20× |
| `cuda/f32/reduce_to_shape/ragged` | none | 0.71× | 0.75× |
| `cuda/f32/reduce_to_shape/ragged/dense_r128_c129` | none | 0.09× | 0.10× |
| `cuda/f32/reduce_to_shape/ragged/dense_r128_c32` | none | 0.11× | 0.11× |
| `cuda/f32/reduce_to_shape/ragged/dense_r128_c5` | none | 0.18× | 0.19× |
| `cuda/f32/reduce_to_shape/ragged/dense_r128_c65` | none | 0.10× | 0.10× |
| `cuda/f32/reduce_to_shape/ragged/dense_r16_c129` | none | 0.11× | 0.11× |
| `cuda/f32/reduce_to_shape/ragged/dense_r16_c32` | none | 0.15× | 0.16× |
| `cuda/f32/reduce_to_shape/ragged/dense_r16_c5` | none | 0.37× | 0.43× |
| `cuda/f32/reduce_to_shape/ragged/dense_r16_c65` | none | 0.11× | 0.12× |
| `cuda/f32/reduce_to_shape/ragged/dense_r33_c129` | none | 0.10× | 0.10× |
| `cuda/f32/reduce_to_shape/ragged/dense_r33_c32` | none | 0.13× | 0.14× |
| `cuda/f32/reduce_to_shape/ragged/dense_r33_c5` | none | 0.28× | 0.30× |
| `cuda/f32/reduce_to_shape/ragged/dense_r33_c65` | none | 0.10× | 0.11× |
| `cuda/f32/reduce_to_shape/ragged/dense_r3_c129` | none | 0.17× | 0.19× |
| `cuda/f32/reduce_to_shape/ragged/dense_r3_c32` | none | 0.33× | 0.37× |
| `cuda/f32/reduce_to_shape/ragged/dense_r3_c5` | none | 0.71× | 0.81× |
| `cuda/f32/reduce_to_shape/ragged/dense_r3_c65` | none | 0.22× | 0.25× |
| `cuda/f32/reduce_to_shape/ragged/offset_r128_c129` | none | 0.10× | 0.09× |
| `cuda/f32/reduce_to_shape/ragged/offset_r128_c32` | none | 0.10× | 0.11× |
| `cuda/f32/reduce_to_shape/ragged/offset_r128_c5` | none | 0.18× | 0.19× |
| `cuda/f32/reduce_to_shape/ragged/offset_r128_c65` | none | 0.09× | 0.10× |
| `cuda/f32/reduce_to_shape/ragged/offset_r16_c129` | none | 0.11× | 0.11× |
| `cuda/f32/reduce_to_shape/ragged/offset_r16_c32` | none | 0.14× | 0.16× |
| `cuda/f32/reduce_to_shape/ragged/offset_r16_c5` | none | 0.39× | 0.43× |
| `cuda/f32/reduce_to_shape/ragged/offset_r16_c65` | none | 0.12× | 0.12× |
| `cuda/f32/reduce_to_shape/ragged/offset_r33_c129` | none | 0.10× | 0.10× |
| `cuda/f32/reduce_to_shape/ragged/offset_r33_c32` | none | 0.13× | 0.13× |
| `cuda/f32/reduce_to_shape/ragged/offset_r33_c5` | none | 0.27× | 0.31× |
| `cuda/f32/reduce_to_shape/ragged/offset_r33_c65` | none | 0.11× | 0.11× |
| `cuda/f32/reduce_to_shape/ragged/offset_r3_c129` | none | 0.17× | 0.19× |
| `cuda/f32/reduce_to_shape/ragged/offset_r3_c32` | none | 0.34× | 0.37× |
| `cuda/f32/reduce_to_shape/ragged/offset_r3_c5` | none | 0.74× | 0.75× |
| `cuda/f32/reduce_to_shape/ragged/offset_r3_c65` | none | 0.22× | 0.25× |
| `cuda/f32/reduce_to_shape/scalar` | none | 1.00× | 0.98× |
| `cuda/f32/reduce_to_shape/short_axis` | none | 0.73× | 0.77× |
| `cuda/f32/reduce_to_shape/short_axis/dense_r128_c129` | none | 0.01× | 0.01× |
| `cuda/f32/reduce_to_shape/short_axis/dense_r128_c32` | none | 0.02× | 0.03× |
| `cuda/f32/reduce_to_shape/short_axis/dense_r128_c5` | none | 0.13× | 0.14× |
| `cuda/f32/reduce_to_shape/short_axis/dense_r128_c65` | none | 0.01× | 0.01× |
| `cuda/f32/reduce_to_shape/short_axis/dense_r16_c129` | none | 0.02× | 0.03× |
| `cuda/f32/reduce_to_shape/short_axis/dense_r16_c32` | none | 0.08× | 0.10× |
| `cuda/f32/reduce_to_shape/short_axis/dense_r16_c5` | none | 0.35× | 0.41× |
| `cuda/f32/reduce_to_shape/short_axis/dense_r16_c65` | none | 0.04× | 0.05× |
| `cuda/f32/reduce_to_shape/short_axis/dense_r33_c129` | none | 0.01× | 0.02× |
| `cuda/f32/reduce_to_shape/short_axis/dense_r33_c32` | none | 0.04× | 0.06× |
| `cuda/f32/reduce_to_shape/short_axis/dense_r33_c5` | none | 0.24× | 0.27× |
| `cuda/f32/reduce_to_shape/short_axis/dense_r33_c65` | none | 0.03× | 0.03× |
| `cuda/f32/reduce_to_shape/short_axis/dense_r3_c129` | none | 0.09× | 0.12× |
| `cuda/f32/reduce_to_shape/short_axis/dense_r3_c32` | none | 0.28× | 0.34× |
| `cuda/f32/reduce_to_shape/short_axis/dense_r3_c5` | none | 0.70× | 0.79× |
| `cuda/f32/reduce_to_shape/short_axis/dense_r3_c65` | none | 0.17× | 0.21× |
| `cuda/f32/reduce_to_shape/short_axis/offset_r128_c129` | none | 0.01× | 0.01× |
| `cuda/f32/reduce_to_shape/short_axis/offset_r128_c32` | none | 0.02× | 0.03× |
| `cuda/f32/reduce_to_shape/short_axis/offset_r128_c5` | none | 0.13× | 0.15× |
| `cuda/f32/reduce_to_shape/short_axis/offset_r128_c65` | none | 0.01× | 0.01× |
| `cuda/f32/reduce_to_shape/short_axis/offset_r16_c129` | none | 0.02× | 0.03× |
| `cuda/f32/reduce_to_shape/short_axis/offset_r16_c32` | none | 0.08× | 0.09× |
| `cuda/f32/reduce_to_shape/short_axis/offset_r16_c5` | none | 0.35× | 0.41× |
| `cuda/f32/reduce_to_shape/short_axis/offset_r16_c65` | none | 0.04× | 0.05× |
| `cuda/f32/reduce_to_shape/short_axis/offset_r33_c129` | none | 0.01× | 0.02× |
| `cuda/f32/reduce_to_shape/short_axis/offset_r33_c32` | none | 0.05× | 0.06× |
| `cuda/f32/reduce_to_shape/short_axis/offset_r33_c5` | none | 0.24× | 0.28× |
| `cuda/f32/reduce_to_shape/short_axis/offset_r33_c65` | none | 0.03× | 0.03× |
| `cuda/f32/reduce_to_shape/short_axis/offset_r3_c129` | none | 0.09× | 0.11× |
| `cuda/f32/reduce_to_shape/short_axis/offset_r3_c32` | none | 0.28× | 0.33× |
| `cuda/f32/reduce_to_shape/short_axis/offset_r3_c5` | none | 0.71× | 0.76× |
| `cuda/f32/reduce_to_shape/short_axis/offset_r3_c65` | none | 0.17× | 0.20× |
| `cuda/f32/reduce_to_shape/strided` | none | 0.02× | 0.03× |
