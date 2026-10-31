# Parity regression check

Target: `Mac16,1/26.6.2/arm64`

- Cells compared: 128
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
| `metal/f32/reduce_to_shape/long_axis/dense_r128_c129` | none | 0.02× | 0.02× |
| `metal/f32/reduce_to_shape/long_axis/dense_r128_c32` | none | 0.07× | 0.08× |
| `metal/f32/reduce_to_shape/long_axis/dense_r128_c5` | none | 0.23× | 0.25× |
| `metal/f32/reduce_to_shape/long_axis/dense_r128_c65` | none | 0.03× | 0.03× |
| `metal/f32/reduce_to_shape/long_axis/dense_r16_c129` | none | 0.14× | 0.15× |
| `metal/f32/reduce_to_shape/long_axis/dense_r16_c32` | none | 0.37× | 0.41× |
| `metal/f32/reduce_to_shape/long_axis/dense_r16_c5` | none | 0.89× | 0.94× |
| `metal/f32/reduce_to_shape/long_axis/dense_r16_c65` | none | 0.21× | 0.25× |
| `metal/f32/reduce_to_shape/long_axis/dense_r33_c129` | none | 0.07× | 0.07× |
| `metal/f32/reduce_to_shape/long_axis/dense_r33_c32` | none | 0.17× | 0.25× |
| `metal/f32/reduce_to_shape/long_axis/dense_r33_c5` | none | 0.61× | 0.63× |
| `metal/f32/reduce_to_shape/long_axis/dense_r33_c65` | none | 0.11× | 0.13× |
| `metal/f32/reduce_to_shape/long_axis/dense_r3_c129` | none | 0.53× | 0.52× |
| `metal/f32/reduce_to_shape/long_axis/dense_r3_c32` | none | 1.10× | 1.12× |
| `metal/f32/reduce_to_shape/long_axis/dense_r3_c5` | none | 1.04× | 0.90× |
| `metal/f32/reduce_to_shape/long_axis/dense_r3_c65` | none | 0.69× | 0.68× |
| `metal/f32/reduce_to_shape/long_axis/offset_r128_c129` | none | 0.02× | 0.02× |
| `metal/f32/reduce_to_shape/long_axis/offset_r128_c32` | none | 0.06× | 0.08× |
| `metal/f32/reduce_to_shape/long_axis/offset_r128_c5` | none | 0.22× | 0.26× |
| `metal/f32/reduce_to_shape/long_axis/offset_r128_c65` | none | 0.05× | 0.03× |
| `metal/f32/reduce_to_shape/long_axis/offset_r16_c129` | none | 0.14× | 0.14× |
| `metal/f32/reduce_to_shape/long_axis/offset_r16_c32` | none | 0.45× | 0.33× |
| `metal/f32/reduce_to_shape/long_axis/offset_r16_c5` | none | 0.85× | 0.87× |
| `metal/f32/reduce_to_shape/long_axis/offset_r16_c65` | none | 0.25× | 0.21× |
| `metal/f32/reduce_to_shape/long_axis/offset_r33_c129` | none | 0.09× | 0.07× |
| `metal/f32/reduce_to_shape/long_axis/offset_r33_c32` | none | 0.23× | 0.28× |
| `metal/f32/reduce_to_shape/long_axis/offset_r33_c5` | none | 0.57× | 0.56× |
| `metal/f32/reduce_to_shape/long_axis/offset_r33_c65` | none | 0.11× | 0.11× |
| `metal/f32/reduce_to_shape/long_axis/offset_r3_c129` | none | 0.52× | 0.50× |
| `metal/f32/reduce_to_shape/long_axis/offset_r3_c32` | none | 1.09× | 0.82× |
| `metal/f32/reduce_to_shape/long_axis/offset_r3_c5` | none | 0.85× | 0.86× |
| `metal/f32/reduce_to_shape/long_axis/offset_r3_c65` | none | 0.68× | 0.64× |
| `metal/f32/reduce_to_shape/outer_axis/dense_r128_c129` | none | 0.02× | 0.02× |
| `metal/f32/reduce_to_shape/outer_axis/dense_r128_c32` | none | 0.06× | 0.06× |
| `metal/f32/reduce_to_shape/outer_axis/dense_r128_c5` | none | 0.31× | 0.36× |
| `metal/f32/reduce_to_shape/outer_axis/dense_r128_c65` | none | 0.03× | 0.04× |
| `metal/f32/reduce_to_shape/outer_axis/dense_r16_c129` | none | 0.11× | 0.11× |
| `metal/f32/reduce_to_shape/outer_axis/dense_r16_c32` | none | 0.36× | 0.34× |
| `metal/f32/reduce_to_shape/outer_axis/dense_r16_c5` | none | 0.86× | 0.71× |
| `metal/f32/reduce_to_shape/outer_axis/dense_r16_c65` | none | 0.20× | 0.22× |
| `metal/f32/reduce_to_shape/outer_axis/dense_r33_c129` | none | 0.06× | 0.08× |
| `metal/f32/reduce_to_shape/outer_axis/dense_r33_c32` | none | 0.19× | 0.19× |
| `metal/f32/reduce_to_shape/outer_axis/dense_r33_c5` | none | 0.67× | 0.72× |
| `metal/f32/reduce_to_shape/outer_axis/dense_r33_c65` | none | 0.11× | 0.11× |
| `metal/f32/reduce_to_shape/outer_axis/dense_r3_c129` | none | 0.45× | 0.52× |
| `metal/f32/reduce_to_shape/outer_axis/dense_r3_c32` | none | 0.65× | 0.79× |
| `metal/f32/reduce_to_shape/outer_axis/dense_r3_c5` | none | 0.78× | 1.11× |
| `metal/f32/reduce_to_shape/outer_axis/dense_r3_c65` | none | 0.72× | 0.61× |
| `metal/f32/reduce_to_shape/outer_axis/offset_r128_c129` | none | 0.02× | 0.02× |
| `metal/f32/reduce_to_shape/outer_axis/offset_r128_c32` | none | 0.06× | 0.07× |
| `metal/f32/reduce_to_shape/outer_axis/offset_r128_c5` | none | 0.27× | 0.33× |
| `metal/f32/reduce_to_shape/outer_axis/offset_r128_c65` | none | 0.04× | 0.04× |
| `metal/f32/reduce_to_shape/outer_axis/offset_r16_c129` | none | 0.12× | 0.12× |
| `metal/f32/reduce_to_shape/outer_axis/offset_r16_c32` | none | 0.38× | 0.40× |
| `metal/f32/reduce_to_shape/outer_axis/offset_r16_c5` | none | 0.97× | 0.89× |
| `metal/f32/reduce_to_shape/outer_axis/offset_r16_c65` | none | 0.21× | 0.22× |
| `metal/f32/reduce_to_shape/outer_axis/offset_r33_c129` | none | 0.06× | 0.09× |
| `metal/f32/reduce_to_shape/outer_axis/offset_r33_c32` | none | 0.18× | 0.20× |
| `metal/f32/reduce_to_shape/outer_axis/offset_r33_c5` | none | 0.65× | 0.69× |
| `metal/f32/reduce_to_shape/outer_axis/offset_r33_c65` | none | 0.11× | 0.11× |
| `metal/f32/reduce_to_shape/outer_axis/offset_r3_c129` | none | 0.52× | 0.49× |
| `metal/f32/reduce_to_shape/outer_axis/offset_r3_c32` | none | 0.84× | 0.81× |
| `metal/f32/reduce_to_shape/outer_axis/offset_r3_c5` | none | 1.02× | 0.81× |
| `metal/f32/reduce_to_shape/outer_axis/offset_r3_c65` | none | 0.63× | 0.69× |
| `metal/f32/reduce_to_shape/ragged/dense_r128_c129` | none | 0.93× | 0.96× |
| `metal/f32/reduce_to_shape/ragged/dense_r128_c32` | none | 1.01× | 0.94× |
| `metal/f32/reduce_to_shape/ragged/dense_r128_c5` | none | 0.96× | 0.98× |
| `metal/f32/reduce_to_shape/ragged/dense_r128_c65` | none | 0.90× | 0.93× |
| `metal/f32/reduce_to_shape/ragged/dense_r16_c129` | none | 1.00× | 1.00× |
| `metal/f32/reduce_to_shape/ragged/dense_r16_c32` | none | 1.04× | 0.99× |
| `metal/f32/reduce_to_shape/ragged/dense_r16_c5` | none | 1.04× | 0.99× |
| `metal/f32/reduce_to_shape/ragged/dense_r16_c65` | none | 1.06× | 1.00× |
| `metal/f32/reduce_to_shape/ragged/dense_r33_c129` | none | 0.91× | 0.92× |
| `metal/f32/reduce_to_shape/ragged/dense_r33_c32` | none | 0.98× | 0.98× |
| `metal/f32/reduce_to_shape/ragged/dense_r33_c5` | none | 1.00× | 1.07× |
| `metal/f32/reduce_to_shape/ragged/dense_r33_c65` | none | 1.02× | 1.01× |
| `metal/f32/reduce_to_shape/ragged/dense_r3_c129` | none | 1.01× | 1.02× |
| `metal/f32/reduce_to_shape/ragged/dense_r3_c32` | none | 0.97× | 0.99× |
| `metal/f32/reduce_to_shape/ragged/dense_r3_c5` | none | 0.87× | 1.02× |
| `metal/f32/reduce_to_shape/ragged/dense_r3_c65` | none | 1.02× | 1.01× |
| `metal/f32/reduce_to_shape/ragged/offset_r128_c129` | none | 0.90× | 0.95× |
| `metal/f32/reduce_to_shape/ragged/offset_r128_c32` | none | 0.97× | 0.97× |
| `metal/f32/reduce_to_shape/ragged/offset_r128_c5` | none | 1.05× | 1.06× |
| `metal/f32/reduce_to_shape/ragged/offset_r128_c65` | none | 0.92× | 0.93× |
| `metal/f32/reduce_to_shape/ragged/offset_r16_c129` | none | 1.02× | 0.93× |
| `metal/f32/reduce_to_shape/ragged/offset_r16_c32` | none | 1.04× | 1.08× |
| `metal/f32/reduce_to_shape/ragged/offset_r16_c5` | none | 0.98× | 1.04× |
| `metal/f32/reduce_to_shape/ragged/offset_r16_c65` | none | 1.04× | 1.03× |
| `metal/f32/reduce_to_shape/ragged/offset_r33_c129` | none | 1.04× | 0.90× |
| `metal/f32/reduce_to_shape/ragged/offset_r33_c32` | none | 1.01× | 0.97× |
| `metal/f32/reduce_to_shape/ragged/offset_r33_c5` | none | 1.01× | 0.96× |
| `metal/f32/reduce_to_shape/ragged/offset_r33_c65` | none | 0.97× | 0.97× |
| `metal/f32/reduce_to_shape/ragged/offset_r3_c129` | none | 0.97× | 1.04× |
| `metal/f32/reduce_to_shape/ragged/offset_r3_c32` | none | 0.99× | 0.81× |
| `metal/f32/reduce_to_shape/ragged/offset_r3_c5` | none | 1.35× | 1.20× |
| `metal/f32/reduce_to_shape/ragged/offset_r3_c65` | none | 1.36× | 1.01× |
| `metal/f32/reduce_to_shape/short_axis/dense_r128_c129` | none | 0.02× | 0.02× |
| `metal/f32/reduce_to_shape/short_axis/dense_r128_c32` | none | 0.06× | 0.06× |
| `metal/f32/reduce_to_shape/short_axis/dense_r128_c5` | none | 0.29× | 0.29× |
| `metal/f32/reduce_to_shape/short_axis/dense_r128_c65` | none | 0.03× | 0.03× |
| `metal/f32/reduce_to_shape/short_axis/dense_r16_c129` | none | 0.12× | 0.14× |
| `metal/f32/reduce_to_shape/short_axis/dense_r16_c32` | none | 0.35× | 0.33× |
| `metal/f32/reduce_to_shape/short_axis/dense_r16_c5` | none | 0.82× | 1.01× |
| `metal/f32/reduce_to_shape/short_axis/dense_r16_c65` | none | 0.26× | 0.27× |
| `metal/f32/reduce_to_shape/short_axis/dense_r33_c129` | none | 0.06× | 0.06× |
| `metal/f32/reduce_to_shape/short_axis/dense_r33_c32` | none | 0.25× | 0.27× |
| `metal/f32/reduce_to_shape/short_axis/dense_r33_c5` | none | 0.63× | 0.65× |
| `metal/f32/reduce_to_shape/short_axis/dense_r33_c65` | none | 0.13× | 0.12× |
| `metal/f32/reduce_to_shape/short_axis/dense_r3_c129` | none | 0.48× | 0.49× |
| `metal/f32/reduce_to_shape/short_axis/dense_r3_c32` | none | 0.81× | 0.85× |
| `metal/f32/reduce_to_shape/short_axis/dense_r3_c5` | none | 1.05× | 0.85× |
| `metal/f32/reduce_to_shape/short_axis/dense_r3_c65` | none | 0.63× | 0.69× |
| `metal/f32/reduce_to_shape/short_axis/offset_r128_c129` | none | 0.02× | 0.02× |
| `metal/f32/reduce_to_shape/short_axis/offset_r128_c32` | none | 0.07× | 0.06× |
| `metal/f32/reduce_to_shape/short_axis/offset_r128_c5` | none | 0.26× | 0.27× |
| `metal/f32/reduce_to_shape/short_axis/offset_r128_c65` | none | 0.04× | 0.03× |
| `metal/f32/reduce_to_shape/short_axis/offset_r16_c129` | none | 0.13× | 0.15× |
| `metal/f32/reduce_to_shape/short_axis/offset_r16_c32` | none | 0.37× | 0.39× |
| `metal/f32/reduce_to_shape/short_axis/offset_r16_c5` | none | 1.02× | 1.05× |
| `metal/f32/reduce_to_shape/short_axis/offset_r16_c65` | none | 0.26× | 0.27× |
| `metal/f32/reduce_to_shape/short_axis/offset_r33_c129` | none | 0.08× | 0.06× |
| `metal/f32/reduce_to_shape/short_axis/offset_r33_c32` | none | 0.19× | 0.19× |
| `metal/f32/reduce_to_shape/short_axis/offset_r33_c5` | none | 0.58× | 0.60× |
| `metal/f32/reduce_to_shape/short_axis/offset_r33_c65` | none | 0.12× | 0.15× |
| `metal/f32/reduce_to_shape/short_axis/offset_r3_c129` | none | 0.52× | 0.44× |
| `metal/f32/reduce_to_shape/short_axis/offset_r3_c32` | none | 0.71× | 0.83× |
| `metal/f32/reduce_to_shape/short_axis/offset_r3_c5` | none | 0.91× | 0.87× |
| `metal/f32/reduce_to_shape/short_axis/offset_r3_c65` | none | 0.67× | 0.71× |
