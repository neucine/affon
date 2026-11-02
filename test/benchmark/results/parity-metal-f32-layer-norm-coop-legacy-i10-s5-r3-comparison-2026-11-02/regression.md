# Parity regression check

Target: `Mac16,1/26.6.2/arm64`

- Cells compared: 34
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
| `metal/f32/layer_norm/long_row` | none | 1.05× | 1.06× |
| `metal/f32/layer_norm/long_row/r128_c129` | none | 1.09× | 1.05× |
| `metal/f32/layer_norm/long_row/r256_c31` | none | 0.85× | 1.01× |
| `metal/f32/layer_norm/long_row/r256_c64` | none | 1.20× | 1.11× |
| `metal/f32/layer_norm/long_row/r32_c127` | none | 0.94× | 0.94× |
| `metal/f32/layer_norm/long_row/r32_c128` | none | 0.94× | 0.95× |
| `metal/f32/layer_norm/long_row/r32_c129` | none | 1.18× | 0.92× |
| `metal/f32/layer_norm/long_row/r32_c257` | none | 1.09× | 0.92× |
| `metal/f32/layer_norm/long_row/r32_c63` | none | 0.84× | 0.87× |
| `metal/f32/layer_norm/long_row/r32_c64` | none | 1.13× | 1.20× |
| `metal/f32/layer_norm/long_row/r32_c65` | none | 1.04× | 0.97× |
| `metal/f32/layer_norm/long_row/r33_c255` | none | 0.84× | 1.10× |
| `metal/f32/layer_norm/long_row/r64_c255` | none | 1.05× | 0.98× |
| `metal/f32/layer_norm/outer_axis` | none | 1.25× | 0.95× |
| `metal/f32/layer_norm/outer_axis/r127_c64` | none | 0.99× | 1.00× |
| `metal/f32/layer_norm/outer_axis/r64_c129` | none | 1.20× | 1.00× |
| `metal/f32/layer_norm/ragged` | none | 1.31× | 1.36× |
| `metal/f32/layer_norm/ragged/r17_c65` | none | 0.92× | 0.82× |
| `metal/f32/layer_norm/ragged/r33_c129` | none | 0.90× | 0.91× |
| `metal/f32/layer_norm/short_row` | none | 0.85× | 1.01× |
| `metal/f32/layer_norm/short_row/r16_c31` | none | 0.98× | 0.98× |
| `metal/f32/layer_norm/short_row/r16_c32` | none | 1.11× | 1.02× |
| `metal/f32/layer_norm/short_row/r16_c33` | none | 1.24× | 0.98× |
| `metal/f32/layer_norm/short_row/r31_c16` | none | 1.02× | 0.89× |
| `metal/f32/layer_norm/short_row/r32_c31` | none | 0.98× | 0.90× |
| `metal/f32/layer_norm/short_row/r32_c32` | none | 0.72× | 1.18× |
| `metal/f32/layer_norm/short_row/r32_c33` | none | 1.30× | 1.02× |
| `metal/f32/layer_norm/short_row/r8_c15` | none | 1.01× | 1.01× |
| `metal/f32/layer_norm/short_row/r8_c16` | none | 1.02× | 1.00× |
| `metal/f32/layer_norm/short_row/r8_c17` | none | 1.03× | 1.00× |
| `metal/f32/layer_norm/short_row/r8_c7` | none | 1.00× | 1.06× |
| `metal/f32/layer_norm/strided` | none | 0.74× | 0.87× |
| `metal/f32/layer_norm/strided/r32_c127` | none | 0.72× | 0.96× |
| `metal/f32/layer_norm/strided/r64_c128` | none | 0.88× | 0.81× |
