# Parity regression check

Target: `Mac16,1/26.6.2/arm64`

- Cells compared: 32
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
| `metal/bf16/layer_norm/long_row/r32_c257` | none | 1.04× | 0.99× |
| `metal/bf16/layer_norm/outer_axis` | none | 1.03× | 1.17× |
| `metal/bf16/layer_norm/strided/r64_c128` | none | 0.96× | 0.94× |
| `metal/bf16/log_softmax/long_row/r128_c129` | none | 0.90× | 1.30× |
| `metal/bf16/log_softmax/long_row/r256_c31` | none | 0.86× | 1.06× |
| `metal/bf16/log_softmax/long_row/r33_c255` | none | 1.05× | 0.90× |
| `metal/bf16/log_softmax/long_row/r64_c255` | none | 0.94× | 1.03× |
| `metal/bf16/log_softmax/outer_axis/r64_c129` | none | 1.11× | 1.21× |
| `metal/bf16/log_softmax/strided/r64_c128` | none | 0.85× | 1.04× |
| `metal/bf16/rms_norm/long_row/r128_c129` | none | 0.89× | 1.03× |
| `metal/bf16/rms_norm/long_row/r256_c31` | none | 1.02× | 1.02× |
| `metal/bf16/rms_norm/long_row/r32_c127` | none | 0.51× | 0.44× |
| `metal/bf16/rms_norm/long_row/r32_c128` | none | 1.01× | 0.82× |
| `metal/bf16/rms_norm/long_row/r32_c129` | none | 0.94× | 0.83× |
| `metal/bf16/rms_norm/long_row/r32_c257` | none | 1.10× | 1.02× |
| `metal/bf16/rms_norm/long_row/r33_c255` | none | 0.86× | 0.97× |
| `metal/bf16/rms_norm/outer_axis` | none | 1.03× | 0.67× |
| `metal/bf16/rms_norm/outer_axis/r127_c64` | none | 1.02× | 1.00× |
| `metal/bf16/rms_norm/outer_axis/r64_c129` | none | 1.04× | 0.98× |
| `metal/bf16/rms_norm/ragged/r33_c129` | none | 0.99× | 1.27× |
| `metal/bf16/rms_norm/strided/r32_c127` | none | 0.49× | 0.49× |
| `metal/bf16/rms_norm/strided/r64_c128` | none | 1.05× | 1.11× |
| `metal/f32/layer_norm/strided/r64_c128` | none | 1.25× | 0.91× |
| `metal/f32/log_softmax/long_row/r256_c31` | none | 1.24× | 1.14× |
| `metal/f32/log_softmax/long_row/r256_c64` | none | 0.99× | 1.02× |
| `metal/f32/log_softmax/long_row/r64_c255` | none | 1.00× | 1.15× |
| `metal/f32/log_softmax/strided/r64_c128` | none | 0.95× | 1.06× |
| `metal/f32/rms_norm/long_row/r128_c129` | none | 1.14× | 1.18× |
| `metal/f32/rms_norm/long_row/r256_c31` | none | 1.00× | 1.03× |
| `metal/f32/rms_norm/long_row/r33_c255` | none | 1.02× | 1.19× |
| `metal/f32/rms_norm/long_row/r64_c255` | none | 1.02× | 1.17× |
| `metal/f32/rms_norm/outer_axis/r127_c64` | none | 1.00× | 1.00× |
