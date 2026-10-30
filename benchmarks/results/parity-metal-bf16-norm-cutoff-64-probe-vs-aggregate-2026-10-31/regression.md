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
| `metal/bf16/layer_norm/long_row/r32_c257` | none | 0.17× | 0.19× |
| `metal/bf16/layer_norm/outer_axis` | none | 0.11× | 0.10× |
| `metal/bf16/layer_norm/strided/r64_c128` | none | 0.26× | 0.29× |
| `metal/bf16/log_softmax/long_row/r128_c129` | none | 0.17× | 0.23× |
| `metal/bf16/log_softmax/long_row/r256_c31` | none | 0.29× | 0.30× |
| `metal/bf16/log_softmax/long_row/r33_c255` | none | 0.36× | 0.30× |
| `metal/bf16/log_softmax/long_row/r64_c255` | none | 0.17× | 0.18× |
| `metal/bf16/log_softmax/outer_axis/r64_c129` | none | 0.30× | 0.37× |
| `metal/bf16/log_softmax/strided/r64_c128` | none | 0.26× | 0.26× |
| `metal/bf16/rms_norm/long_row/r128_c129` | none | 0.14× | 0.13× |
| `metal/bf16/rms_norm/long_row/r256_c31` | none | 0.35× | 0.29× |
| `metal/bf16/rms_norm/long_row/r32_c127` | none | 0.15× | 0.26× |
| `metal/bf16/rms_norm/long_row/r32_c128` | none | 0.18× | 0.25× |
| `metal/bf16/rms_norm/long_row/r32_c129` | none | 0.16× | 0.19× |
| `metal/bf16/rms_norm/long_row/r32_c257` | none | 0.21× | 0.19× |
| `metal/bf16/rms_norm/long_row/r33_c255` | none | 0.17× | 0.24× |
| `metal/bf16/rms_norm/outer_axis` | none | 0.11× | 0.08× |
| `metal/bf16/rms_norm/outer_axis/r127_c64` | none | 0.24× | 0.23× |
| `metal/bf16/rms_norm/outer_axis/r64_c129` | none | 0.17× | 0.22× |
| `metal/bf16/rms_norm/ragged/r33_c129` | none | 0.16× | 0.19× |
| `metal/bf16/rms_norm/strided/r32_c127` | none | 0.20× | 0.18× |
| `metal/bf16/rms_norm/strided/r64_c128` | none | 0.15× | 0.17× |
| `metal/f32/layer_norm/strided/r64_c128` | none | 0.34× | 0.32× |
| `metal/f32/log_softmax/long_row/r256_c31` | none | 0.36× | 0.38× |
| `metal/f32/log_softmax/long_row/r256_c64` | none | 0.19× | 0.18× |
| `metal/f32/log_softmax/long_row/r64_c255` | none | 0.16× | 0.20× |
| `metal/f32/log_softmax/strided/r64_c128` | none | 0.29× | 0.32× |
| `metal/f32/rms_norm/long_row/r128_c129` | none | 0.18× | 0.20× |
| `metal/f32/rms_norm/long_row/r256_c31` | none | 0.28× | 0.31× |
| `metal/f32/rms_norm/long_row/r33_c255` | none | 0.28× | 0.37× |
| `metal/f32/rms_norm/long_row/r64_c255` | none | 0.19× | 0.19× |
| `metal/f32/rms_norm/outer_axis/r127_c64` | none | 0.25× | 0.29× |
