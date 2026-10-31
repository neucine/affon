# Parity regression check

Target: `Mac16,1/26.6.2/arm64`

- Cells compared: 5
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
| `metal/bf16/log_softmax/long_row/r256_c64` | none | 0.19× | 0.16× |
| `metal/bf16/rms_norm/long_row/r256_c64` | none | 0.19× | 0.23× |
| `metal/bf16/rms_norm/long_row/r64_c255` | none | 0.40× | 0.40× |
| `metal/f32/log_softmax/long_row/r128_c129` | none | 0.15× | 0.19× |
| `metal/f32/rms_norm/long_row/r256_c64` | none | 0.21× | 0.20× |
