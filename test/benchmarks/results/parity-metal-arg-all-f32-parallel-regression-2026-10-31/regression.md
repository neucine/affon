# Parity regression check

Target: `Mac16,1/26.6.2/arm64`

- Cells compared: 10
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
| `metal/f32/argmax_all/long_axis` | none | 0.09× | 0.09× |
| `metal/f32/argmax_all/outer_axis` | none | 0.01× | 0.02× |
| `metal/f32/argmax_all/ragged` | none | 0.25× | 0.31× |
| `metal/f32/argmax_all/short_axis` | none | 1.46× | 1.12× |
| `metal/f32/argmax_all/strided` | none | 0.29× | 0.28× |
| `metal/f32/argmin_all/long_axis` | none | 0.08× | 0.12× |
| `metal/f32/argmin_all/outer_axis` | none | 0.02× | 0.02× |
| `metal/f32/argmin_all/ragged` | none | 0.26× | 0.23× |
| `metal/f32/argmin_all/short_axis` | none | 0.96× | 0.93× |
| `metal/f32/argmin_all/strided` | none | 0.25× | 0.30× |
