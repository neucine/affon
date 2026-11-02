# Parity regression check

Target: `Mac16,1/26.6.2/arm64`

- Cells compared: 4
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
| `metal/f32/layer_norm/short_row/r16_c33` | none | 0.99× | 0.82× |
| `metal/f32/layer_norm/short_row/r32_c31` | none | 0.89× | 0.81× |
| `metal/f32/layer_norm/short_row/r32_c32` | none | 0.72× | 0.96× |
| `metal/f32/layer_norm/short_row/r8_c7` | none | 0.75× | 0.81× |
