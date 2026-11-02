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
| `metal/f32/layer_norm/strided/inner_offset_r16_c32` | none | 1.00× | 0.80× |
| `metal/f32/layer_norm/strided/inner_offset_r3_c5` | none | 0.87× | 0.87× |
| `metal/f32/layer_norm/strided/inner_strided_r16_c32` | none | 0.60× | 1.02× |
| `metal/f32/layer_norm/strided/inner_strided_r3_c5` | none | 0.97× | 0.81× |
