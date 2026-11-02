# Parity regression check

Target: `Mac16,1/26.6.2/arm64`

- Cells compared: 8
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
| `metal/f32/layer_norm/strided/inner_offset_r128_c129` | none | 0.85× | 0.83× |
| `metal/f32/layer_norm/strided/inner_offset_r16_c129` | none | 0.99× | 1.08× |
| `metal/f32/layer_norm/strided/inner_offset_r33_c129` | none | 0.94× | 1.00× |
| `metal/f32/layer_norm/strided/inner_offset_r3_c129` | none | 0.92× | 0.84× |
| `metal/f32/layer_norm/strided/inner_strided_r128_c129` | none | 0.93× | 0.79× |
| `metal/f32/layer_norm/strided/inner_strided_r16_c129` | none | 0.93× | 1.10× |
| `metal/f32/layer_norm/strided/inner_strided_r33_c129` | none | 0.80× | 0.90× |
| `metal/f32/layer_norm/strided/inner_strided_r3_c129` | none | 0.89× | 0.90× |
