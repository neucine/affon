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
| `metal/bf16/gather/large/offset_axis0_r16_c129` | none | 1.00× | 0.74× |
| `metal/bf16/gather/large/offset_axis1_r16_c129` | none | 0.85× | 1.00× |
| `metal/bf16/gather/strided/axis0_r16_c129` | none | 0.78× | 1.09× |
| `metal/bf16/gather/strided/axis1_r16_c129` | none | 1.01× | 1.05× |
| `metal/f32/gather/large/offset_axis0_r16_c129` | none | 1.32× | 0.83× |
| `metal/f32/gather/large/offset_axis1_r16_c129` | none | 1.27× | 0.97× |
| `metal/f32/gather/strided/axis0_r16_c129` | none | 0.74× | 0.80× |
| `metal/f32/gather/strided/axis1_r16_c129` | none | 1.27× | 0.74× |
