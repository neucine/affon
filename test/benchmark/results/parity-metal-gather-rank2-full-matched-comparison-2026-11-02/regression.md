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
| `metal/bf16/gather/large/offset_axis0_r16_c129` | none | 0.85× | 0.76× |
| `metal/bf16/gather/large/offset_axis0_r16_c17` | none | 1.04× | 1.12× |
| `metal/bf16/gather/large/offset_axis0_r16_c3` | none | 1.04× | 1.38× |
| `metal/bf16/gather/large/offset_axis0_r16_c64` | none | 0.92× | 1.06× |
| `metal/bf16/gather/large/offset_axis0_r33_c129` | none | 0.78× | 0.66× |
| `metal/bf16/gather/large/offset_axis0_r33_c17` | none | 1.00× | 0.97× |
| `metal/bf16/gather/large/offset_axis0_r33_c3` | none | 0.91× | 1.17× |
| `metal/bf16/gather/large/offset_axis0_r33_c64` | none | 0.86× | 0.78× |
| `metal/bf16/gather/large/offset_axis0_r3_c129` | none | 1.07× | 0.97× |
| `metal/bf16/gather/large/offset_axis0_r3_c17` | none | 0.99× | 0.98× |
| `metal/bf16/gather/large/offset_axis0_r3_c3` | none | 1.04× | 1.18× |
| `metal/bf16/gather/large/offset_axis0_r3_c64` | none | 1.17× | 1.22× |
| `metal/bf16/gather/large/offset_axis0_r7_c129` | none | 0.94× | 0.94× |
| `metal/bf16/gather/large/offset_axis0_r7_c17` | none | 0.92× | 1.08× |
| `metal/bf16/gather/large/offset_axis0_r7_c3` | none | 0.99× | 1.03× |
| `metal/bf16/gather/large/offset_axis0_r7_c64` | none | 1.00× | 0.97× |
| `metal/bf16/gather/large/offset_axis1_r16_c129` | none | 0.84× | 0.81× |
| `metal/bf16/gather/large/offset_axis1_r16_c17` | none | 1.22× | 0.91× |
| `metal/bf16/gather/large/offset_axis1_r16_c3` | none | 0.97× | 0.96× |
| `metal/bf16/gather/large/offset_axis1_r16_c64` | none | 1.12× | 0.92× |
| `metal/bf16/gather/large/offset_axis1_r33_c129` | none | 0.71× | 0.72× |
| `metal/bf16/gather/large/offset_axis1_r33_c17` | none | 0.93× | 0.92× |
| `metal/bf16/gather/large/offset_axis1_r33_c3` | none | 0.94× | 0.96× |
| `metal/bf16/gather/large/offset_axis1_r33_c64` | none | 0.90× | 0.77× |
| `metal/bf16/gather/large/offset_axis1_r3_c129` | none | 1.02× | 1.03× |
| `metal/bf16/gather/large/offset_axis1_r3_c17` | none | 1.17× | 0.94× |
| `metal/bf16/gather/large/offset_axis1_r3_c3` | none | 1.10× | 1.19× |
| `metal/bf16/gather/large/offset_axis1_r3_c64` | none | 0.95× | 1.06× |
| `metal/bf16/gather/large/offset_axis1_r7_c129` | none | 0.87× | 0.92× |
| `metal/bf16/gather/large/offset_axis1_r7_c17` | none | 0.97× | 1.15× |
| `metal/bf16/gather/large/offset_axis1_r7_c3` | none | 1.13× | 1.03× |
| `metal/bf16/gather/large/offset_axis1_r7_c64` | none | 1.01× | 0.93× |
| `metal/bf16/gather/strided/axis0_r16_c129` | none | 0.78× | 0.74× |
| `metal/bf16/gather/strided/axis0_r16_c17` | none | 0.96× | 1.27× |
| `metal/bf16/gather/strided/axis0_r16_c3` | none | 1.09× | 1.16× |
| `metal/bf16/gather/strided/axis0_r16_c64` | none | 0.86× | 0.96× |
| `metal/bf16/gather/strided/axis0_r33_c129` | none | 0.70× | 0.68× |
| `metal/bf16/gather/strided/axis0_r33_c17` | none | 0.89× | 1.11× |
| `metal/bf16/gather/strided/axis0_r33_c3` | none | 1.19× | 1.05× |
| `metal/bf16/gather/strided/axis0_r33_c64` | none | 0.76× | 0.85× |
| `metal/bf16/gather/strided/axis0_r3_c129` | none | 0.88× | 0.96× |
| `metal/bf16/gather/strided/axis0_r3_c17` | none | 1.11× | 1.19× |
| `metal/bf16/gather/strided/axis0_r3_c3` | none | 1.20× | 1.02× |
| `metal/bf16/gather/strided/axis0_r3_c64` | none | 0.98× | 1.07× |
| `metal/bf16/gather/strided/axis0_r7_c129` | none | 0.99× | 0.95× |
| `metal/bf16/gather/strided/axis0_r7_c17` | none | 1.18× | 0.98× |
| `metal/bf16/gather/strided/axis0_r7_c3` | none | 0.99× | 1.28× |
| `metal/bf16/gather/strided/axis0_r7_c64` | none | 0.99× | 0.95× |
| `metal/bf16/gather/strided/axis1_r16_c129` | none | 0.78× | 0.78× |
| `metal/bf16/gather/strided/axis1_r16_c17` | none | 1.05× | 0.98× |
| `metal/bf16/gather/strided/axis1_r16_c3` | none | 0.98× | 1.20× |
| `metal/bf16/gather/strided/axis1_r16_c64` | none | 1.11× | 0.82× |
| `metal/bf16/gather/strided/axis1_r33_c129` | none | 0.70× | 0.68× |
| `metal/bf16/gather/strided/axis1_r33_c17` | none | 0.98× | 1.00× |
| `metal/bf16/gather/strided/axis1_r33_c3` | none | 0.95× | 1.00× |
| `metal/bf16/gather/strided/axis1_r33_c64` | none | 0.80× | 0.82× |
| `metal/bf16/gather/strided/axis1_r3_c129` | none | 1.12× | 1.20× |
| `metal/bf16/gather/strided/axis1_r3_c17` | none | 0.96× | 1.26× |
| `metal/bf16/gather/strided/axis1_r3_c3` | none | 1.01× | 0.95× |
| `metal/bf16/gather/strided/axis1_r3_c64` | none | 0.97× | 1.13× |
| `metal/bf16/gather/strided/axis1_r7_c129` | none | 1.19× | 1.03× |
| `metal/bf16/gather/strided/axis1_r7_c17` | none | 1.09× | 0.96× |
| `metal/bf16/gather/strided/axis1_r7_c3` | none | 0.95× | 1.08× |
| `metal/bf16/gather/strided/axis1_r7_c64` | none | 1.03× | 1.25× |
| `metal/f32/gather/large/offset_axis0_r16_c129` | none | 0.93× | 0.82× |
| `metal/f32/gather/large/offset_axis0_r16_c17` | none | 0.98× | 0.98× |
| `metal/f32/gather/large/offset_axis0_r16_c3` | none | 1.08× | 0.98× |
| `metal/f32/gather/large/offset_axis0_r16_c64` | none | 0.88× | 0.93× |
| `metal/f32/gather/large/offset_axis0_r33_c129` | none | 0.71× | 0.93× |
| `metal/f32/gather/large/offset_axis0_r33_c17` | none | 0.91× | 0.97× |
| `metal/f32/gather/large/offset_axis0_r33_c3` | none | 0.93× | 0.97× |
| `metal/f32/gather/large/offset_axis0_r33_c64` | none | 0.83× | 0.92× |
| `metal/f32/gather/large/offset_axis0_r3_c129` | none | 1.12× | 1.10× |
| `metal/f32/gather/large/offset_axis0_r3_c17` | none | 1.02× | 1.09× |
| `metal/f32/gather/large/offset_axis0_r3_c3` | none | 0.97× | 1.07× |
| `metal/f32/gather/large/offset_axis0_r3_c64` | none | 1.28× | 0.97× |
| `metal/f32/gather/large/offset_axis0_r7_c129` | none | 0.92× | 0.92× |
| `metal/f32/gather/large/offset_axis0_r7_c17` | none | 1.16× | 1.05× |
| `metal/f32/gather/large/offset_axis0_r7_c3` | none | 1.19× | 1.06× |
| `metal/f32/gather/large/offset_axis0_r7_c64` | none | 1.26× | 1.00× |
| `metal/f32/gather/large/offset_axis1_r16_c129` | none | 0.98× | 0.80× |
| `metal/f32/gather/large/offset_axis1_r16_c17` | none | 1.12× | 1.14× |
| `metal/f32/gather/large/offset_axis1_r16_c3` | none | 1.09× | 1.12× |
| `metal/f32/gather/large/offset_axis1_r16_c64` | none | 0.98× | 1.01× |
| `metal/f32/gather/large/offset_axis1_r33_c129` | none | 0.85× | 0.81× |
| `metal/f32/gather/large/offset_axis1_r33_c17` | none | 0.90× | 1.06× |
| `metal/f32/gather/large/offset_axis1_r33_c3` | none | 1.13× | 1.03× |
| `metal/f32/gather/large/offset_axis1_r33_c64` | none | 0.87× | 0.95× |
| `metal/f32/gather/large/offset_axis1_r3_c129` | none | 1.15× | 0.96× |
| `metal/f32/gather/large/offset_axis1_r3_c17` | none | 0.93× | 0.96× |
| `metal/f32/gather/large/offset_axis1_r3_c3` | none | 0.94× | 0.99× |
| `metal/f32/gather/large/offset_axis1_r3_c64` | none | 1.11× | 1.11× |
| `metal/f32/gather/large/offset_axis1_r7_c129` | none | 0.89× | 0.93× |
| `metal/f32/gather/large/offset_axis1_r7_c17` | none | 1.23× | 0.94× |
| `metal/f32/gather/large/offset_axis1_r7_c3` | none | 1.05× | 0.99× |
| `metal/f32/gather/large/offset_axis1_r7_c64` | none | 0.92× | 0.93× |
| `metal/f32/gather/strided/axis0_r16_c129` | none | 1.12× | 0.82× |
| `metal/f32/gather/strided/axis0_r16_c17` | none | 1.15× | 0.89× |
| `metal/f32/gather/strided/axis0_r16_c3` | none | 1.00× | 1.13× |
| `metal/f32/gather/strided/axis0_r16_c64` | none | 0.96× | 0.86× |
| `metal/f32/gather/strided/axis0_r33_c129` | none | 0.70× | 0.73× |
| `metal/f32/gather/strided/axis0_r33_c17` | none | 0.93× | 1.14× |
| `metal/f32/gather/strided/axis0_r33_c3` | none | 1.02× | 0.95× |
| `metal/f32/gather/strided/axis0_r33_c64` | none | 0.76× | 0.97× |
| `metal/f32/gather/strided/axis0_r3_c129` | none | 1.26× | 1.04× |
| `metal/f32/gather/strided/axis0_r3_c17` | none | 1.18× | 1.06× |
| `metal/f32/gather/strided/axis0_r3_c3` | none | 1.06× | 1.29× |
| `metal/f32/gather/strided/axis0_r3_c64` | none | 0.98× | 1.18× |
| `metal/f32/gather/strided/axis0_r7_c129` | none | 0.96× | 1.04× |
| `metal/f32/gather/strided/axis0_r7_c17` | none | 1.09× | 0.96× |
| `metal/f32/gather/strided/axis0_r7_c3` | none | 1.04× | 1.01× |
| `metal/f32/gather/strided/axis0_r7_c64` | none | 1.08× | 0.98× |
| `metal/f32/gather/strided/axis1_r16_c129` | none | 0.86× | 0.81× |
| `metal/f32/gather/strided/axis1_r16_c17` | none | 0.92× | 1.08× |
| `metal/f32/gather/strided/axis1_r16_c3` | none | 1.16× | 1.03× |
| `metal/f32/gather/strided/axis1_r16_c64` | none | 0.96× | 0.93× |
| `metal/f32/gather/strided/axis1_r33_c129` | none | 0.79× | 0.68× |
| `metal/f32/gather/strided/axis1_r33_c17` | none | 0.92× | 0.91× |
| `metal/f32/gather/strided/axis1_r33_c3` | none | 0.99× | 0.93× |
| `metal/f32/gather/strided/axis1_r33_c64` | none | 0.86× | 0.96× |
| `metal/f32/gather/strided/axis1_r3_c129` | none | 1.07× | 0.97× |
| `metal/f32/gather/strided/axis1_r3_c17` | none | 0.98× | 1.05× |
| `metal/f32/gather/strided/axis1_r3_c3` | none | 1.11× | 1.04× |
| `metal/f32/gather/strided/axis1_r3_c64` | none | 1.15× | 1.08× |
| `metal/f32/gather/strided/axis1_r7_c129` | none | 1.03× | 1.16× |
| `metal/f32/gather/strided/axis1_r7_c17` | none | 0.91× | 1.04× |
| `metal/f32/gather/strided/axis1_r7_c3` | none | 1.23× | 1.07× |
| `metal/f32/gather/strided/axis1_r7_c64` | none | 1.07× | 0.95× |
