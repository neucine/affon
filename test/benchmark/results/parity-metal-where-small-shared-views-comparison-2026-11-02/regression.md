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
| `metal/bf16/where/broadcast/grid_r16_c129` | none | 0.23× | 0.87× |
| `metal/bf16/where/broadcast/grid_r16_c17` | none | 0.09× | 0.80× |
| `metal/bf16/where/broadcast/grid_r16_c3` | none | 0.06× | 0.94× |
| `metal/bf16/where/broadcast/grid_r16_c64` | none | 0.14× | 0.94× |
| `metal/bf16/where/broadcast/grid_r33_c129` | none | 1.06× | 0.96× |
| `metal/bf16/where/broadcast/grid_r33_c17` | none | 0.14× | 1.12× |
| `metal/bf16/where/broadcast/grid_r33_c3` | none | 0.07× | 0.98× |
| `metal/bf16/where/broadcast/grid_r33_c64` | none | 0.26× | 1.17× |
| `metal/bf16/where/broadcast/grid_r3_c129` | none | 0.08× | 0.97× |
| `metal/bf16/where/broadcast/grid_r3_c17` | none | 0.04× | 0.77× |
| `metal/bf16/where/broadcast/grid_r3_c3` | none | 0.04× | 0.98× |
| `metal/bf16/where/broadcast/grid_r3_c64` | none | 0.05× | 0.97× |
| `metal/bf16/where/broadcast/grid_r7_c129` | none | 0.13× | 1.00× |
| `metal/bf16/where/broadcast/grid_r7_c17` | none | 0.06× | 0.97× |
| `metal/bf16/where/broadcast/grid_r7_c3` | none | 0.04× | 0.84× |
| `metal/bf16/where/broadcast/grid_r7_c64` | none | 0.09× | 1.26× |
| `metal/bf16/where/offset/grid_r16_c129` | none | 0.26× | 1.06× |
| `metal/bf16/where/offset/grid_r16_c17` | none | 0.08× | 0.99× |
| `metal/bf16/where/offset/grid_r16_c3` | none | 0.05× | 0.96× |
| `metal/bf16/where/offset/grid_r16_c64` | none | 0.17× | 0.92× |
| `metal/bf16/where/offset/grid_r33_c129` | none | 0.91× | 0.98× |
| `metal/bf16/where/offset/grid_r33_c17` | none | 0.13× | 0.99× |
| `metal/bf16/where/offset/grid_r33_c3` | none | 0.07× | 0.98× |
| `metal/bf16/where/offset/grid_r33_c64` | none | 0.28× | 1.22× |
| `metal/bf16/where/offset/grid_r3_c129` | none | 0.08× | 1.18× |
| `metal/bf16/where/offset/grid_r3_c17` | none | 0.05× | 0.96× |
| `metal/bf16/where/offset/grid_r3_c3` | none | 0.04× | 0.91× |
| `metal/bf16/where/offset/grid_r3_c64` | none | 0.06× | 0.98× |
| `metal/bf16/where/offset/grid_r7_c129` | none | 0.12× | 0.92× |
| `metal/bf16/where/offset/grid_r7_c17` | none | 0.07× | 0.97× |
| `metal/bf16/where/offset/grid_r7_c3` | none | 0.05× | 1.01× |
| `metal/bf16/where/offset/grid_r7_c64` | none | 0.09× | 1.03× |
| `metal/bf16/where/small_dense/grid_r16_c129` | none | 0.23× | 1.03× |
| `metal/bf16/where/small_dense/grid_r16_c17` | none | 0.08× | 1.01× |
| `metal/bf16/where/small_dense/grid_r16_c3` | none | 0.05× | 0.90× |
| `metal/bf16/where/small_dense/grid_r16_c64` | none | 0.15× | 0.95× |
| `metal/bf16/where/small_dense/grid_r33_c129` | none | 0.85× | 0.93× |
| `metal/bf16/where/small_dense/grid_r33_c17` | none | 0.13× | 0.84× |
| `metal/bf16/where/small_dense/grid_r33_c3` | none | 0.06× | 0.75× |
| `metal/bf16/where/small_dense/grid_r33_c64` | none | 0.26× | 1.00× |
| `metal/bf16/where/small_dense/grid_r3_c129` | none | 0.08× | 0.99× |
| `metal/bf16/where/small_dense/grid_r3_c17` | none | 0.05× | 0.97× |
| `metal/bf16/where/small_dense/grid_r3_c3` | none | 0.04× | 0.98× |
| `metal/bf16/where/small_dense/grid_r3_c64` | none | 0.06× | 1.17× |
| `metal/bf16/where/small_dense/grid_r7_c129` | none | 0.12× | 1.09× |
| `metal/bf16/where/small_dense/grid_r7_c17` | none | 0.07× | 0.96× |
| `metal/bf16/where/small_dense/grid_r7_c3` | none | 0.04× | 0.93× |
| `metal/bf16/where/small_dense/grid_r7_c64` | none | 0.09× | 1.03× |
| `metal/bf16/where/strided/grid_r16_c129` | none | 0.25× | 1.19× |
| `metal/bf16/where/strided/grid_r16_c17` | none | 0.10× | 1.01× |
| `metal/bf16/where/strided/grid_r16_c3` | none | 0.05× | 0.96× |
| `metal/bf16/where/strided/grid_r16_c64` | none | 0.14× | 0.94× |
| `metal/bf16/where/strided/grid_r33_c129` | none | 1.05× | 0.97× |
| `metal/bf16/where/strided/grid_r33_c17` | none | 0.14× | 1.15× |
| `metal/bf16/where/strided/grid_r33_c3` | none | 0.06× | 0.96× |
| `metal/bf16/where/strided/grid_r33_c64` | none | 0.28× | 0.98× |
| `metal/bf16/where/strided/grid_r3_c129` | none | 0.08× | 1.14× |
| `metal/bf16/where/strided/grid_r3_c17` | none | 0.05× | 1.12× |
| `metal/bf16/where/strided/grid_r3_c3` | none | 0.04× | 0.96× |
| `metal/bf16/where/strided/grid_r3_c64` | none | 0.06× | 0.93× |
| `metal/bf16/where/strided/grid_r7_c129` | none | 0.15× | 0.97× |
| `metal/bf16/where/strided/grid_r7_c17` | none | 0.05× | 1.13× |
| `metal/bf16/where/strided/grid_r7_c3` | none | 0.04× | 1.08× |
| `metal/bf16/where/strided/grid_r7_c64` | none | 0.09× | 1.20× |
| `metal/f32/where/broadcast/grid_r16_c129` | none | 0.24× | 1.00× |
| `metal/f32/where/broadcast/grid_r16_c17` | none | 0.07× | 0.85× |
| `metal/f32/where/broadcast/grid_r16_c3` | none | 0.05× | 0.82× |
| `metal/f32/where/broadcast/grid_r16_c64` | none | 0.15× | 1.16× |
| `metal/f32/where/broadcast/grid_r33_c129` | none | 1.20× | 1.00× |
| `metal/f32/where/broadcast/grid_r33_c17` | none | 0.13× | 1.05× |
| `metal/f32/where/broadcast/grid_r33_c3` | none | 0.07× | 0.90× |
| `metal/f32/where/broadcast/grid_r33_c64` | none | 0.26× | 0.88× |
| `metal/f32/where/broadcast/grid_r3_c129` | none | 0.08× | 0.89× |
| `metal/f32/where/broadcast/grid_r3_c17` | none | 0.04× | 0.86× |
| `metal/f32/where/broadcast/grid_r3_c3` | none | 0.04× | 0.83× |
| `metal/f32/where/broadcast/grid_r3_c64` | none | 0.07× | 0.88× |
| `metal/f32/where/broadcast/grid_r7_c129` | none | 0.13× | 0.86× |
| `metal/f32/where/broadcast/grid_r7_c17` | none | 0.07× | 0.86× |
| `metal/f32/where/broadcast/grid_r7_c3` | none | 0.04× | 0.81× |
| `metal/f32/where/broadcast/grid_r7_c64` | none | 0.10× | 0.84× |
| `metal/f32/where/offset/grid_r16_c129` | none | 0.25× | 0.95× |
| `metal/f32/where/offset/grid_r16_c17` | none | 0.08× | 0.75× |
| `metal/f32/where/offset/grid_r16_c3` | none | 0.06× | 0.84× |
| `metal/f32/where/offset/grid_r16_c64` | none | 0.15× | 1.10× |
| `metal/f32/where/offset/grid_r33_c129` | none | 0.87× | 1.06× |
| `metal/f32/where/offset/grid_r33_c17` | none | 0.13× | 0.83× |
| `metal/f32/where/offset/grid_r33_c3` | none | 0.09× | 0.96× |
| `metal/f32/where/offset/grid_r33_c64` | none | 0.24× | 0.92× |
| `metal/f32/where/offset/grid_r3_c129` | none | 0.07× | 0.85× |
| `metal/f32/where/offset/grid_r3_c17` | none | 0.05× | 1.07× |
| `metal/f32/where/offset/grid_r3_c3` | none | 0.04× | 0.82× |
| `metal/f32/where/offset/grid_r3_c64` | none | 0.06× | 0.93× |
| `metal/f32/where/offset/grid_r7_c129` | none | 0.12× | 0.87× |
| `metal/f32/where/offset/grid_r7_c17` | none | 0.06× | 0.84× |
| `metal/f32/where/offset/grid_r7_c3` | none | 0.05× | 0.80× |
| `metal/f32/where/offset/grid_r7_c64` | none | 0.11× | 0.80× |
| `metal/f32/where/small_dense/grid_r16_c129` | none | 0.25× | 0.95× |
| `metal/f32/where/small_dense/grid_r16_c17` | none | 0.08× | 0.83× |
| `metal/f32/where/small_dense/grid_r16_c3` | none | 0.05× | 1.11× |
| `metal/f32/where/small_dense/grid_r16_c64` | none | 0.14× | 0.83× |
| `metal/f32/where/small_dense/grid_r33_c129` | none | 0.81× | 0.98× |
| `metal/f32/where/small_dense/grid_r33_c17` | none | 0.12× | 0.89× |
| `metal/f32/where/small_dense/grid_r33_c3` | none | 0.06× | 1.15× |
| `metal/f32/where/small_dense/grid_r33_c64` | none | 0.25× | 0.95× |
| `metal/f32/where/small_dense/grid_r3_c129` | none | 0.07× | 1.06× |
| `metal/f32/where/small_dense/grid_r3_c17` | none | 0.04× | 0.84× |
| `metal/f32/where/small_dense/grid_r3_c3` | none | 0.04× | 0.74× |
| `metal/f32/where/small_dense/grid_r3_c64` | none | 0.06× | 0.88× |
| `metal/f32/where/small_dense/grid_r7_c129` | none | 0.12× | 0.90× |
| `metal/f32/where/small_dense/grid_r7_c17` | none | 0.05× | 0.76× |
| `metal/f32/where/small_dense/grid_r7_c3` | none | 0.04× | 1.04× |
| `metal/f32/where/small_dense/grid_r7_c64` | none | 0.09× | 0.80× |
| `metal/f32/where/strided/grid_r16_c129` | none | 0.27× | 1.00× |
| `metal/f32/where/strided/grid_r16_c17` | none | 0.10× | 0.89× |
| `metal/f32/where/strided/grid_r16_c3` | none | 0.05× | 0.82× |
| `metal/f32/where/strided/grid_r16_c64` | none | 0.18× | 0.87× |
| `metal/f32/where/strided/grid_r33_c129` | none | 1.02× | 1.27× |
| `metal/f32/where/strided/grid_r33_c17` | none | 0.13× | 0.88× |
| `metal/f32/where/strided/grid_r33_c3` | none | 0.08× | 0.89× |
| `metal/f32/where/strided/grid_r33_c64` | none | 0.33× | 1.07× |
| `metal/f32/where/strided/grid_r3_c129` | none | 0.08× | 0.88× |
| `metal/f32/where/strided/grid_r3_c17` | none | 0.05× | 0.77× |
| `metal/f32/where/strided/grid_r3_c3` | none | 0.04× | 0.83× |
| `metal/f32/where/strided/grid_r3_c64` | none | 0.06× | 0.85× |
| `metal/f32/where/strided/grid_r7_c129` | none | 0.13× | 0.97× |
| `metal/f32/where/strided/grid_r7_c17` | none | 0.06× | 0.84× |
| `metal/f32/where/strided/grid_r7_c3` | none | 0.04× | 0.80× |
| `metal/f32/where/strided/grid_r7_c64` | none | 0.08× | 1.06× |
