# Parity regression check

Target: `Mac16,1/26.6.2/arm64`

- Cells compared: 16
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
| `metal/bf16/where/broadcast/grid_r16_c129` | none | 1.17× | 1.09× |
| `metal/bf16/where/broadcast/grid_r7_c17` | none | 1.00× | 1.09× |
| `metal/bf16/where/offset/grid_r16_c129` | none | 1.13× | 0.82× |
| `metal/bf16/where/offset/grid_r7_c17` | none | 1.02× | 1.12× |
| `metal/bf16/where/small_dense/grid_r16_c129` | none | 0.98× | 0.88× |
| `metal/bf16/where/small_dense/grid_r7_c17` | none | 1.02× | 1.14× |
| `metal/bf16/where/strided/grid_r16_c129` | none | 1.02× | 1.08× |
| `metal/bf16/where/strided/grid_r7_c17` | none | 1.07× | 1.01× |
| `metal/f32/where/broadcast/grid_r16_c129` | none | 1.02× | 1.06× |
| `metal/f32/where/broadcast/grid_r7_c17` | none | 1.01× | 0.82× |
| `metal/f32/where/offset/grid_r16_c129` | none | 0.97× | 0.85× |
| `metal/f32/where/offset/grid_r7_c17` | none | 1.03× | 1.06× |
| `metal/f32/where/small_dense/grid_r16_c129` | none | 1.05× | 1.05× |
| `metal/f32/where/small_dense/grid_r7_c17` | none | 1.48× | 1.23× |
| `metal/f32/where/strided/grid_r16_c129` | none | 1.05× | 1.01× |
| `metal/f32/where/strided/grid_r7_c17` | none | 1.03× | 0.99× |
