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
| `metal/bf16/where/broadcast/grid_r16_c129` | none | 1.02× | 0.98× |
| `metal/bf16/where/offset/grid_r16_c129` | none | 1.05× | 1.01× |
| `metal/bf16/where/small_dense/grid_r16_c129` | none | 1.00× | 0.93× |
| `metal/bf16/where/strided/grid_r16_c129` | none | 1.07× | 0.97× |
| `metal/f32/where/broadcast/grid_r16_c129` | none | 1.03× | 1.23× |
| `metal/f32/where/offset/grid_r16_c129` | none | 0.94× | 1.17× |
| `metal/f32/where/small_dense/grid_r16_c129` | none | 0.88× | 1.19× |
| `metal/f32/where/strided/grid_r16_c129` | none | 1.05× | 1.11× |
