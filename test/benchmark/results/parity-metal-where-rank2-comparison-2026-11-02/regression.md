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
| `metal/bf16/where/broadcast/grid_r7_c17` | none | 1.00× | 1.02× |
| `metal/bf16/where/offset/grid_r7_c17` | none | 0.93× | 1.04× |
| `metal/bf16/where/small_dense/grid_r7_c17` | none | 1.05× | 0.99× |
| `metal/bf16/where/strided/grid_r7_c17` | none | 0.97× | 1.16× |
| `metal/f32/where/broadcast/grid_r7_c17` | none | 0.94× | 0.98× |
| `metal/f32/where/offset/grid_r7_c17` | none | 0.88× | 1.05× |
| `metal/f32/where/small_dense/grid_r7_c17` | none | 1.05× | 1.01× |
| `metal/f32/where/strided/grid_r7_c17` | none | 0.95× | 0.96× |
