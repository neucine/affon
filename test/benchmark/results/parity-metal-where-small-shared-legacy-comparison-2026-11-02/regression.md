# Parity regression check

Target: `Mac16,1/26.6.2/arm64`

- Cells compared: 22
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
| `metal/bf16/where/broadcast` | none | 0.06× | 1.21× |
| `metal/bf16/where/broadcast/r32_c32` | none | 0.17× | 0.85× |
| `metal/bf16/where/broadcast/r64_c128` | none | 1.06× | 0.89× |
| `metal/bf16/where/large_dense` | none | 0.26× | 1.09× |
| `metal/bf16/where/offset` | none | 0.06× | 0.96× |
| `metal/bf16/where/offset/r32_c127` | none | 0.42× | 1.23× |
| `metal/bf16/where/ragged` | none | 0.17× | 1.02× |
| `metal/bf16/where/scalar` | none | 0.04× | 0.79× |
| `metal/bf16/where/small_dense` | none | 0.04× | 0.73× |
| `metal/bf16/where/strided` | none | 0.06× | 0.96× |
| `metal/bf16/where/strided/r32_c127` | none | 0.40× | 1.21× |
| `metal/f32/where/broadcast` | none | 0.05× | 0.86× |
| `metal/f32/where/broadcast/r32_c32` | none | 0.17× | 0.87× |
| `metal/f32/where/broadcast/r64_c128` | none | 1.05× | 0.91× |
| `metal/f32/where/large_dense` | none | 0.26× | 0.93× |
| `metal/f32/where/offset` | none | 0.06× | 0.85× |
| `metal/f32/where/offset/r32_c127` | none | 0.40× | 0.99× |
| `metal/f32/where/ragged` | none | 0.17× | 0.92× |
| `metal/f32/where/scalar` | none | 0.03× | 0.86× |
| `metal/f32/where/small_dense` | none | 0.04× | 0.83× |
| `metal/f32/where/strided` | none | 0.06× | 0.84× |
| `metal/f32/where/strided/r32_c127` | none | 0.40× | 0.99× |
