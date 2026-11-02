# Parity regression check

Target: `Mac16,1/26.6.2/arm64`

- Cells compared: 9
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
| `metal/bf16/embedding/strided` | none | 0.56× | 0.89× |
| `metal/bf16/gather/strided` | none | 0.54× | 0.98× |
| `metal/bf16/index_select/strided` | none | 0.55× | 0.97× |
| `metal/f32/embedding/large` | none | 0.53× | 0.97× |
| `metal/f32/embedding/small` | none | 0.60× | 0.98× |
| `metal/f32/gather/large` | none | 0.58× | 0.90× |
| `metal/f32/gather/small` | none | 0.59× | 0.92× |
| `metal/f32/index_select/large` | none | 0.64× | 0.85× |
| `metal/f32/index_select/small` | none | 0.55× | 0.89× |
