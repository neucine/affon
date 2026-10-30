# Parity regression check

Target: `Mac16,1/26.6.2/arm64`

- Cells compared: 3
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
| `metal/bf16/cat/strided/r64_c127` | none | 0.19× | 0.18× |
| `metal/bf16/stack/strided/r64_c127` | none | 0.18× | 0.20× |
| `metal/f32/one_hot/invalid_index/b8_n33_c33` | none | 0.29× | 0.26× |
