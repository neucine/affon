# Parity regression check

Target: `Mac16,1/26.6.2/arm64`

- Cells compared: 2
- Added cells: 0
- Removed cells: 0
- Changed case definitions: 0
- New correctness failures: 0
- Diagnostic timing suspects: 0
- Target matches: yes
- Measurement protocol matches: no

Timing changes are diagnostic and do not fail this check. Timing is compared only when target and protocol match. No timing baseline is accepted by this comparison.

| Cell | Correctness regression | Operator relative timing | Program relative timing |
| --- | --- | ---: | ---: |
| `metal/bf16/scatter_add/invalid_index` | none | — | — |
| `metal/f32/scatter_add/invalid_index` | none | — | — |
