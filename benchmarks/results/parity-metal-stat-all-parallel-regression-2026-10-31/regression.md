# Parity regression check

Target: `Mac16,1/26.6.2/arm64`

- Cells compared: 4
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
| `metal/f32/std_all/long_axis` | none | — | — |
| `metal/f32/std_all/strided/r64_c127` | none | — | — |
| `metal/f32/variance_all/long_axis` | none | — | — |
| `metal/f32/variance_all/strided/r64_c127` | none | — | — |
