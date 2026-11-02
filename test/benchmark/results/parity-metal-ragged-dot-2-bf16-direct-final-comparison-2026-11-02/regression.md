# Parity regression check

Target: `Mac16,1/26.6.2/arm64`

- Cells compared: 1
- Added cells: 0
- Removed cells: 1
- Changed case definitions: 0
- New correctness failures: 0
- Diagnostic timing suspects: 0
- Target matches: yes
- Measurement protocol matches: no

Timing changes are diagnostic and do not fail this check. Timing is compared only when target and protocol match. No timing baseline is accepted by this comparison.

| Cell | Correctness regression | Operator relative timing | Program relative timing |
| --- | --- | ---: | ---: |
| `metal/bf16/dot/ragged` | none | — | — |

## Removed cells

- `metal/f32/dot/ragged`
