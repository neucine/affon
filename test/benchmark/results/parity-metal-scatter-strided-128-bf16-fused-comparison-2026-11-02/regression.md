# Parity regression check

Target: `Mac16,1/26.6.2/arm64`

- Cells compared: 64
- Added cells: 0
- Removed cells: 64
- Changed case definitions: 0
- New correctness failures: 0
- Diagnostic timing suspects: 0
- Target matches: yes
- Measurement protocol matches: no

Timing changes are diagnostic and do not fail this check. Timing is compared only when target and protocol match. No timing baseline is accepted by this comparison.

| Cell | Correctness regression | Operator relative timing | Program relative timing |
| --- | --- | ---: | ---: |
| `metal/bf16/scatter_add/strided/cols_axis0_r16_c129` | none | — | — |
| `metal/bf16/scatter_add/strided/cols_axis0_r16_c17` | none | — | — |
| `metal/bf16/scatter_add/strided/cols_axis0_r16_c3` | none | — | — |
| `metal/bf16/scatter_add/strided/cols_axis0_r16_c64` | none | — | — |
| `metal/bf16/scatter_add/strided/cols_axis0_r33_c129` | none | — | — |
| `metal/bf16/scatter_add/strided/cols_axis0_r33_c17` | none | — | — |
| `metal/bf16/scatter_add/strided/cols_axis0_r33_c3` | none | — | — |
| `metal/bf16/scatter_add/strided/cols_axis0_r33_c64` | none | — | — |
| `metal/bf16/scatter_add/strided/cols_axis0_r3_c129` | none | — | — |
| `metal/bf16/scatter_add/strided/cols_axis0_r3_c17` | none | — | — |
| `metal/bf16/scatter_add/strided/cols_axis0_r3_c3` | none | — | — |
| `metal/bf16/scatter_add/strided/cols_axis0_r3_c64` | none | — | — |
| `metal/bf16/scatter_add/strided/cols_axis0_r7_c129` | none | — | — |
| `metal/bf16/scatter_add/strided/cols_axis0_r7_c17` | none | — | — |
| `metal/bf16/scatter_add/strided/cols_axis0_r7_c3` | none | — | — |
| `metal/bf16/scatter_add/strided/cols_axis0_r7_c64` | none | — | — |
| `metal/bf16/scatter_add/strided/cols_axis1_r16_c129` | none | — | — |
| `metal/bf16/scatter_add/strided/cols_axis1_r16_c17` | none | — | — |
| `metal/bf16/scatter_add/strided/cols_axis1_r16_c3` | none | — | — |
| `metal/bf16/scatter_add/strided/cols_axis1_r16_c64` | none | — | — |
| `metal/bf16/scatter_add/strided/cols_axis1_r33_c129` | none | — | — |
| `metal/bf16/scatter_add/strided/cols_axis1_r33_c17` | none | — | — |
| `metal/bf16/scatter_add/strided/cols_axis1_r33_c3` | none | — | — |
| `metal/bf16/scatter_add/strided/cols_axis1_r33_c64` | none | — | — |
| `metal/bf16/scatter_add/strided/cols_axis1_r3_c129` | none | — | — |
| `metal/bf16/scatter_add/strided/cols_axis1_r3_c17` | none | — | — |
| `metal/bf16/scatter_add/strided/cols_axis1_r3_c3` | none | — | — |
| `metal/bf16/scatter_add/strided/cols_axis1_r3_c64` | none | — | — |
| `metal/bf16/scatter_add/strided/cols_axis1_r7_c129` | none | — | — |
| `metal/bf16/scatter_add/strided/cols_axis1_r7_c17` | none | — | — |
| `metal/bf16/scatter_add/strided/cols_axis1_r7_c3` | none | — | — |
| `metal/bf16/scatter_add/strided/cols_axis1_r7_c64` | none | — | — |
| `metal/bf16/scatter_add/strided/rows_axis0_r16_c129` | none | — | — |
| `metal/bf16/scatter_add/strided/rows_axis0_r16_c17` | none | — | — |
| `metal/bf16/scatter_add/strided/rows_axis0_r16_c3` | none | — | — |
| `metal/bf16/scatter_add/strided/rows_axis0_r16_c64` | none | — | — |
| `metal/bf16/scatter_add/strided/rows_axis0_r33_c129` | none | — | — |
| `metal/bf16/scatter_add/strided/rows_axis0_r33_c17` | none | — | — |
| `metal/bf16/scatter_add/strided/rows_axis0_r33_c3` | none | — | — |
| `metal/bf16/scatter_add/strided/rows_axis0_r33_c64` | none | — | — |
| `metal/bf16/scatter_add/strided/rows_axis0_r3_c129` | none | — | — |
| `metal/bf16/scatter_add/strided/rows_axis0_r3_c17` | none | — | — |
| `metal/bf16/scatter_add/strided/rows_axis0_r3_c3` | none | — | — |
| `metal/bf16/scatter_add/strided/rows_axis0_r3_c64` | none | — | — |
| `metal/bf16/scatter_add/strided/rows_axis0_r7_c129` | none | — | — |
| `metal/bf16/scatter_add/strided/rows_axis0_r7_c17` | none | — | — |
| `metal/bf16/scatter_add/strided/rows_axis0_r7_c3` | none | — | — |
| `metal/bf16/scatter_add/strided/rows_axis0_r7_c64` | none | — | — |
| `metal/bf16/scatter_add/strided/rows_axis1_r16_c129` | none | — | — |
| `metal/bf16/scatter_add/strided/rows_axis1_r16_c17` | none | — | — |
| `metal/bf16/scatter_add/strided/rows_axis1_r16_c3` | none | — | — |
| `metal/bf16/scatter_add/strided/rows_axis1_r16_c64` | none | — | — |
| `metal/bf16/scatter_add/strided/rows_axis1_r33_c129` | none | — | — |
| `metal/bf16/scatter_add/strided/rows_axis1_r33_c17` | none | — | — |
| `metal/bf16/scatter_add/strided/rows_axis1_r33_c3` | none | — | — |
| `metal/bf16/scatter_add/strided/rows_axis1_r33_c64` | none | — | — |
| `metal/bf16/scatter_add/strided/rows_axis1_r3_c129` | none | — | — |
| `metal/bf16/scatter_add/strided/rows_axis1_r3_c17` | none | — | — |
| `metal/bf16/scatter_add/strided/rows_axis1_r3_c3` | none | — | — |
| `metal/bf16/scatter_add/strided/rows_axis1_r3_c64` | none | — | — |
| `metal/bf16/scatter_add/strided/rows_axis1_r7_c129` | none | — | — |
| `metal/bf16/scatter_add/strided/rows_axis1_r7_c17` | none | — | — |
| `metal/bf16/scatter_add/strided/rows_axis1_r7_c3` | none | — | — |
| `metal/bf16/scatter_add/strided/rows_axis1_r7_c64` | none | — | — |

## Removed cells

- `metal/f32/scatter_add/strided/cols_axis0_r16_c129`
- `metal/f32/scatter_add/strided/cols_axis0_r16_c17`
- `metal/f32/scatter_add/strided/cols_axis0_r16_c3`
- `metal/f32/scatter_add/strided/cols_axis0_r16_c64`
- `metal/f32/scatter_add/strided/cols_axis0_r33_c129`
- `metal/f32/scatter_add/strided/cols_axis0_r33_c17`
- `metal/f32/scatter_add/strided/cols_axis0_r33_c3`
- `metal/f32/scatter_add/strided/cols_axis0_r33_c64`
- `metal/f32/scatter_add/strided/cols_axis0_r3_c129`
- `metal/f32/scatter_add/strided/cols_axis0_r3_c17`
- `metal/f32/scatter_add/strided/cols_axis0_r3_c3`
- `metal/f32/scatter_add/strided/cols_axis0_r3_c64`
- `metal/f32/scatter_add/strided/cols_axis0_r7_c129`
- `metal/f32/scatter_add/strided/cols_axis0_r7_c17`
- `metal/f32/scatter_add/strided/cols_axis0_r7_c3`
- `metal/f32/scatter_add/strided/cols_axis0_r7_c64`
- `metal/f32/scatter_add/strided/cols_axis1_r16_c129`
- `metal/f32/scatter_add/strided/cols_axis1_r16_c17`
- `metal/f32/scatter_add/strided/cols_axis1_r16_c3`
- `metal/f32/scatter_add/strided/cols_axis1_r16_c64`
- `metal/f32/scatter_add/strided/cols_axis1_r33_c129`
- `metal/f32/scatter_add/strided/cols_axis1_r33_c17`
- `metal/f32/scatter_add/strided/cols_axis1_r33_c3`
- `metal/f32/scatter_add/strided/cols_axis1_r33_c64`
- `metal/f32/scatter_add/strided/cols_axis1_r3_c129`
- `metal/f32/scatter_add/strided/cols_axis1_r3_c17`
- `metal/f32/scatter_add/strided/cols_axis1_r3_c3`
- `metal/f32/scatter_add/strided/cols_axis1_r3_c64`
- `metal/f32/scatter_add/strided/cols_axis1_r7_c129`
- `metal/f32/scatter_add/strided/cols_axis1_r7_c17`
- `metal/f32/scatter_add/strided/cols_axis1_r7_c3`
- `metal/f32/scatter_add/strided/cols_axis1_r7_c64`
- `metal/f32/scatter_add/strided/rows_axis0_r16_c129`
- `metal/f32/scatter_add/strided/rows_axis0_r16_c17`
- `metal/f32/scatter_add/strided/rows_axis0_r16_c3`
- `metal/f32/scatter_add/strided/rows_axis0_r16_c64`
- `metal/f32/scatter_add/strided/rows_axis0_r33_c129`
- `metal/f32/scatter_add/strided/rows_axis0_r33_c17`
- `metal/f32/scatter_add/strided/rows_axis0_r33_c3`
- `metal/f32/scatter_add/strided/rows_axis0_r33_c64`
- `metal/f32/scatter_add/strided/rows_axis0_r3_c129`
- `metal/f32/scatter_add/strided/rows_axis0_r3_c17`
- `metal/f32/scatter_add/strided/rows_axis0_r3_c3`
- `metal/f32/scatter_add/strided/rows_axis0_r3_c64`
- `metal/f32/scatter_add/strided/rows_axis0_r7_c129`
- `metal/f32/scatter_add/strided/rows_axis0_r7_c17`
- `metal/f32/scatter_add/strided/rows_axis0_r7_c3`
- `metal/f32/scatter_add/strided/rows_axis0_r7_c64`
- `metal/f32/scatter_add/strided/rows_axis1_r16_c129`
- `metal/f32/scatter_add/strided/rows_axis1_r16_c17`
- `metal/f32/scatter_add/strided/rows_axis1_r16_c3`
- `metal/f32/scatter_add/strided/rows_axis1_r16_c64`
- `metal/f32/scatter_add/strided/rows_axis1_r33_c129`
- `metal/f32/scatter_add/strided/rows_axis1_r33_c17`
- `metal/f32/scatter_add/strided/rows_axis1_r33_c3`
- `metal/f32/scatter_add/strided/rows_axis1_r33_c64`
- `metal/f32/scatter_add/strided/rows_axis1_r3_c129`
- `metal/f32/scatter_add/strided/rows_axis1_r3_c17`
- `metal/f32/scatter_add/strided/rows_axis1_r3_c3`
- `metal/f32/scatter_add/strided/rows_axis1_r3_c64`
- `metal/f32/scatter_add/strided/rows_axis1_r7_c129`
- `metal/f32/scatter_add/strided/rows_axis1_r7_c17`
- `metal/f32/scatter_add/strided/rows_axis1_r7_c3`
- `metal/f32/scatter_add/strided/rows_axis1_r7_c64`
