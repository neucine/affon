# Parity regression check

Target: `Mac16,1/26.6.2/arm64`

- Cells compared: 65
- Added cells: 0
- Removed cells: 65
- Changed case definitions: 0
- New correctness failures: 0
- Diagnostic timing suspects: 0
- Target matches: yes
- Measurement protocol matches: no

Timing changes are diagnostic and do not fail this check. Timing is compared only when target and protocol match. No timing baseline is accepted by this comparison.

| Cell | Correctness regression | Operator relative timing | Program relative timing |
| --- | --- | ---: | ---: |
| `metal/bf16/scatter_add/batched` | none | — | — |
| `metal/bf16/scatter_add/batched/dense_axis1_b2_r16_c129_n8` | none | — | — |
| `metal/bf16/scatter_add/batched/dense_axis1_b2_r16_c17_n8` | none | — | — |
| `metal/bf16/scatter_add/batched/dense_axis1_b2_r16_c5_n8` | none | — | — |
| `metal/bf16/scatter_add/batched/dense_axis1_b2_r16_c65_n8` | none | — | — |
| `metal/bf16/scatter_add/batched/dense_axis1_b2_r33_c129_n16` | none | — | — |
| `metal/bf16/scatter_add/batched/dense_axis1_b2_r33_c17_n16` | none | — | — |
| `metal/bf16/scatter_add/batched/dense_axis1_b2_r33_c5_n16` | none | — | — |
| `metal/bf16/scatter_add/batched/dense_axis1_b2_r33_c65_n16` | none | — | — |
| `metal/bf16/scatter_add/batched/dense_axis1_b2_r3_c129_n1` | none | — | — |
| `metal/bf16/scatter_add/batched/dense_axis1_b2_r3_c17_n1` | none | — | — |
| `metal/bf16/scatter_add/batched/dense_axis1_b2_r3_c5_n1` | none | — | — |
| `metal/bf16/scatter_add/batched/dense_axis1_b2_r3_c65_n1` | none | — | — |
| `metal/bf16/scatter_add/batched/dense_axis1_b2_r7_c129_n3` | none | — | — |
| `metal/bf16/scatter_add/batched/dense_axis1_b2_r7_c17_n3` | none | — | — |
| `metal/bf16/scatter_add/batched/dense_axis1_b2_r7_c5_n3` | none | — | — |
| `metal/bf16/scatter_add/batched/dense_axis1_b2_r7_c65_n3` | none | — | — |
| `metal/bf16/scatter_add/batched/dense_axis1_b4_r16_c129_n8` | none | — | — |
| `metal/bf16/scatter_add/batched/dense_axis1_b4_r16_c17_n8` | none | — | — |
| `metal/bf16/scatter_add/batched/dense_axis1_b4_r16_c5_n8` | none | — | — |
| `metal/bf16/scatter_add/batched/dense_axis1_b4_r16_c65_n8` | none | — | — |
| `metal/bf16/scatter_add/batched/dense_axis1_b4_r33_c129_n16` | none | — | — |
| `metal/bf16/scatter_add/batched/dense_axis1_b4_r33_c17_n16` | none | — | — |
| `metal/bf16/scatter_add/batched/dense_axis1_b4_r33_c5_n16` | none | — | — |
| `metal/bf16/scatter_add/batched/dense_axis1_b4_r33_c65_n16` | none | — | — |
| `metal/bf16/scatter_add/batched/dense_axis1_b4_r3_c129_n1` | none | — | — |
| `metal/bf16/scatter_add/batched/dense_axis1_b4_r3_c17_n1` | none | — | — |
| `metal/bf16/scatter_add/batched/dense_axis1_b4_r3_c5_n1` | none | — | — |
| `metal/bf16/scatter_add/batched/dense_axis1_b4_r3_c65_n1` | none | — | — |
| `metal/bf16/scatter_add/batched/dense_axis1_b4_r7_c129_n3` | none | — | — |
| `metal/bf16/scatter_add/batched/dense_axis1_b4_r7_c17_n3` | none | — | — |
| `metal/bf16/scatter_add/batched/dense_axis1_b4_r7_c5_n3` | none | — | — |
| `metal/bf16/scatter_add/batched/dense_axis1_b4_r7_c65_n3` | none | — | — |
| `metal/bf16/scatter_add/batched/dense_axis2_b2_r16_c129_n64` | none | — | — |
| `metal/bf16/scatter_add/batched/dense_axis2_b2_r16_c17_n8` | none | — | — |
| `metal/bf16/scatter_add/batched/dense_axis2_b2_r16_c5_n2` | none | — | — |
| `metal/bf16/scatter_add/batched/dense_axis2_b2_r16_c65_n32` | none | — | — |
| `metal/bf16/scatter_add/batched/dense_axis2_b2_r33_c129_n64` | none | — | — |
| `metal/bf16/scatter_add/batched/dense_axis2_b2_r33_c17_n8` | none | — | — |
| `metal/bf16/scatter_add/batched/dense_axis2_b2_r33_c5_n2` | none | — | — |
| `metal/bf16/scatter_add/batched/dense_axis2_b2_r33_c65_n32` | none | — | — |
| `metal/bf16/scatter_add/batched/dense_axis2_b2_r3_c129_n64` | none | — | — |
| `metal/bf16/scatter_add/batched/dense_axis2_b2_r3_c17_n8` | none | — | — |
| `metal/bf16/scatter_add/batched/dense_axis2_b2_r3_c5_n2` | none | — | — |
| `metal/bf16/scatter_add/batched/dense_axis2_b2_r3_c65_n32` | none | — | — |
| `metal/bf16/scatter_add/batched/dense_axis2_b2_r7_c129_n64` | none | — | — |
| `metal/bf16/scatter_add/batched/dense_axis2_b2_r7_c17_n8` | none | — | — |
| `metal/bf16/scatter_add/batched/dense_axis2_b2_r7_c5_n2` | none | — | — |
| `metal/bf16/scatter_add/batched/dense_axis2_b2_r7_c65_n32` | none | — | — |
| `metal/bf16/scatter_add/batched/dense_axis2_b4_r16_c129_n64` | none | — | — |
| `metal/bf16/scatter_add/batched/dense_axis2_b4_r16_c17_n8` | none | — | — |
| `metal/bf16/scatter_add/batched/dense_axis2_b4_r16_c5_n2` | none | — | — |
| `metal/bf16/scatter_add/batched/dense_axis2_b4_r16_c65_n32` | none | — | — |
| `metal/bf16/scatter_add/batched/dense_axis2_b4_r33_c129_n64` | none | — | — |
| `metal/bf16/scatter_add/batched/dense_axis2_b4_r33_c17_n8` | none | — | — |
| `metal/bf16/scatter_add/batched/dense_axis2_b4_r33_c5_n2` | none | — | — |
| `metal/bf16/scatter_add/batched/dense_axis2_b4_r33_c65_n32` | none | — | — |
| `metal/bf16/scatter_add/batched/dense_axis2_b4_r3_c129_n64` | none | — | — |
| `metal/bf16/scatter_add/batched/dense_axis2_b4_r3_c17_n8` | none | — | — |
| `metal/bf16/scatter_add/batched/dense_axis2_b4_r3_c5_n2` | none | — | — |
| `metal/bf16/scatter_add/batched/dense_axis2_b4_r3_c65_n32` | none | — | — |
| `metal/bf16/scatter_add/batched/dense_axis2_b4_r7_c129_n64` | none | — | — |
| `metal/bf16/scatter_add/batched/dense_axis2_b4_r7_c17_n8` | none | — | — |
| `metal/bf16/scatter_add/batched/dense_axis2_b4_r7_c5_n2` | none | — | — |
| `metal/bf16/scatter_add/batched/dense_axis2_b4_r7_c65_n32` | none | — | — |

## Removed cells

- `metal/f32/scatter_add/batched`
- `metal/f32/scatter_add/batched/dense_axis1_b2_r16_c129_n8`
- `metal/f32/scatter_add/batched/dense_axis1_b2_r16_c17_n8`
- `metal/f32/scatter_add/batched/dense_axis1_b2_r16_c5_n8`
- `metal/f32/scatter_add/batched/dense_axis1_b2_r16_c65_n8`
- `metal/f32/scatter_add/batched/dense_axis1_b2_r33_c129_n16`
- `metal/f32/scatter_add/batched/dense_axis1_b2_r33_c17_n16`
- `metal/f32/scatter_add/batched/dense_axis1_b2_r33_c5_n16`
- `metal/f32/scatter_add/batched/dense_axis1_b2_r33_c65_n16`
- `metal/f32/scatter_add/batched/dense_axis1_b2_r3_c129_n1`
- `metal/f32/scatter_add/batched/dense_axis1_b2_r3_c17_n1`
- `metal/f32/scatter_add/batched/dense_axis1_b2_r3_c5_n1`
- `metal/f32/scatter_add/batched/dense_axis1_b2_r3_c65_n1`
- `metal/f32/scatter_add/batched/dense_axis1_b2_r7_c129_n3`
- `metal/f32/scatter_add/batched/dense_axis1_b2_r7_c17_n3`
- `metal/f32/scatter_add/batched/dense_axis1_b2_r7_c5_n3`
- `metal/f32/scatter_add/batched/dense_axis1_b2_r7_c65_n3`
- `metal/f32/scatter_add/batched/dense_axis1_b4_r16_c129_n8`
- `metal/f32/scatter_add/batched/dense_axis1_b4_r16_c17_n8`
- `metal/f32/scatter_add/batched/dense_axis1_b4_r16_c5_n8`
- `metal/f32/scatter_add/batched/dense_axis1_b4_r16_c65_n8`
- `metal/f32/scatter_add/batched/dense_axis1_b4_r33_c129_n16`
- `metal/f32/scatter_add/batched/dense_axis1_b4_r33_c17_n16`
- `metal/f32/scatter_add/batched/dense_axis1_b4_r33_c5_n16`
- `metal/f32/scatter_add/batched/dense_axis1_b4_r33_c65_n16`
- `metal/f32/scatter_add/batched/dense_axis1_b4_r3_c129_n1`
- `metal/f32/scatter_add/batched/dense_axis1_b4_r3_c17_n1`
- `metal/f32/scatter_add/batched/dense_axis1_b4_r3_c5_n1`
- `metal/f32/scatter_add/batched/dense_axis1_b4_r3_c65_n1`
- `metal/f32/scatter_add/batched/dense_axis1_b4_r7_c129_n3`
- `metal/f32/scatter_add/batched/dense_axis1_b4_r7_c17_n3`
- `metal/f32/scatter_add/batched/dense_axis1_b4_r7_c5_n3`
- `metal/f32/scatter_add/batched/dense_axis1_b4_r7_c65_n3`
- `metal/f32/scatter_add/batched/dense_axis2_b2_r16_c129_n64`
- `metal/f32/scatter_add/batched/dense_axis2_b2_r16_c17_n8`
- `metal/f32/scatter_add/batched/dense_axis2_b2_r16_c5_n2`
- `metal/f32/scatter_add/batched/dense_axis2_b2_r16_c65_n32`
- `metal/f32/scatter_add/batched/dense_axis2_b2_r33_c129_n64`
- `metal/f32/scatter_add/batched/dense_axis2_b2_r33_c17_n8`
- `metal/f32/scatter_add/batched/dense_axis2_b2_r33_c5_n2`
- `metal/f32/scatter_add/batched/dense_axis2_b2_r33_c65_n32`
- `metal/f32/scatter_add/batched/dense_axis2_b2_r3_c129_n64`
- `metal/f32/scatter_add/batched/dense_axis2_b2_r3_c17_n8`
- `metal/f32/scatter_add/batched/dense_axis2_b2_r3_c5_n2`
- `metal/f32/scatter_add/batched/dense_axis2_b2_r3_c65_n32`
- `metal/f32/scatter_add/batched/dense_axis2_b2_r7_c129_n64`
- `metal/f32/scatter_add/batched/dense_axis2_b2_r7_c17_n8`
- `metal/f32/scatter_add/batched/dense_axis2_b2_r7_c5_n2`
- `metal/f32/scatter_add/batched/dense_axis2_b2_r7_c65_n32`
- `metal/f32/scatter_add/batched/dense_axis2_b4_r16_c129_n64`
- `metal/f32/scatter_add/batched/dense_axis2_b4_r16_c17_n8`
- `metal/f32/scatter_add/batched/dense_axis2_b4_r16_c5_n2`
- `metal/f32/scatter_add/batched/dense_axis2_b4_r16_c65_n32`
- `metal/f32/scatter_add/batched/dense_axis2_b4_r33_c129_n64`
- `metal/f32/scatter_add/batched/dense_axis2_b4_r33_c17_n8`
- `metal/f32/scatter_add/batched/dense_axis2_b4_r33_c5_n2`
- `metal/f32/scatter_add/batched/dense_axis2_b4_r33_c65_n32`
- `metal/f32/scatter_add/batched/dense_axis2_b4_r3_c129_n64`
- `metal/f32/scatter_add/batched/dense_axis2_b4_r3_c17_n8`
- `metal/f32/scatter_add/batched/dense_axis2_b4_r3_c5_n2`
- `metal/f32/scatter_add/batched/dense_axis2_b4_r3_c65_n32`
- `metal/f32/scatter_add/batched/dense_axis2_b4_r7_c129_n64`
- `metal/f32/scatter_add/batched/dense_axis2_b4_r7_c17_n8`
- `metal/f32/scatter_add/batched/dense_axis2_b4_r7_c5_n2`
- `metal/f32/scatter_add/batched/dense_axis2_b4_r7_c65_n32`
