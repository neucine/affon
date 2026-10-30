# Parity regression check

Target: `Mac16,1/26.6.2/arm64`

- Cells compared: 36
- Added cells: 32
- Removed cells: 220
- Changed case definitions: 0
- New correctness failures: 0
- Diagnostic timing suspects: 0
- Target matches: yes
- Measurement protocol matches: no

Timing changes are diagnostic and do not fail this check. Timing is compared only when target and protocol match. No timing baseline is accepted by this comparison.

| Cell | Correctness regression | Operator relative timing | Program relative timing |
| --- | --- | ---: | ---: |
| `metal/bf16/layer_norm/outer_axis/outer_offset_r129_c127` | none | — | — |
| `metal/bf16/layer_norm/outer_axis/outer_offset_r129_c17` | none | — | — |
| `metal/bf16/layer_norm/outer_axis/outer_offset_r129_c257` | none | — | — |
| `metal/bf16/layer_norm/outer_axis/outer_offset_r129_c63` | none | — | — |
| `metal/bf16/layer_norm/outer_axis/outer_offset_r64_c127` | none | — | — |
| `metal/bf16/layer_norm/outer_axis/outer_offset_r64_c17` | none | — | — |
| `metal/bf16/layer_norm/outer_axis/outer_offset_r64_c257` | none | — | — |
| `metal/bf16/layer_norm/outer_axis/outer_offset_r64_c63` | none | — | — |
| `metal/bf16/layer_norm/outer_axis/outer_offset_r7_c257` | none | — | — |
| `metal/bf16/layer_norm/outer_axis/outer_strided_r129_c127` | none | — | — |
| `metal/bf16/layer_norm/outer_axis/outer_strided_r129_c17` | none | — | — |
| `metal/bf16/layer_norm/outer_axis/outer_strided_r129_c257` | none | — | — |
| `metal/bf16/layer_norm/outer_axis/outer_strided_r129_c63` | none | — | — |
| `metal/bf16/layer_norm/outer_axis/outer_strided_r64_c127` | none | — | — |
| `metal/bf16/layer_norm/outer_axis/outer_strided_r64_c17` | none | — | — |
| `metal/bf16/layer_norm/outer_axis/outer_strided_r64_c257` | none | — | — |
| `metal/bf16/layer_norm/outer_axis/outer_strided_r64_c63` | none | — | — |
| `metal/bf16/layer_norm/outer_axis/outer_strided_r7_c257` | none | — | — |
| `metal/bf16/rms_norm/outer_axis/offset_r129_c127` | none | — | — |
| `metal/bf16/rms_norm/outer_axis/offset_r129_c17` | none | — | — |
| `metal/bf16/rms_norm/outer_axis/offset_r129_c257` | none | — | — |
| `metal/bf16/rms_norm/outer_axis/offset_r129_c63` | none | — | — |
| `metal/bf16/rms_norm/outer_axis/offset_r64_c127` | none | — | — |
| `metal/bf16/rms_norm/outer_axis/offset_r64_c17` | none | — | — |
| `metal/bf16/rms_norm/outer_axis/offset_r64_c257` | none | — | — |
| `metal/bf16/rms_norm/outer_axis/offset_r64_c63` | none | — | — |
| `metal/bf16/rms_norm/outer_axis/offset_r7_c257` | none | — | — |
| `metal/bf16/rms_norm/outer_axis/strided_r129_c127` | none | — | — |
| `metal/bf16/rms_norm/outer_axis/strided_r129_c17` | none | — | — |
| `metal/bf16/rms_norm/outer_axis/strided_r129_c257` | none | — | — |
| `metal/bf16/rms_norm/outer_axis/strided_r129_c63` | none | — | — |
| `metal/bf16/rms_norm/outer_axis/strided_r64_c127` | none | — | — |
| `metal/bf16/rms_norm/outer_axis/strided_r64_c17` | none | — | — |
| `metal/bf16/rms_norm/outer_axis/strided_r64_c257` | none | — | — |
| `metal/bf16/rms_norm/outer_axis/strided_r64_c63` | none | — | — |
| `metal/bf16/rms_norm/outer_axis/strided_r7_c257` | none | — | — |

## Added cells

- `metal/bf16/layer_norm/outer_axis/outer_offset_r128_c129`
- `metal/bf16/layer_norm/outer_axis/outer_offset_r128_c32`
- `metal/bf16/layer_norm/outer_axis/outer_offset_r128_c65`
- `metal/bf16/layer_norm/outer_axis/outer_offset_r16_c129`
- `metal/bf16/layer_norm/outer_axis/outer_offset_r16_c65`
- `metal/bf16/layer_norm/outer_axis/outer_offset_r33_c129`
- `metal/bf16/layer_norm/outer_axis/outer_offset_r33_c32`
- `metal/bf16/layer_norm/outer_axis/outer_offset_r33_c65`
- `metal/bf16/layer_norm/outer_axis/outer_strided_r128_c129`
- `metal/bf16/layer_norm/outer_axis/outer_strided_r128_c32`
- `metal/bf16/layer_norm/outer_axis/outer_strided_r128_c65`
- `metal/bf16/layer_norm/outer_axis/outer_strided_r16_c129`
- `metal/bf16/layer_norm/outer_axis/outer_strided_r16_c65`
- `metal/bf16/layer_norm/outer_axis/outer_strided_r33_c129`
- `metal/bf16/layer_norm/outer_axis/outer_strided_r33_c32`
- `metal/bf16/layer_norm/outer_axis/outer_strided_r33_c65`
- `metal/bf16/rms_norm/outer_axis/offset_r128_c129`
- `metal/bf16/rms_norm/outer_axis/offset_r128_c32`
- `metal/bf16/rms_norm/outer_axis/offset_r128_c65`
- `metal/bf16/rms_norm/outer_axis/offset_r16_c129`
- `metal/bf16/rms_norm/outer_axis/offset_r16_c65`
- `metal/bf16/rms_norm/outer_axis/offset_r33_c129`
- `metal/bf16/rms_norm/outer_axis/offset_r33_c32`
- `metal/bf16/rms_norm/outer_axis/offset_r33_c65`
- `metal/bf16/rms_norm/outer_axis/strided_r128_c129`
- `metal/bf16/rms_norm/outer_axis/strided_r128_c32`
- `metal/bf16/rms_norm/outer_axis/strided_r128_c65`
- `metal/bf16/rms_norm/outer_axis/strided_r16_c129`
- `metal/bf16/rms_norm/outer_axis/strided_r16_c65`
- `metal/bf16/rms_norm/outer_axis/strided_r33_c129`
- `metal/bf16/rms_norm/outer_axis/strided_r33_c32`
- `metal/bf16/rms_norm/outer_axis/strided_r33_c65`

## Removed cells

- `metal/bf16/layer_norm/outer_axis/outer_offset_r2_c127`
- `metal/bf16/layer_norm/outer_axis/outer_offset_r2_c17`
- `metal/bf16/layer_norm/outer_axis/outer_offset_r2_c257`
- `metal/bf16/layer_norm/outer_axis/outer_offset_r2_c63`
- `metal/bf16/layer_norm/outer_axis/outer_offset_r7_c127`
- `metal/bf16/layer_norm/outer_axis/outer_offset_r7_c17`
- `metal/bf16/layer_norm/outer_axis/outer_offset_r7_c63`
- `metal/bf16/layer_norm/outer_axis/outer_strided_r2_c127`
- `metal/bf16/layer_norm/outer_axis/outer_strided_r2_c17`
- `metal/bf16/layer_norm/outer_axis/outer_strided_r2_c257`
- `metal/bf16/layer_norm/outer_axis/outer_strided_r2_c63`
- `metal/bf16/layer_norm/outer_axis/outer_strided_r7_c127`
- `metal/bf16/layer_norm/outer_axis/outer_strided_r7_c17`
- `metal/bf16/layer_norm/outer_axis/outer_strided_r7_c63`
- `metal/bf16/layer_norm/strided/inner_offset_r129_c127`
- `metal/bf16/layer_norm/strided/inner_offset_r129_c17`
- `metal/bf16/layer_norm/strided/inner_offset_r129_c257`
- `metal/bf16/layer_norm/strided/inner_offset_r129_c63`
- `metal/bf16/layer_norm/strided/inner_offset_r2_c127`
- `metal/bf16/layer_norm/strided/inner_offset_r2_c17`
- `metal/bf16/layer_norm/strided/inner_offset_r2_c257`
- `metal/bf16/layer_norm/strided/inner_offset_r2_c63`
- `metal/bf16/layer_norm/strided/inner_offset_r64_c127`
- `metal/bf16/layer_norm/strided/inner_offset_r64_c17`
- `metal/bf16/layer_norm/strided/inner_offset_r64_c257`
- `metal/bf16/layer_norm/strided/inner_offset_r64_c63`
- `metal/bf16/layer_norm/strided/inner_offset_r7_c127`
- `metal/bf16/layer_norm/strided/inner_offset_r7_c17`
- `metal/bf16/layer_norm/strided/inner_offset_r7_c257`
- `metal/bf16/layer_norm/strided/inner_offset_r7_c63`
- `metal/bf16/layer_norm/strided/inner_strided_r129_c127`
- `metal/bf16/layer_norm/strided/inner_strided_r129_c17`
- `metal/bf16/layer_norm/strided/inner_strided_r129_c257`
- `metal/bf16/layer_norm/strided/inner_strided_r129_c63`
- `metal/bf16/layer_norm/strided/inner_strided_r2_c127`
- `metal/bf16/layer_norm/strided/inner_strided_r2_c17`
- `metal/bf16/layer_norm/strided/inner_strided_r2_c257`
- `metal/bf16/layer_norm/strided/inner_strided_r2_c63`
- `metal/bf16/layer_norm/strided/inner_strided_r64_c127`
- `metal/bf16/layer_norm/strided/inner_strided_r64_c17`
- `metal/bf16/layer_norm/strided/inner_strided_r64_c257`
- `metal/bf16/layer_norm/strided/inner_strided_r64_c63`
- `metal/bf16/layer_norm/strided/inner_strided_r7_c127`
- `metal/bf16/layer_norm/strided/inner_strided_r7_c17`
- `metal/bf16/layer_norm/strided/inner_strided_r7_c257`
- `metal/bf16/layer_norm/strided/inner_strided_r7_c63`
- `metal/bf16/rms_norm/long_row/offset_r129_c127`
- `metal/bf16/rms_norm/long_row/offset_r129_c17`
- `metal/bf16/rms_norm/long_row/offset_r129_c257`
- `metal/bf16/rms_norm/long_row/offset_r129_c63`
- `metal/bf16/rms_norm/long_row/offset_r2_c127`
- `metal/bf16/rms_norm/long_row/offset_r2_c17`
- `metal/bf16/rms_norm/long_row/offset_r2_c257`
- `metal/bf16/rms_norm/long_row/offset_r2_c63`
- `metal/bf16/rms_norm/long_row/offset_r64_c127`
- `metal/bf16/rms_norm/long_row/offset_r64_c17`
- `metal/bf16/rms_norm/long_row/offset_r64_c257`
- `metal/bf16/rms_norm/long_row/offset_r64_c63`
- `metal/bf16/rms_norm/long_row/offset_r7_c127`
- `metal/bf16/rms_norm/long_row/offset_r7_c17`
- `metal/bf16/rms_norm/long_row/offset_r7_c257`
- `metal/bf16/rms_norm/long_row/offset_r7_c63`
- `metal/bf16/rms_norm/outer_axis/offset_r2_c127`
- `metal/bf16/rms_norm/outer_axis/offset_r2_c17`
- `metal/bf16/rms_norm/outer_axis/offset_r2_c257`
- `metal/bf16/rms_norm/outer_axis/offset_r2_c63`
- `metal/bf16/rms_norm/outer_axis/offset_r7_c127`
- `metal/bf16/rms_norm/outer_axis/offset_r7_c17`
- `metal/bf16/rms_norm/outer_axis/offset_r7_c63`
- `metal/bf16/rms_norm/outer_axis/strided_r2_c127`
- `metal/bf16/rms_norm/outer_axis/strided_r2_c17`
- `metal/bf16/rms_norm/outer_axis/strided_r2_c257`
- `metal/bf16/rms_norm/outer_axis/strided_r2_c63`
- `metal/bf16/rms_norm/outer_axis/strided_r7_c127`
- `metal/bf16/rms_norm/outer_axis/strided_r7_c17`
- `metal/bf16/rms_norm/outer_axis/strided_r7_c63`
- `metal/bf16/rms_norm/strided/strided_r129_c127`
- `metal/bf16/rms_norm/strided/strided_r129_c17`
- `metal/bf16/rms_norm/strided/strided_r129_c257`
- `metal/bf16/rms_norm/strided/strided_r129_c63`
- `metal/bf16/rms_norm/strided/strided_r2_c127`
- `metal/bf16/rms_norm/strided/strided_r2_c17`
- `metal/bf16/rms_norm/strided/strided_r2_c257`
- `metal/bf16/rms_norm/strided/strided_r2_c63`
- `metal/bf16/rms_norm/strided/strided_r64_c127`
- `metal/bf16/rms_norm/strided/strided_r64_c17`
- `metal/bf16/rms_norm/strided/strided_r64_c257`
- `metal/bf16/rms_norm/strided/strided_r64_c63`
- `metal/bf16/rms_norm/strided/strided_r7_c127`
- `metal/bf16/rms_norm/strided/strided_r7_c17`
- `metal/bf16/rms_norm/strided/strided_r7_c257`
- `metal/bf16/rms_norm/strided/strided_r7_c63`
- `metal/f32/layer_norm/outer_axis/outer_offset_r129_c127`
- `metal/f32/layer_norm/outer_axis/outer_offset_r129_c17`
- `metal/f32/layer_norm/outer_axis/outer_offset_r129_c257`
- `metal/f32/layer_norm/outer_axis/outer_offset_r129_c63`
- `metal/f32/layer_norm/outer_axis/outer_offset_r2_c127`
- `metal/f32/layer_norm/outer_axis/outer_offset_r2_c17`
- `metal/f32/layer_norm/outer_axis/outer_offset_r2_c257`
- `metal/f32/layer_norm/outer_axis/outer_offset_r2_c63`
- `metal/f32/layer_norm/outer_axis/outer_offset_r64_c127`
- `metal/f32/layer_norm/outer_axis/outer_offset_r64_c17`
- `metal/f32/layer_norm/outer_axis/outer_offset_r64_c257`
- `metal/f32/layer_norm/outer_axis/outer_offset_r64_c63`
- `metal/f32/layer_norm/outer_axis/outer_offset_r7_c127`
- `metal/f32/layer_norm/outer_axis/outer_offset_r7_c17`
- `metal/f32/layer_norm/outer_axis/outer_offset_r7_c257`
- `metal/f32/layer_norm/outer_axis/outer_offset_r7_c63`
- `metal/f32/layer_norm/outer_axis/outer_strided_r129_c127`
- `metal/f32/layer_norm/outer_axis/outer_strided_r129_c17`
- `metal/f32/layer_norm/outer_axis/outer_strided_r129_c257`
- `metal/f32/layer_norm/outer_axis/outer_strided_r129_c63`
- `metal/f32/layer_norm/outer_axis/outer_strided_r2_c127`
- `metal/f32/layer_norm/outer_axis/outer_strided_r2_c17`
- `metal/f32/layer_norm/outer_axis/outer_strided_r2_c257`
- `metal/f32/layer_norm/outer_axis/outer_strided_r2_c63`
- `metal/f32/layer_norm/outer_axis/outer_strided_r64_c127`
- `metal/f32/layer_norm/outer_axis/outer_strided_r64_c17`
- `metal/f32/layer_norm/outer_axis/outer_strided_r64_c257`
- `metal/f32/layer_norm/outer_axis/outer_strided_r64_c63`
- `metal/f32/layer_norm/outer_axis/outer_strided_r7_c127`
- `metal/f32/layer_norm/outer_axis/outer_strided_r7_c17`
- `metal/f32/layer_norm/outer_axis/outer_strided_r7_c257`
- `metal/f32/layer_norm/outer_axis/outer_strided_r7_c63`
- `metal/f32/layer_norm/strided/inner_offset_r129_c127`
- `metal/f32/layer_norm/strided/inner_offset_r129_c17`
- `metal/f32/layer_norm/strided/inner_offset_r129_c257`
- `metal/f32/layer_norm/strided/inner_offset_r129_c63`
- `metal/f32/layer_norm/strided/inner_offset_r2_c127`
- `metal/f32/layer_norm/strided/inner_offset_r2_c17`
- `metal/f32/layer_norm/strided/inner_offset_r2_c257`
- `metal/f32/layer_norm/strided/inner_offset_r2_c63`
- `metal/f32/layer_norm/strided/inner_offset_r64_c127`
- `metal/f32/layer_norm/strided/inner_offset_r64_c17`
- `metal/f32/layer_norm/strided/inner_offset_r64_c257`
- `metal/f32/layer_norm/strided/inner_offset_r64_c63`
- `metal/f32/layer_norm/strided/inner_offset_r7_c127`
- `metal/f32/layer_norm/strided/inner_offset_r7_c17`
- `metal/f32/layer_norm/strided/inner_offset_r7_c257`
- `metal/f32/layer_norm/strided/inner_offset_r7_c63`
- `metal/f32/layer_norm/strided/inner_strided_r129_c127`
- `metal/f32/layer_norm/strided/inner_strided_r129_c17`
- `metal/f32/layer_norm/strided/inner_strided_r129_c257`
- `metal/f32/layer_norm/strided/inner_strided_r129_c63`
- `metal/f32/layer_norm/strided/inner_strided_r2_c127`
- `metal/f32/layer_norm/strided/inner_strided_r2_c17`
- `metal/f32/layer_norm/strided/inner_strided_r2_c257`
- `metal/f32/layer_norm/strided/inner_strided_r2_c63`
- `metal/f32/layer_norm/strided/inner_strided_r64_c127`
- `metal/f32/layer_norm/strided/inner_strided_r64_c17`
- `metal/f32/layer_norm/strided/inner_strided_r64_c257`
- `metal/f32/layer_norm/strided/inner_strided_r64_c63`
- `metal/f32/layer_norm/strided/inner_strided_r7_c127`
- `metal/f32/layer_norm/strided/inner_strided_r7_c17`
- `metal/f32/layer_norm/strided/inner_strided_r7_c257`
- `metal/f32/layer_norm/strided/inner_strided_r7_c63`
- `metal/f32/rms_norm/long_row/offset_r129_c127`
- `metal/f32/rms_norm/long_row/offset_r129_c17`
- `metal/f32/rms_norm/long_row/offset_r129_c257`
- `metal/f32/rms_norm/long_row/offset_r129_c63`
- `metal/f32/rms_norm/long_row/offset_r2_c127`
- `metal/f32/rms_norm/long_row/offset_r2_c17`
- `metal/f32/rms_norm/long_row/offset_r2_c257`
- `metal/f32/rms_norm/long_row/offset_r2_c63`
- `metal/f32/rms_norm/long_row/offset_r64_c127`
- `metal/f32/rms_norm/long_row/offset_r64_c17`
- `metal/f32/rms_norm/long_row/offset_r64_c257`
- `metal/f32/rms_norm/long_row/offset_r64_c63`
- `metal/f32/rms_norm/long_row/offset_r7_c127`
- `metal/f32/rms_norm/long_row/offset_r7_c17`
- `metal/f32/rms_norm/long_row/offset_r7_c257`
- `metal/f32/rms_norm/long_row/offset_r7_c63`
- `metal/f32/rms_norm/outer_axis/offset_r129_c127`
- `metal/f32/rms_norm/outer_axis/offset_r129_c17`
- `metal/f32/rms_norm/outer_axis/offset_r129_c257`
- `metal/f32/rms_norm/outer_axis/offset_r129_c63`
- `metal/f32/rms_norm/outer_axis/offset_r2_c127`
- `metal/f32/rms_norm/outer_axis/offset_r2_c17`
- `metal/f32/rms_norm/outer_axis/offset_r2_c257`
- `metal/f32/rms_norm/outer_axis/offset_r2_c63`
- `metal/f32/rms_norm/outer_axis/offset_r64_c127`
- `metal/f32/rms_norm/outer_axis/offset_r64_c17`
- `metal/f32/rms_norm/outer_axis/offset_r64_c257`
- `metal/f32/rms_norm/outer_axis/offset_r64_c63`
- `metal/f32/rms_norm/outer_axis/offset_r7_c127`
- `metal/f32/rms_norm/outer_axis/offset_r7_c17`
- `metal/f32/rms_norm/outer_axis/offset_r7_c257`
- `metal/f32/rms_norm/outer_axis/offset_r7_c63`
- `metal/f32/rms_norm/outer_axis/strided_r129_c127`
- `metal/f32/rms_norm/outer_axis/strided_r129_c17`
- `metal/f32/rms_norm/outer_axis/strided_r129_c257`
- `metal/f32/rms_norm/outer_axis/strided_r129_c63`
- `metal/f32/rms_norm/outer_axis/strided_r2_c127`
- `metal/f32/rms_norm/outer_axis/strided_r2_c17`
- `metal/f32/rms_norm/outer_axis/strided_r2_c257`
- `metal/f32/rms_norm/outer_axis/strided_r2_c63`
- `metal/f32/rms_norm/outer_axis/strided_r64_c127`
- `metal/f32/rms_norm/outer_axis/strided_r64_c17`
- `metal/f32/rms_norm/outer_axis/strided_r64_c257`
- `metal/f32/rms_norm/outer_axis/strided_r64_c63`
- `metal/f32/rms_norm/outer_axis/strided_r7_c127`
- `metal/f32/rms_norm/outer_axis/strided_r7_c17`
- `metal/f32/rms_norm/outer_axis/strided_r7_c257`
- `metal/f32/rms_norm/outer_axis/strided_r7_c63`
- `metal/f32/rms_norm/strided/strided_r129_c127`
- `metal/f32/rms_norm/strided/strided_r129_c17`
- `metal/f32/rms_norm/strided/strided_r129_c257`
- `metal/f32/rms_norm/strided/strided_r129_c63`
- `metal/f32/rms_norm/strided/strided_r2_c127`
- `metal/f32/rms_norm/strided/strided_r2_c17`
- `metal/f32/rms_norm/strided/strided_r2_c257`
- `metal/f32/rms_norm/strided/strided_r2_c63`
- `metal/f32/rms_norm/strided/strided_r64_c127`
- `metal/f32/rms_norm/strided/strided_r64_c17`
- `metal/f32/rms_norm/strided/strided_r64_c257`
- `metal/f32/rms_norm/strided/strided_r64_c63`
- `metal/f32/rms_norm/strided/strided_r7_c127`
- `metal/f32/rms_norm/strided/strided_r7_c17`
- `metal/f32/rms_norm/strided/strided_r7_c257`
- `metal/f32/rms_norm/strided/strided_r7_c63`
