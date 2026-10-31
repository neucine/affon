# Parity regression check

Target: `Mac16,1/26.6.2/arm64`

- Cells compared: 28
- Added cells: 0
- Removed cells: 228
- Changed case definitions: 0
- New correctness failures: 0
- Diagnostic timing suspects: 0
- Target matches: yes
- Measurement protocol matches: yes

Timing changes are diagnostic and do not fail this check. Timing is compared only when target and protocol match. No timing baseline is accepted by this comparison.

| Cell | Correctness regression | Operator relative timing | Program relative timing |
| --- | --- | ---: | ---: |
| `metal/bf16/layer_norm/outer_axis/outer_offset_r2_c257` | none | 0.38× | 0.39× |
| `metal/bf16/layer_norm/outer_axis/outer_offset_r7_c127` | none | 0.20× | 0.15× |
| `metal/bf16/layer_norm/outer_axis/outer_offset_r7_c63` | none | 0.27× | 0.29× |
| `metal/bf16/layer_norm/outer_axis/outer_strided_r2_c257` | none | 0.28× | 0.38× |
| `metal/bf16/layer_norm/outer_axis/outer_strided_r7_c127` | none | 0.22× | 0.28× |
| `metal/bf16/layer_norm/outer_axis/outer_strided_r7_c63` | none | 0.35× | 0.36× |
| `metal/bf16/layer_norm/strided/inner_offset_r129_c257` | none | 0.60× | 0.73× |
| `metal/bf16/layer_norm/strided/inner_offset_r2_c257` | none | 0.29× | 0.46× |
| `metal/bf16/layer_norm/strided/inner_offset_r64_c257` | none | 0.38× | 0.45× |
| `metal/bf16/layer_norm/strided/inner_offset_r7_c257` | none | 0.36× | 0.37× |
| `metal/bf16/layer_norm/strided/inner_strided_r129_c257` | none | 0.60× | 0.63× |
| `metal/bf16/layer_norm/strided/inner_strided_r2_c257` | none | 0.43× | 0.70× |
| `metal/bf16/layer_norm/strided/inner_strided_r64_c257` | none | 0.40× | 0.46× |
| `metal/bf16/layer_norm/strided/inner_strided_r7_c257` | none | 0.30× | 0.32× |
| `metal/bf16/rms_norm/long_row/offset_r129_c257` | none | 0.59× | 0.62× |
| `metal/bf16/rms_norm/long_row/offset_r2_c257` | none | 0.39× | 0.54× |
| `metal/bf16/rms_norm/long_row/offset_r64_c257` | none | 0.40× | 0.43× |
| `metal/bf16/rms_norm/long_row/offset_r7_c257` | none | 0.33× | 0.39× |
| `metal/bf16/rms_norm/outer_axis/offset_r2_c257` | none | 0.37× | 0.41× |
| `metal/bf16/rms_norm/outer_axis/offset_r7_c127` | none | 0.20× | 0.26× |
| `metal/bf16/rms_norm/outer_axis/offset_r7_c63` | none | 0.22× | 0.34× |
| `metal/bf16/rms_norm/outer_axis/strided_r2_c257` | none | 0.34× | 0.37× |
| `metal/bf16/rms_norm/outer_axis/strided_r7_c127` | none | 0.21× | 0.19× |
| `metal/bf16/rms_norm/outer_axis/strided_r7_c63` | none | 0.29× | 0.29× |
| `metal/bf16/rms_norm/strided/strided_r129_c257` | none | 0.58× | 0.56× |
| `metal/bf16/rms_norm/strided/strided_r2_c257` | none | 0.40× | 0.56× |
| `metal/bf16/rms_norm/strided/strided_r64_c257` | none | 0.40× | 0.46× |
| `metal/bf16/rms_norm/strided/strided_r7_c257` | none | 0.33× | 0.33× |

## Removed cells

- `metal/bf16/layer_norm/outer_axis/outer_offset_r129_c127`
- `metal/bf16/layer_norm/outer_axis/outer_offset_r129_c17`
- `metal/bf16/layer_norm/outer_axis/outer_offset_r129_c257`
- `metal/bf16/layer_norm/outer_axis/outer_offset_r129_c63`
- `metal/bf16/layer_norm/outer_axis/outer_offset_r2_c127`
- `metal/bf16/layer_norm/outer_axis/outer_offset_r2_c17`
- `metal/bf16/layer_norm/outer_axis/outer_offset_r2_c63`
- `metal/bf16/layer_norm/outer_axis/outer_offset_r64_c127`
- `metal/bf16/layer_norm/outer_axis/outer_offset_r64_c17`
- `metal/bf16/layer_norm/outer_axis/outer_offset_r64_c257`
- `metal/bf16/layer_norm/outer_axis/outer_offset_r64_c63`
- `metal/bf16/layer_norm/outer_axis/outer_offset_r7_c17`
- `metal/bf16/layer_norm/outer_axis/outer_offset_r7_c257`
- `metal/bf16/layer_norm/outer_axis/outer_strided_r129_c127`
- `metal/bf16/layer_norm/outer_axis/outer_strided_r129_c17`
- `metal/bf16/layer_norm/outer_axis/outer_strided_r129_c257`
- `metal/bf16/layer_norm/outer_axis/outer_strided_r129_c63`
- `metal/bf16/layer_norm/outer_axis/outer_strided_r2_c127`
- `metal/bf16/layer_norm/outer_axis/outer_strided_r2_c17`
- `metal/bf16/layer_norm/outer_axis/outer_strided_r2_c63`
- `metal/bf16/layer_norm/outer_axis/outer_strided_r64_c127`
- `metal/bf16/layer_norm/outer_axis/outer_strided_r64_c17`
- `metal/bf16/layer_norm/outer_axis/outer_strided_r64_c257`
- `metal/bf16/layer_norm/outer_axis/outer_strided_r64_c63`
- `metal/bf16/layer_norm/outer_axis/outer_strided_r7_c17`
- `metal/bf16/layer_norm/outer_axis/outer_strided_r7_c257`
- `metal/bf16/layer_norm/strided/inner_offset_r129_c127`
- `metal/bf16/layer_norm/strided/inner_offset_r129_c17`
- `metal/bf16/layer_norm/strided/inner_offset_r129_c63`
- `metal/bf16/layer_norm/strided/inner_offset_r2_c127`
- `metal/bf16/layer_norm/strided/inner_offset_r2_c17`
- `metal/bf16/layer_norm/strided/inner_offset_r2_c63`
- `metal/bf16/layer_norm/strided/inner_offset_r64_c127`
- `metal/bf16/layer_norm/strided/inner_offset_r64_c17`
- `metal/bf16/layer_norm/strided/inner_offset_r64_c63`
- `metal/bf16/layer_norm/strided/inner_offset_r7_c127`
- `metal/bf16/layer_norm/strided/inner_offset_r7_c17`
- `metal/bf16/layer_norm/strided/inner_offset_r7_c63`
- `metal/bf16/layer_norm/strided/inner_strided_r129_c127`
- `metal/bf16/layer_norm/strided/inner_strided_r129_c17`
- `metal/bf16/layer_norm/strided/inner_strided_r129_c63`
- `metal/bf16/layer_norm/strided/inner_strided_r2_c127`
- `metal/bf16/layer_norm/strided/inner_strided_r2_c17`
- `metal/bf16/layer_norm/strided/inner_strided_r2_c63`
- `metal/bf16/layer_norm/strided/inner_strided_r64_c127`
- `metal/bf16/layer_norm/strided/inner_strided_r64_c17`
- `metal/bf16/layer_norm/strided/inner_strided_r64_c63`
- `metal/bf16/layer_norm/strided/inner_strided_r7_c127`
- `metal/bf16/layer_norm/strided/inner_strided_r7_c17`
- `metal/bf16/layer_norm/strided/inner_strided_r7_c63`
- `metal/bf16/rms_norm/long_row/offset_r129_c127`
- `metal/bf16/rms_norm/long_row/offset_r129_c17`
- `metal/bf16/rms_norm/long_row/offset_r129_c63`
- `metal/bf16/rms_norm/long_row/offset_r2_c127`
- `metal/bf16/rms_norm/long_row/offset_r2_c17`
- `metal/bf16/rms_norm/long_row/offset_r2_c63`
- `metal/bf16/rms_norm/long_row/offset_r64_c127`
- `metal/bf16/rms_norm/long_row/offset_r64_c17`
- `metal/bf16/rms_norm/long_row/offset_r64_c63`
- `metal/bf16/rms_norm/long_row/offset_r7_c127`
- `metal/bf16/rms_norm/long_row/offset_r7_c17`
- `metal/bf16/rms_norm/long_row/offset_r7_c63`
- `metal/bf16/rms_norm/outer_axis/offset_r129_c127`
- `metal/bf16/rms_norm/outer_axis/offset_r129_c17`
- `metal/bf16/rms_norm/outer_axis/offset_r129_c257`
- `metal/bf16/rms_norm/outer_axis/offset_r129_c63`
- `metal/bf16/rms_norm/outer_axis/offset_r2_c127`
- `metal/bf16/rms_norm/outer_axis/offset_r2_c17`
- `metal/bf16/rms_norm/outer_axis/offset_r2_c63`
- `metal/bf16/rms_norm/outer_axis/offset_r64_c127`
- `metal/bf16/rms_norm/outer_axis/offset_r64_c17`
- `metal/bf16/rms_norm/outer_axis/offset_r64_c257`
- `metal/bf16/rms_norm/outer_axis/offset_r64_c63`
- `metal/bf16/rms_norm/outer_axis/offset_r7_c17`
- `metal/bf16/rms_norm/outer_axis/offset_r7_c257`
- `metal/bf16/rms_norm/outer_axis/strided_r129_c127`
- `metal/bf16/rms_norm/outer_axis/strided_r129_c17`
- `metal/bf16/rms_norm/outer_axis/strided_r129_c257`
- `metal/bf16/rms_norm/outer_axis/strided_r129_c63`
- `metal/bf16/rms_norm/outer_axis/strided_r2_c127`
- `metal/bf16/rms_norm/outer_axis/strided_r2_c17`
- `metal/bf16/rms_norm/outer_axis/strided_r2_c63`
- `metal/bf16/rms_norm/outer_axis/strided_r64_c127`
- `metal/bf16/rms_norm/outer_axis/strided_r64_c17`
- `metal/bf16/rms_norm/outer_axis/strided_r64_c257`
- `metal/bf16/rms_norm/outer_axis/strided_r64_c63`
- `metal/bf16/rms_norm/outer_axis/strided_r7_c17`
- `metal/bf16/rms_norm/outer_axis/strided_r7_c257`
- `metal/bf16/rms_norm/strided/strided_r129_c127`
- `metal/bf16/rms_norm/strided/strided_r129_c17`
- `metal/bf16/rms_norm/strided/strided_r129_c63`
- `metal/bf16/rms_norm/strided/strided_r2_c127`
- `metal/bf16/rms_norm/strided/strided_r2_c17`
- `metal/bf16/rms_norm/strided/strided_r2_c63`
- `metal/bf16/rms_norm/strided/strided_r64_c127`
- `metal/bf16/rms_norm/strided/strided_r64_c17`
- `metal/bf16/rms_norm/strided/strided_r64_c63`
- `metal/bf16/rms_norm/strided/strided_r7_c127`
- `metal/bf16/rms_norm/strided/strided_r7_c17`
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
