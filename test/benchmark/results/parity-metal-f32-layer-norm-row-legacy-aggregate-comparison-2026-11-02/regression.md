# f32 layer norm legacy aggregate comparison

Old aggregate reports have no retained parity manifest. Raw reports match cell ID, kind, target, peer engine, shape, dtype, layout, input hash, iterations, and rounds. Historical PyTorch versions differ.

- 98 cells; 196 tier reports; no correctness regressions or contract mismatches.
- Initial reports: 16 timing suspects above 1.25× execution ratio.
- 11 cells retained old reports after repeat timing review.

| Cell | Tier | Protocol | Initial execution ratio | Repeat ratio |
| --- | --- | --- | ---: | ---: |
| `metal/f32/layer_norm/strided/inner_strided_r7_c63` | operator_peer | i5-s3-r2 | 0.83× | 1.17× |
| `metal/f32/layer_norm/strided/inner_offset_r64_c257` | operator_peer | i5-s3-r2 | 1.00× | 0.90× |
| `metal/f32/layer_norm/outer_axis/outer_strided_r2_c257` | operator_peer | i5-s3-r2 | 0.77× | 0.89× |
| `metal/f32/layer_norm/outer_axis/outer_strided_r64_c63` | operator_peer | i5-s3-r2 | 1.24× | 1.48× |
| `metal/f32/layer_norm/outer_axis/outer_strided_r2_c17` | operator_peer | i5-s3-r2 | 0.91× | 0.86× |
| `metal/f32/layer_norm/strided/inner_strided_r2_c257` | operator_peer | i5-s3-r2 | 0.75× | 1.04× |
| `metal/f32/layer_norm/outer_axis/outer_offset_r2_c17` | operator_peer | i5-s3-r2 | 0.80× | 0.91× |
| `metal/f32/layer_norm/outer_axis/outer_offset_r64_c63` | operator_peer | i5-s3-r2 | 1.02× | 0.71× |
| `metal/f32/layer_norm/strided/inner_offset_r2_c257` | operator_peer | i5-s3-r2 | 1.17× | 1.39× |
| `metal/f32/layer_norm/strided/inner_strided_r7_c257` | operator_peer | i5-s3-r2 | 1.26× | 1.19× |
| `metal/f32/layer_norm/strided/inner_strided_r2_c127` | operator_peer | i5-s3-r2 | 0.93× | 0.96× |
| `metal/f32/layer_norm/outer_axis/outer_offset_r64_c127` | operator_peer | i5-s3-r2 | 1.33× | 1.01× |
| `metal/f32/layer_norm/strided/inner_strided_r2_c63` | operator_peer | i5-s3-r2 | 1.16× | 1.03× |
| `metal/f32/layer_norm/strided/inner_strided_r7_c17` | operator_peer | i5-s3-r2 | 0.52× | 0.61× |
| `metal/f32/layer_norm/strided/inner_offset_r64_c127` | operator_peer | i5-s3-r2 | 0.99× | 0.92× |
| `metal/f32/layer_norm/strided/inner_strided_r7_c63` | program_peer | i5-s3-r2 | 1.33× | 1.54× |
| `metal/f32/layer_norm/strided/inner_offset_r64_c257` | program_peer | i5-s3-r2 | 1.52× | 1.14× |
| `metal/f32/layer_norm/outer_axis/outer_strided_r2_c257` | program_peer | i5-s3-r2 | 1.28× | 1.46× |
| `metal/f32/layer_norm/outer_axis/outer_strided_r64_c63` | program_peer | i5-s3-r2 | 1.63× | 1.56× |
| `metal/f32/layer_norm/outer_axis/outer_strided_r2_c17` | program_peer | i5-s3-r2 | 1.32× | 0.92× |
| `metal/f32/layer_norm/strided/inner_strided_r2_c257` | program_peer | i5-s3-r2 | 1.40× | 1.59× |
| `metal/f32/layer_norm/outer_axis/outer_offset_r2_c17` | program_peer | i5-s3-r2 | 1.50× | 1.34× |
| `metal/f32/layer_norm/outer_axis/outer_offset_r64_c63` | program_peer | i5-s3-r2 | 1.67× | 0.83× |
| `metal/f32/layer_norm/strided/inner_offset_r2_c257` | program_peer | i5-s3-r2 | 1.26× | 1.07× |
| `metal/f32/layer_norm/strided/inner_strided_r7_c257` | program_peer | i5-s3-r2 | 0.95× | 0.97× |
| `metal/f32/layer_norm/strided/inner_strided_r2_c127` | program_peer | i5-s3-r2 | 1.36× | 1.37× |
| `metal/f32/layer_norm/outer_axis/outer_offset_r64_c127` | program_peer | i5-s3-r2 | 1.00× | 1.51× |
| `metal/f32/layer_norm/strided/inner_strided_r2_c63` | program_peer | i5-s3-r2 | 1.51× | 1.44× |
| `metal/f32/layer_norm/strided/inner_strided_r7_c17` | program_peer | i5-s3-r2 | 1.28× | 1.06× |
| `metal/f32/layer_norm/strided/inner_offset_r64_c127` | program_peer | i5-s3-r2 | 1.33× | 1.35× |
| `metal/f32/layer_norm/long_row/r33_c255` | operator_peer | i10-s5-r3 | 1.41× | 1.29× |
| `metal/f32/layer_norm/long_row/r33_c255` | program_peer | i10-s5-r3 | 0.74× | 0.94× |
