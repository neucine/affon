# Parity regression check

Target: `Mac16,1/26.6.2/arm64`

- Cells compared: 128
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
| `metal/f32/one_hot/batched` | none | 0.67× | 0.68× |
| `metal/f32/one_hot/batched/b1_r16_c17_k2` | none | 0.51× | 0.56× |
| `metal/f32/one_hot/batched/b1_r16_c17_k33` | none | 0.28× | 0.26× |
| `metal/f32/one_hot/batched/b1_r16_c1_k2` | none | 0.63× | 0.65× |
| `metal/f32/one_hot/batched/b1_r16_c1_k33` | none | 0.47× | 0.61× |
| `metal/f32/one_hot/batched/b1_r16_c3_k2` | none | 0.68× | 0.52× |
| `metal/f32/one_hot/batched/b1_r16_c3_k33` | none | 0.51× | 0.51× |
| `metal/f32/one_hot/batched/b1_r16_c8_k2` | none | 0.60× | 0.58× |
| `metal/f32/one_hot/batched/b1_r16_c8_k33` | none | 0.33× | 0.41× |
| `metal/f32/one_hot/batched/b1_r1_c17_k2` | none | 0.63× | 0.63× |
| `metal/f32/one_hot/batched/b1_r1_c17_k33` | none | 0.61× | 0.60× |
| `metal/f32/one_hot/batched/b1_r1_c1_k33` | none | 0.63× | 0.62× |
| `metal/f32/one_hot/batched/b1_r1_c3_k2` | none | 0.69× | 0.59× |
| `metal/f32/one_hot/batched/b1_r1_c3_k33` | none | 0.57× | 0.54× |
| `metal/f32/one_hot/batched/b1_r1_c8_k2` | none | 0.72× | 0.52× |
| `metal/f32/one_hot/batched/b1_r1_c8_k33` | none | 0.64× | 0.64× |
| `metal/f32/one_hot/batched/b1_r3_c17_k2` | none | 0.68× | 0.66× |
| `metal/f32/one_hot/batched/b1_r3_c17_k33` | none | 0.53× | 0.50× |
| `metal/f32/one_hot/batched/b1_r3_c1_k2` | none | 0.59× | 0.68× |
| `metal/f32/one_hot/batched/b1_r3_c1_k33` | none | 0.64× | 0.66× |
| `metal/f32/one_hot/batched/b1_r3_c3_k2` | none | 0.67× | 0.65× |
| `metal/f32/one_hot/batched/b1_r3_c3_k33` | none | 0.63× | 0.57× |
| `metal/f32/one_hot/batched/b1_r3_c8_k2` | none | 0.67× | 0.78× |
| `metal/f32/one_hot/batched/b1_r3_c8_k33` | none | 0.60× | 0.58× |
| `metal/f32/one_hot/batched/b1_r7_c17_k2` | none | 0.60× | 0.62× |
| `metal/f32/one_hot/batched/b1_r7_c17_k33` | none | 0.40× | 0.43× |
| `metal/f32/one_hot/batched/b1_r7_c1_k2` | none | 0.62× | 0.67× |
| `metal/f32/one_hot/batched/b1_r7_c1_k33` | none | 0.62× | 0.59× |
| `metal/f32/one_hot/batched/b1_r7_c3_k2` | none | 0.57× | 0.66× |
| `metal/f32/one_hot/batched/b1_r7_c3_k33` | none | 0.60× | 0.58× |
| `metal/f32/one_hot/batched/b1_r7_c8_k2` | none | 0.62× | 0.64× |
| `metal/f32/one_hot/batched/b1_r7_c8_k33` | none | 0.54× | 0.52× |
| `metal/f32/one_hot/batched/b2_r16_c17_k2` | none | 0.47× | 0.48× |
| `metal/f32/one_hot/batched/b2_r16_c17_k33` | none | 0.21× | 0.21× |
| `metal/f32/one_hot/batched/b2_r16_c1_k2` | none | 0.52× | 0.63× |
| `metal/f32/one_hot/batched/b2_r16_c1_k33` | none | 0.54× | 0.53× |
| `metal/f32/one_hot/batched/b2_r16_c3_k2` | none | 0.50× | 0.60× |
| `metal/f32/one_hot/batched/b2_r16_c3_k33` | none | 0.42× | 0.49× |
| `metal/f32/one_hot/batched/b2_r16_c8_k2` | none | 0.52× | 0.51× |
| `metal/f32/one_hot/batched/b2_r16_c8_k33` | none | 0.28× | 0.30× |
| `metal/f32/one_hot/batched/b2_r1_c17_k2` | none | 0.67× | 0.68× |
| `metal/f32/one_hot/batched/b2_r1_c17_k33` | none | 0.54× | 0.49× |
| `metal/f32/one_hot/batched/b2_r1_c1_k2` | none | 0.66× | 0.70× |
| `metal/f32/one_hot/batched/b2_r1_c1_k33` | none | 0.66× | 0.67× |
| `metal/f32/one_hot/batched/b2_r1_c3_k2` | none | 0.63× | 0.77× |
| `metal/f32/one_hot/batched/b2_r1_c3_k33` | none | 0.67× | 0.66× |
| `metal/f32/one_hot/batched/b2_r1_c8_k2` | none | 0.68× | 0.71× |
| `metal/f32/one_hot/batched/b2_r1_c8_k33` | none | 0.59× | 0.61× |
| `metal/f32/one_hot/batched/b2_r3_c17_k2` | none | 0.59× | 0.59× |
| `metal/f32/one_hot/batched/b2_r3_c17_k33` | none | 0.46× | 0.48× |
| `metal/f32/one_hot/batched/b2_r3_c1_k2` | none | 0.55× | 0.69× |
| `metal/f32/one_hot/batched/b2_r3_c1_k33` | none | 0.62× | 0.57× |
| `metal/f32/one_hot/batched/b2_r3_c3_k2` | none | 0.68× | 0.58× |
| `metal/f32/one_hot/batched/b2_r3_c3_k33` | none | 0.75× | 0.54× |
| `metal/f32/one_hot/batched/b2_r3_c8_k2` | none | 0.64× | 0.67× |
| `metal/f32/one_hot/batched/b2_r3_c8_k33` | none | 0.57× | 0.55× |
| `metal/f32/one_hot/batched/b2_r7_c17_k2` | none | 0.58× | 0.54× |
| `metal/f32/one_hot/batched/b2_r7_c17_k33` | none | 0.28× | 0.30× |
| `metal/f32/one_hot/batched/b2_r7_c1_k2` | none | 0.57× | 0.78× |
| `metal/f32/one_hot/batched/b2_r7_c1_k33` | none | 0.59× | 0.62× |
| `metal/f32/one_hot/batched/b2_r7_c3_k2` | none | 0.78× | 0.59× |
| `metal/f32/one_hot/batched/b2_r7_c3_k33` | none | 0.44× | 0.52× |
| `metal/f32/one_hot/batched/b2_r7_c8_k2` | none | 0.59× | 0.60× |
| `metal/f32/one_hot/batched/b2_r7_c8_k33` | none | 0.45× | 0.49× |
| `metal/f32/one_hot/batched/b4_r16_c17_k2` | none | 0.42× | 0.34× |
| `metal/f32/one_hot/batched/b4_r16_c17_k33` | none | 0.19× | 0.19× |
| `metal/f32/one_hot/batched/b4_r16_c1_k2` | none | 0.60× | 0.57× |
| `metal/f32/one_hot/batched/b4_r16_c1_k33` | none | 0.46× | 0.51× |
| `metal/f32/one_hot/batched/b4_r16_c3_k2` | none | 0.55× | 0.54× |
| `metal/f32/one_hot/batched/b4_r16_c3_k33` | none | 0.33× | 0.32× |
| `metal/f32/one_hot/batched/b4_r16_c8_k2` | none | 0.44× | 0.44× |
| `metal/f32/one_hot/batched/b4_r16_c8_k33` | none | 0.22× | 0.25× |
| `metal/f32/one_hot/batched/b4_r1_c17_k2` | none | 0.66× | 0.63× |
| `metal/f32/one_hot/batched/b4_r1_c17_k33` | none | 0.46× | 0.43× |
| `metal/f32/one_hot/batched/b4_r1_c1_k2` | none | 0.65× | 0.68× |
| `metal/f32/one_hot/batched/b4_r1_c1_k33` | none | 0.63× | 0.69× |
| `metal/f32/one_hot/batched/b4_r1_c3_k2` | none | 0.74× | 0.65× |
| `metal/f32/one_hot/batched/b4_r1_c3_k33` | none | 0.72× | 0.58× |
| `metal/f32/one_hot/batched/b4_r1_c8_k2` | none | 0.67× | 0.66× |
| `metal/f32/one_hot/batched/b4_r1_c8_k33` | none | 0.62× | 0.52× |
| `metal/f32/one_hot/batched/b4_r3_c17_k2` | none | 0.50× | 0.55× |
| `metal/f32/one_hot/batched/b4_r3_c17_k33` | none | 0.40× | 0.36× |
| `metal/f32/one_hot/batched/b4_r3_c1_k2` | none | 0.67× | 0.68× |
| `metal/f32/one_hot/batched/b4_r3_c1_k33` | none | 0.65× | 0.62× |
| `metal/f32/one_hot/batched/b4_r3_c3_k2` | none | 0.64× | 0.64× |
| `metal/f32/one_hot/batched/b4_r3_c3_k33` | none | 0.57× | 0.57× |
| `metal/f32/one_hot/batched/b4_r3_c8_k2` | none | 0.58× | 0.62× |
| `metal/f32/one_hot/batched/b4_r3_c8_k33` | none | 0.49× | 0.52× |
| `metal/f32/one_hot/batched/b4_r7_c17_k2` | none | 0.51× | 0.41× |
| `metal/f32/one_hot/batched/b4_r7_c17_k33` | none | 0.25× | 0.25× |
| `metal/f32/one_hot/batched/b4_r7_c1_k2` | none | 0.70× | 0.64× |
| `metal/f32/one_hot/batched/b4_r7_c1_k33` | none | 0.54× | 0.57× |
| `metal/f32/one_hot/batched/b4_r7_c3_k2` | none | 0.58× | 0.62× |
| `metal/f32/one_hot/batched/b4_r7_c3_k33` | none | 0.50× | 0.56× |
| `metal/f32/one_hot/batched/b4_r7_c8_k2` | none | 0.55× | 0.55× |
| `metal/f32/one_hot/batched/b4_r7_c8_k33` | none | 0.37× | 0.37× |
| `metal/f32/one_hot/batched/b8_r16_c17_k2` | none | 0.29× | 0.24× |
| `metal/f32/one_hot/batched/b8_r16_c17_k33` | none | 0.21× | 0.21× |
| `metal/f32/one_hot/batched/b8_r16_c1_k2` | none | 0.53× | 0.53× |
| `metal/f32/one_hot/batched/b8_r16_c1_k33` | none | 0.39× | 0.36× |
| `metal/f32/one_hot/batched/b8_r16_c3_k2` | none | 0.47× | 0.49× |
| `metal/f32/one_hot/batched/b8_r16_c3_k33` | none | 0.24× | 0.23× |
| `metal/f32/one_hot/batched/b8_r16_c8_k2` | none | 0.41× | 0.42× |
| `metal/f32/one_hot/batched/b8_r16_c8_k33` | none | 0.19× | 0.20× |
| `metal/f32/one_hot/batched/b8_r1_c17_k2` | none | 0.58× | 0.60× |
| `metal/f32/one_hot/batched/b8_r1_c17_k33` | none | 0.40× | 0.40× |
| `metal/f32/one_hot/batched/b8_r1_c1_k2` | none | 0.62× | 0.67× |
| `metal/f32/one_hot/batched/b8_r1_c1_k33` | none | 0.62× | 0.64× |
| `metal/f32/one_hot/batched/b8_r1_c3_k2` | none | 0.65× | 0.65× |
| `metal/f32/one_hot/batched/b8_r1_c3_k33` | none | 0.56× | 0.57× |
| `metal/f32/one_hot/batched/b8_r1_c8_k2` | none | 0.61× | 0.62× |
| `metal/f32/one_hot/batched/b8_r1_c8_k33` | none | 0.51× | 0.51× |
| `metal/f32/one_hot/batched/b8_r3_c17_k2` | none | 0.47× | 0.48× |
| `metal/f32/one_hot/batched/b8_r3_c17_k33` | none | 0.23× | 0.23× |
| `metal/f32/one_hot/batched/b8_r3_c1_k2` | none | 0.63× | 0.65× |
| `metal/f32/one_hot/batched/b8_r3_c1_k33` | none | 0.56× | 0.58× |
| `metal/f32/one_hot/batched/b8_r3_c3_k2` | none | 0.57× | 0.65× |
| `metal/f32/one_hot/batched/b8_r3_c3_k33` | none | 0.46× | 0.46× |
| `metal/f32/one_hot/batched/b8_r3_c8_k2` | none | 0.51× | 0.58× |
| `metal/f32/one_hot/batched/b8_r3_c8_k33` | none | 0.32× | 0.37× |
| `metal/f32/one_hot/batched/b8_r7_c17_k2` | none | 0.41× | 0.43× |
| `metal/f32/one_hot/batched/b8_r7_c17_k33` | none | 0.19× | 0.19× |
| `metal/f32/one_hot/batched/b8_r7_c1_k2` | none | 0.61× | 0.60× |
| `metal/f32/one_hot/batched/b8_r7_c1_k33` | none | 0.47× | 0.48× |
| `metal/f32/one_hot/batched/b8_r7_c3_k2` | none | 0.48× | 0.49× |
| `metal/f32/one_hot/batched/b8_r7_c3_k33` | none | 0.39× | 0.34× |
| `metal/f32/one_hot/batched/b8_r7_c8_k2` | none | 0.56× | 0.45× |
| `metal/f32/one_hot/batched/b8_r7_c8_k33` | none | 0.24× | 0.26× |
