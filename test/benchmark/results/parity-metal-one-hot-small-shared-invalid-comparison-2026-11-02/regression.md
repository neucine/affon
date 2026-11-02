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
| `metal/f32/one_hot/invalid_index` | none | 0.02× | 0.45× |
| `metal/f32/one_hot/invalid_index/b1_n16_c16` | none | 0.04× | 0.48× |
| `metal/f32/one_hot/invalid_index/b1_n16_c17` | none | 0.04× | 0.38× |
| `metal/f32/one_hot/invalid_index/b1_n16_c2` | none | 0.03× | 0.41× |
| `metal/f32/one_hot/invalid_index/b1_n16_c3` | none | 0.03× | 0.49× |
| `metal/f32/one_hot/invalid_index/b1_n16_c31` | none | 0.04× | 0.40× |
| `metal/f32/one_hot/invalid_index/b1_n16_c32` | none | 0.04× | 0.42× |
| `metal/f32/one_hot/invalid_index/b1_n16_c33` | none | 0.05× | 0.47× |
| `metal/f32/one_hot/invalid_index/b1_n16_c7` | none | 0.04× | 0.42× |
| `metal/f32/one_hot/invalid_index/b1_n33_c16` | none | 0.05× | 0.45× |
| `metal/f32/one_hot/invalid_index/b1_n33_c17` | none | 0.05× | 0.43× |
| `metal/f32/one_hot/invalid_index/b1_n33_c2` | none | 0.04× | 0.51× |
| `metal/f32/one_hot/invalid_index/b1_n33_c3` | none | 0.04× | 0.47× |
| `metal/f32/one_hot/invalid_index/b1_n33_c31` | none | 0.05× | 0.44× |
| `metal/f32/one_hot/invalid_index/b1_n33_c32` | none | 0.06× | 0.43× |
| `metal/f32/one_hot/invalid_index/b1_n33_c33` | none | 0.06× | 0.44× |
| `metal/f32/one_hot/invalid_index/b1_n33_c7` | none | 0.05× | 0.47× |
| `metal/f32/one_hot/invalid_index/b1_n3_c16` | none | 0.03× | 0.46× |
| `metal/f32/one_hot/invalid_index/b1_n3_c17` | none | 0.03× | 0.42× |
| `metal/f32/one_hot/invalid_index/b1_n3_c3` | none | 0.02× | 0.41× |
| `metal/f32/one_hot/invalid_index/b1_n3_c31` | none | 0.03× | 0.41× |
| `metal/f32/one_hot/invalid_index/b1_n3_c32` | none | 0.03× | 0.41× |
| `metal/f32/one_hot/invalid_index/b1_n3_c33` | none | 0.03× | 0.45× |
| `metal/f32/one_hot/invalid_index/b1_n3_c7` | none | 0.02× | 0.42× |
| `metal/f32/one_hot/invalid_index/b1_n7_c16` | none | 0.03× | 0.46× |
| `metal/f32/one_hot/invalid_index/b1_n7_c17` | none | 0.03× | 0.45× |
| `metal/f32/one_hot/invalid_index/b1_n7_c2` | none | 0.03× | 0.42× |
| `metal/f32/one_hot/invalid_index/b1_n7_c3` | none | 0.03× | 0.44× |
| `metal/f32/one_hot/invalid_index/b1_n7_c31` | none | 0.03× | 0.51× |
| `metal/f32/one_hot/invalid_index/b1_n7_c32` | none | 0.03× | 0.49× |
| `metal/f32/one_hot/invalid_index/b1_n7_c33` | none | 0.03× | 0.47× |
| `metal/f32/one_hot/invalid_index/b1_n7_c7` | none | 0.03× | 0.49× |
| `metal/f32/one_hot/invalid_index/b2_n16_c16` | none | 0.05× | 0.14× |
| `metal/f32/one_hot/invalid_index/b2_n16_c17` | none | 0.05× | 0.46× |
| `metal/f32/one_hot/invalid_index/b2_n16_c2` | none | 0.03× | 0.40× |
| `metal/f32/one_hot/invalid_index/b2_n16_c3` | none | 0.04× | 0.48× |
| `metal/f32/one_hot/invalid_index/b2_n16_c31` | none | 0.05× | 0.39× |
| `metal/f32/one_hot/invalid_index/b2_n16_c32` | none | 0.05× | 0.37× |
| `metal/f32/one_hot/invalid_index/b2_n16_c33` | none | 0.06× | 0.34× |
| `metal/f32/one_hot/invalid_index/b2_n16_c7` | none | 0.04× | 0.42× |
| `metal/f32/one_hot/invalid_index/b2_n33_c16` | none | 0.07× | 0.38× |
| `metal/f32/one_hot/invalid_index/b2_n33_c17` | none | 0.07× | 0.46× |
| `metal/f32/one_hot/invalid_index/b2_n33_c2` | none | 0.04× | 0.41× |
| `metal/f32/one_hot/invalid_index/b2_n33_c3` | none | 0.05× | 0.42× |
| `metal/f32/one_hot/invalid_index/b2_n33_c31` | none | 0.09× | 0.33× |
| `metal/f32/one_hot/invalid_index/b2_n33_c32` | none | 0.09× | 0.34× |
| `metal/f32/one_hot/invalid_index/b2_n33_c33` | none | 0.07× | 0.39× |
| `metal/f32/one_hot/invalid_index/b2_n33_c7` | none | 0.06× | 0.39× |
| `metal/f32/one_hot/invalid_index/b2_n3_c16` | none | 0.03× | 0.51× |
| `metal/f32/one_hot/invalid_index/b2_n3_c17` | none | 0.03× | 0.59× |
| `metal/f32/one_hot/invalid_index/b2_n3_c2` | none | 0.03× | 0.42× |
| `metal/f32/one_hot/invalid_index/b2_n3_c3` | none | 0.03× | 0.60× |
| `metal/f32/one_hot/invalid_index/b2_n3_c31` | none | 0.03× | 0.50× |
| `metal/f32/one_hot/invalid_index/b2_n3_c32` | none | 0.03× | 0.53× |
| `metal/f32/one_hot/invalid_index/b2_n3_c33` | none | 0.03× | 0.53× |
| `metal/f32/one_hot/invalid_index/b2_n3_c7` | none | 0.03× | 0.42× |
| `metal/f32/one_hot/invalid_index/b2_n7_c16` | none | 0.03× | 0.57× |
| `metal/f32/one_hot/invalid_index/b2_n7_c17` | none | 0.04× | 0.55× |
| `metal/f32/one_hot/invalid_index/b2_n7_c2` | none | 0.03× | 0.62× |
| `metal/f32/one_hot/invalid_index/b2_n7_c3` | none | 0.03× | 0.53× |
| `metal/f32/one_hot/invalid_index/b2_n7_c31` | none | 0.04× | 0.50× |
| `metal/f32/one_hot/invalid_index/b2_n7_c32` | none | 0.04× | 0.46× |
| `metal/f32/one_hot/invalid_index/b2_n7_c33` | none | 0.04× | 0.45× |
| `metal/f32/one_hot/invalid_index/b2_n7_c7` | none | 0.03× | 0.53× |
| `metal/f32/one_hot/invalid_index/b4_n16_c16` | none | 0.06× | 0.50× |
| `metal/f32/one_hot/invalid_index/b4_n16_c17` | none | 0.07× | 0.50× |
| `metal/f32/one_hot/invalid_index/b4_n16_c2` | none | 0.04× | 0.56× |
| `metal/f32/one_hot/invalid_index/b4_n16_c3` | none | 0.05× | 0.57× |
| `metal/f32/one_hot/invalid_index/b4_n16_c31` | none | 0.08× | 0.43× |
| `metal/f32/one_hot/invalid_index/b4_n16_c32` | none | 0.09× | 0.40× |
| `metal/f32/one_hot/invalid_index/b4_n16_c33` | none | 0.06× | 0.42× |
| `metal/f32/one_hot/invalid_index/b4_n16_c7` | none | 0.06× | 0.47× |
| `metal/f32/one_hot/invalid_index/b4_n33_c16` | none | 0.10× | 0.45× |
| `metal/f32/one_hot/invalid_index/b4_n33_c17` | none | 0.11× | 0.41× |
| `metal/f32/one_hot/invalid_index/b4_n33_c2` | none | 0.06× | 0.49× |
| `metal/f32/one_hot/invalid_index/b4_n33_c3` | none | 0.06× | 0.44× |
| `metal/f32/one_hot/invalid_index/b4_n33_c31` | none | 0.10× | 0.33× |
| `metal/f32/one_hot/invalid_index/b4_n33_c32` | none | 0.11× | 0.38× |
| `metal/f32/one_hot/invalid_index/b4_n33_c33` | none | 0.10× | 0.35× |
| `metal/f32/one_hot/invalid_index/b4_n33_c7` | none | 0.10× | 0.52× |
| `metal/f32/one_hot/invalid_index/b4_n3_c16` | none | 0.03× | 0.42× |
| `metal/f32/one_hot/invalid_index/b4_n3_c17` | none | 0.03× | 0.47× |
| `metal/f32/one_hot/invalid_index/b4_n3_c2` | none | 0.03× | 0.52× |
| `metal/f32/one_hot/invalid_index/b4_n3_c3` | none | 0.03× | 0.32× |
| `metal/f32/one_hot/invalid_index/b4_n3_c31` | none | 0.03× | 0.41× |
| `metal/f32/one_hot/invalid_index/b4_n3_c32` | none | 0.04× | 0.39× |
| `metal/f32/one_hot/invalid_index/b4_n3_c33` | none | 0.04× | 0.33× |
| `metal/f32/one_hot/invalid_index/b4_n3_c7` | none | 0.03× | 0.45× |
| `metal/f32/one_hot/invalid_index/b4_n7_c16` | none | 0.05× | 0.48× |
| `metal/f32/one_hot/invalid_index/b4_n7_c17` | none | 0.04× | 0.35× |
| `metal/f32/one_hot/invalid_index/b4_n7_c2` | none | 0.02× | 0.33× |
| `metal/f32/one_hot/invalid_index/b4_n7_c3` | none | 0.03× | 0.38× |
| `metal/f32/one_hot/invalid_index/b4_n7_c31` | none | 0.05× | 0.72× |
| `metal/f32/one_hot/invalid_index/b4_n7_c32` | none | 0.05× | 0.43× |
| `metal/f32/one_hot/invalid_index/b4_n7_c33` | none | 0.05× | 0.45× |
| `metal/f32/one_hot/invalid_index/b4_n7_c7` | none | 0.05× | 0.45× |
| `metal/f32/one_hot/invalid_index/b8_n16_c16` | none | 0.10× | 0.39× |
| `metal/f32/one_hot/invalid_index/b8_n16_c17` | none | 0.10× | 0.40× |
| `metal/f32/one_hot/invalid_index/b8_n16_c2` | none | 0.06× | 0.50× |
| `metal/f32/one_hot/invalid_index/b8_n16_c3` | none | 0.09× | 0.50× |
| `metal/f32/one_hot/invalid_index/b8_n16_c31` | none | 0.11× | 0.35× |
| `metal/f32/one_hot/invalid_index/b8_n16_c32` | none | 0.11× | 0.34× |
| `metal/f32/one_hot/invalid_index/b8_n16_c33` | none | 0.09× | 0.38× |
| `metal/f32/one_hot/invalid_index/b8_n16_c7` | none | 0.09× | 0.45× |
| `metal/f32/one_hot/invalid_index/b8_n33_c16` | none | 0.13× | 0.32× |
| `metal/f32/one_hot/invalid_index/b8_n33_c17` | none | 0.14× | 0.34× |
| `metal/f32/one_hot/invalid_index/b8_n33_c2` | none | 0.09× | 0.44× |
| `metal/f32/one_hot/invalid_index/b8_n33_c3` | none | 0.09× | 0.41× |
| `metal/f32/one_hot/invalid_index/b8_n33_c31` | none | 0.13× | 0.26× |
| `metal/f32/one_hot/invalid_index/b8_n33_c32` | none | 0.12× | 0.24× |
| `metal/f32/one_hot/invalid_index/b8_n33_c33` | none | 0.13× | 0.25× |
| `metal/f32/one_hot/invalid_index/b8_n33_c7` | none | 0.14× | 0.49× |
| `metal/f32/one_hot/invalid_index/b8_n3_c16` | none | 0.04× | 0.48× |
| `metal/f32/one_hot/invalid_index/b8_n3_c17` | none | 0.05× | 0.48× |
| `metal/f32/one_hot/invalid_index/b8_n3_c2` | none | 0.03× | 0.45× |
| `metal/f32/one_hot/invalid_index/b8_n3_c3` | none | 0.04× | 0.52× |
| `metal/f32/one_hot/invalid_index/b8_n3_c31` | none | 0.05× | 0.54× |
| `metal/f32/one_hot/invalid_index/b8_n3_c32` | none | 0.05× | 0.47× |
| `metal/f32/one_hot/invalid_index/b8_n3_c33` | none | 0.05× | 0.44× |
| `metal/f32/one_hot/invalid_index/b8_n3_c7` | none | 0.04× | 0.61× |
| `metal/f32/one_hot/invalid_index/b8_n7_c16` | none | 0.06× | 0.51× |
| `metal/f32/one_hot/invalid_index/b8_n7_c17` | none | 0.08× | 0.52× |
| `metal/f32/one_hot/invalid_index/b8_n7_c2` | none | 0.04× | 0.49× |
| `metal/f32/one_hot/invalid_index/b8_n7_c3` | none | 0.05× | 0.53× |
| `metal/f32/one_hot/invalid_index/b8_n7_c31` | none | 0.07× | 0.38× |
| `metal/f32/one_hot/invalid_index/b8_n7_c32` | none | 0.07× | 0.48× |
| `metal/f32/one_hot/invalid_index/b8_n7_c33` | none | 0.09× | 0.39× |
| `metal/f32/one_hot/invalid_index/b8_n7_c7` | none | 0.06× | 0.59× |
