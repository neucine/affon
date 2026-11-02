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
| `metal/f32/one_hot/large` | none | 0.03× | 0.54× |
| `metal/f32/one_hot/large/b4_n17_c16` | none | 0.07× | 0.53× |
| `metal/f32/one_hot/large/b4_n17_c17` | none | 0.06× | 0.44× |
| `metal/f32/one_hot/large/b4_n17_c2` | none | 0.04× | 0.49× |
| `metal/f32/one_hot/large/b4_n17_c3` | none | 0.06× | 0.52× |
| `metal/f32/one_hot/large/b4_n17_c31` | none | 0.09× | 0.40× |
| `metal/f32/one_hot/large/b4_n17_c32` | none | 0.09× | 0.43× |
| `metal/f32/one_hot/large/b4_n17_c33` | none | 0.09× | 0.44× |
| `metal/f32/one_hot/large/b4_n17_c7` | none | 0.07× | 0.49× |
| `metal/f32/one_hot/large/b4_n1_c16` | none | 0.03× | 0.56× |
| `metal/f32/one_hot/large/b4_n1_c17` | none | 0.03× | 0.53× |
| `metal/f32/one_hot/large/b4_n1_c3` | none | 0.03× | 0.58× |
| `metal/f32/one_hot/large/b4_n1_c31` | none | 0.03× | 0.55× |
| `metal/f32/one_hot/large/b4_n1_c32` | none | 0.03× | 0.57× |
| `metal/f32/one_hot/large/b4_n1_c33` | none | 0.02× | 0.53× |
| `metal/f32/one_hot/large/b4_n1_c7` | none | 0.03× | 0.53× |
| `metal/f32/one_hot/large/b4_n3_c16` | none | 0.04× | 0.56× |
| `metal/f32/one_hot/large/b4_n3_c17` | none | 0.04× | 0.56× |
| `metal/f32/one_hot/large/b4_n3_c2` | none | 0.01× | 0.65× |
| `metal/f32/one_hot/large/b4_n3_c3` | none | 0.03× | 0.54× |
| `metal/f32/one_hot/large/b4_n3_c31` | none | 0.04× | 0.48× |
| `metal/f32/one_hot/large/b4_n3_c32` | none | 0.04× | 0.48× |
| `metal/f32/one_hot/large/b4_n3_c33` | none | 0.04× | 0.47× |
| `metal/f32/one_hot/large/b4_n3_c7` | none | 0.04× | 0.55× |
| `metal/f32/one_hot/large/b4_n8_c16` | none | 0.06× | 0.50× |
| `metal/f32/one_hot/large/b4_n8_c17` | none | 0.06× | 0.48× |
| `metal/f32/one_hot/large/b4_n8_c2` | none | 0.04× | 0.58× |
| `metal/f32/one_hot/large/b4_n8_c3` | none | 0.03× | 0.53× |
| `metal/f32/one_hot/large/b4_n8_c31` | none | 0.05× | 0.49× |
| `metal/f32/one_hot/large/b4_n8_c32` | none | 0.06× | 0.44× |
| `metal/f32/one_hot/large/b4_n8_c33` | none | 0.06× | 0.49× |
| `metal/f32/one_hot/large/b4_n8_c7` | none | 0.05× | 0.53× |
| `metal/f32/one_hot/large/b8_n17_c16` | none | 0.10× | 0.39× |
| `metal/f32/one_hot/large/b8_n17_c17` | none | 0.11× | 0.43× |
| `metal/f32/one_hot/large/b8_n17_c2` | none | 0.06× | 0.53× |
| `metal/f32/one_hot/large/b8_n17_c3` | none | 0.08× | 0.50× |
| `metal/f32/one_hot/large/b8_n17_c31` | none | 0.11× | 0.36× |
| `metal/f32/one_hot/large/b8_n17_c32` | none | 0.10× | 0.35× |
| `metal/f32/one_hot/large/b8_n17_c33` | none | 0.14× | 0.32× |
| `metal/f32/one_hot/large/b8_n17_c7` | none | 0.11× | 0.45× |
| `metal/f32/one_hot/large/b8_n1_c16` | none | 0.03× | 0.47× |
| `metal/f32/one_hot/large/b8_n1_c17` | none | 0.03× | 0.48× |
| `metal/f32/one_hot/large/b8_n1_c2` | none | 0.03× | 0.55× |
| `metal/f32/one_hot/large/b8_n1_c3` | none | 0.03× | 0.53× |
| `metal/f32/one_hot/large/b8_n1_c31` | none | 0.04× | 0.53× |
| `metal/f32/one_hot/large/b8_n1_c32` | none | 0.04× | 0.55× |
| `metal/f32/one_hot/large/b8_n1_c33` | none | 0.04× | 0.52× |
| `metal/f32/one_hot/large/b8_n1_c7` | none | 0.03× | 0.53× |
| `metal/f32/one_hot/large/b8_n3_c16` | none | 0.05× | 0.53× |
| `metal/f32/one_hot/large/b8_n3_c17` | none | 0.05× | 0.54× |
| `metal/f32/one_hot/large/b8_n3_c2` | none | 0.03× | 0.62× |
| `metal/f32/one_hot/large/b8_n3_c3` | none | 0.03× | 0.56× |
| `metal/f32/one_hot/large/b8_n3_c31` | none | 0.05× | 0.48× |
| `metal/f32/one_hot/large/b8_n3_c32` | none | 0.06× | 0.53× |
| `metal/f32/one_hot/large/b8_n3_c33` | none | 0.05× | 0.49× |
| `metal/f32/one_hot/large/b8_n3_c7` | none | 0.04× | 0.50× |
| `metal/f32/one_hot/large/b8_n8_c16` | none | 0.08× | 0.52× |
| `metal/f32/one_hot/large/b8_n8_c17` | none | 0.08× | 0.43× |
| `metal/f32/one_hot/large/b8_n8_c2` | none | 0.05× | 0.46× |
| `metal/f32/one_hot/large/b8_n8_c3` | none | 0.05× | 0.55× |
| `metal/f32/one_hot/large/b8_n8_c31` | none | 0.09× | 0.39× |
| `metal/f32/one_hot/large/b8_n8_c32` | none | 0.08× | 0.42× |
| `metal/f32/one_hot/large/b8_n8_c33` | none | 0.08× | 0.43× |
| `metal/f32/one_hot/large/b8_n8_c7` | none | 0.07× | 0.44× |
| `metal/f32/one_hot/small` | none | 0.02× | 0.47× |
| `metal/f32/one_hot/small/b1_n17_c16` | none | 0.03× | 0.49× |
| `metal/f32/one_hot/small/b1_n17_c17` | none | 0.04× | 0.52× |
| `metal/f32/one_hot/small/b1_n17_c2` | none | 0.03× | 0.50× |
| `metal/f32/one_hot/small/b1_n17_c3` | none | 0.02× | 0.51× |
| `metal/f32/one_hot/small/b1_n17_c31` | none | 0.04× | 0.50× |
| `metal/f32/one_hot/small/b1_n17_c32` | none | 0.03× | 0.47× |
| `metal/f32/one_hot/small/b1_n17_c33` | none | 0.05× | 0.47× |
| `metal/f32/one_hot/small/b1_n17_c7` | none | 0.03× | 0.57× |
| `metal/f32/one_hot/small/b1_n1_c16` | none | 0.03× | 0.49× |
| `metal/f32/one_hot/small/b1_n1_c17` | none | 0.02× | 0.52× |
| `metal/f32/one_hot/small/b1_n1_c3` | none | 0.02× | 0.49× |
| `metal/f32/one_hot/small/b1_n1_c31` | none | 0.02× | 0.49× |
| `metal/f32/one_hot/small/b1_n1_c32` | none | 0.02× | 0.55× |
| `metal/f32/one_hot/small/b1_n1_c33` | none | 0.03× | 0.60× |
| `metal/f32/one_hot/small/b1_n1_c7` | none | 0.03× | 0.47× |
| `metal/f32/one_hot/small/b1_n3_c16` | none | 0.03× | 0.53× |
| `metal/f32/one_hot/small/b1_n3_c17` | none | 0.03× | 0.48× |
| `metal/f32/one_hot/small/b1_n3_c2` | none | 0.03× | 0.40× |
| `metal/f32/one_hot/small/b1_n3_c3` | none | 0.02× | 0.53× |
| `metal/f32/one_hot/small/b1_n3_c31` | none | 0.03× | 0.50× |
| `metal/f32/one_hot/small/b1_n3_c32` | none | 0.02× | 0.52× |
| `metal/f32/one_hot/small/b1_n3_c33` | none | 0.03× | 0.56× |
| `metal/f32/one_hot/small/b1_n3_c7` | none | 0.03× | 0.54× |
| `metal/f32/one_hot/small/b1_n8_c16` | none | 0.03× | 0.48× |
| `metal/f32/one_hot/small/b1_n8_c17` | none | 0.03× | 0.55× |
| `metal/f32/one_hot/small/b1_n8_c2` | none | 0.03× | 0.53× |
| `metal/f32/one_hot/small/b1_n8_c3` | none | 0.03× | 0.56× |
| `metal/f32/one_hot/small/b1_n8_c31` | none | 0.03× | 0.45× |
| `metal/f32/one_hot/small/b1_n8_c32` | none | 0.04× | 0.44× |
| `metal/f32/one_hot/small/b1_n8_c33` | none | 0.03× | 0.55× |
| `metal/f32/one_hot/small/b1_n8_c7` | none | 0.03× | 0.51× |
| `metal/f32/one_hot/small/b2_n17_c16` | none | 0.05× | 0.51× |
| `metal/f32/one_hot/small/b2_n17_c17` | none | 0.06× | 0.53× |
| `metal/f32/one_hot/small/b2_n17_c2` | none | 0.04× | 0.54× |
| `metal/f32/one_hot/small/b2_n17_c3` | none | 0.04× | 0.52× |
| `metal/f32/one_hot/small/b2_n17_c31` | none | 0.06× | 0.45× |
| `metal/f32/one_hot/small/b2_n17_c32` | none | 0.06× | 0.43× |
| `metal/f32/one_hot/small/b2_n17_c33` | none | 0.06× | 0.46× |
| `metal/f32/one_hot/small/b2_n17_c7` | none | 0.05× | 0.51× |
| `metal/f32/one_hot/small/b2_n1_c16` | none | 0.03× | 0.53× |
| `metal/f32/one_hot/small/b2_n1_c17` | none | 0.03× | 0.50× |
| `metal/f32/one_hot/small/b2_n1_c2` | none | 0.03× | 0.50× |
| `metal/f32/one_hot/small/b2_n1_c3` | none | 0.03× | 0.52× |
| `metal/f32/one_hot/small/b2_n1_c31` | none | 0.03× | 0.51× |
| `metal/f32/one_hot/small/b2_n1_c32` | none | 0.03× | 0.51× |
| `metal/f32/one_hot/small/b2_n1_c33` | none | 0.03× | 0.52× |
| `metal/f32/one_hot/small/b2_n1_c7` | none | 0.02× | 0.53× |
| `metal/f32/one_hot/small/b2_n3_c16` | none | 0.03× | 0.46× |
| `metal/f32/one_hot/small/b2_n3_c17` | none | 0.03× | 0.46× |
| `metal/f32/one_hot/small/b2_n3_c2` | none | 0.03× | 0.54× |
| `metal/f32/one_hot/small/b2_n3_c3` | none | 0.02× | 0.57× |
| `metal/f32/one_hot/small/b2_n3_c31` | none | 0.02× | 0.42× |
| `metal/f32/one_hot/small/b2_n3_c32` | none | 0.03× | 0.51× |
| `metal/f32/one_hot/small/b2_n3_c33` | none | 0.03× | 0.51× |
| `metal/f32/one_hot/small/b2_n3_c7` | none | 0.03× | 0.45× |
| `metal/f32/one_hot/small/b2_n8_c16` | none | 0.04× | 0.57× |
| `metal/f32/one_hot/small/b2_n8_c17` | none | 0.04× | 0.46× |
| `metal/f32/one_hot/small/b2_n8_c2` | none | 0.03× | 0.60× |
| `metal/f32/one_hot/small/b2_n8_c3` | none | 0.03× | 0.66× |
| `metal/f32/one_hot/small/b2_n8_c31` | none | 0.04× | 0.47× |
| `metal/f32/one_hot/small/b2_n8_c32` | none | 0.05× | 0.50× |
| `metal/f32/one_hot/small/b2_n8_c33` | none | 0.05× | 0.53× |
| `metal/f32/one_hot/small/b2_n8_c7` | none | 0.04× | 0.50× |
