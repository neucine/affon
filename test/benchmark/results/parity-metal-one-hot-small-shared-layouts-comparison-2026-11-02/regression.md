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
| `metal/f32/one_hot/large/offset_b4_n16_c16` | none | 0.14× | 0.46× |
| `metal/f32/one_hot/large/offset_b4_n16_c2` | none | 0.11× | 0.44× |
| `metal/f32/one_hot/large/offset_b4_n16_c33` | none | 0.13× | 0.40× |
| `metal/f32/one_hot/large/offset_b4_n16_c7` | none | 0.16× | 0.49× |
| `metal/f32/one_hot/large/offset_b4_n33_c16` | none | 0.12× | 0.38× |
| `metal/f32/one_hot/large/offset_b4_n33_c2` | none | 0.13× | 0.47× |
| `metal/f32/one_hot/large/offset_b4_n33_c33` | none | 0.13× | 0.31× |
| `metal/f32/one_hot/large/offset_b4_n33_c7` | none | 0.14× | 0.43× |
| `metal/f32/one_hot/large/offset_b4_n3_c16` | none | 0.11× | 0.52× |
| `metal/f32/one_hot/large/offset_b4_n3_c2` | none | 0.08× | 0.52× |
| `metal/f32/one_hot/large/offset_b4_n3_c33` | none | 0.11× | 0.44× |
| `metal/f32/one_hot/large/offset_b4_n3_c7` | none | 0.09× | 0.52× |
| `metal/f32/one_hot/large/offset_b4_n7_c16` | none | 0.10× | 0.47× |
| `metal/f32/one_hot/large/offset_b4_n7_c2` | none | 0.10× | 0.38× |
| `metal/f32/one_hot/large/offset_b4_n7_c33` | none | 0.10× | 0.41× |
| `metal/f32/one_hot/large/offset_b4_n7_c7` | none | 0.11× | 0.54× |
| `metal/f32/one_hot/large/offset_b8_n16_c16` | none | 0.10× | 0.33× |
| `metal/f32/one_hot/large/offset_b8_n16_c2` | none | 0.14× | 0.45× |
| `metal/f32/one_hot/large/offset_b8_n16_c33` | none | 0.09× | 0.22× |
| `metal/f32/one_hot/large/offset_b8_n16_c7` | none | 0.18× | 0.41× |
| `metal/f32/one_hot/large/offset_b8_n33_c16` | none | 0.11× | 0.30× |
| `metal/f32/one_hot/large/offset_b8_n33_c2` | none | 0.15× | 0.40× |
| `metal/f32/one_hot/large/offset_b8_n33_c33` | none | 0.13× | 0.22× |
| `metal/f32/one_hot/large/offset_b8_n33_c7` | none | 0.18× | 0.40× |
| `metal/f32/one_hot/large/offset_b8_n3_c16` | none | 0.12× | 0.48× |
| `metal/f32/one_hot/large/offset_b8_n3_c2` | none | 0.09× | 0.49× |
| `metal/f32/one_hot/large/offset_b8_n3_c33` | none | 0.11× | 0.41× |
| `metal/f32/one_hot/large/offset_b8_n3_c7` | none | 0.11× | 0.45× |
| `metal/f32/one_hot/large/offset_b8_n7_c16` | none | 0.12× | 0.44× |
| `metal/f32/one_hot/large/offset_b8_n7_c2` | none | 0.11× | 0.45× |
| `metal/f32/one_hot/large/offset_b8_n7_c33` | none | 0.11× | 0.41× |
| `metal/f32/one_hot/large/offset_b8_n7_c7` | none | 0.14× | 0.51× |
| `metal/f32/one_hot/small/offset_b1_n16_c16` | none | 0.11× | 0.53× |
| `metal/f32/one_hot/small/offset_b1_n16_c2` | none | 0.08× | 0.55× |
| `metal/f32/one_hot/small/offset_b1_n16_c33` | none | 0.11× | 0.48× |
| `metal/f32/one_hot/small/offset_b1_n16_c7` | none | 0.08× | 0.51× |
| `metal/f32/one_hot/small/offset_b1_n33_c16` | none | 0.10× | 0.49× |
| `metal/f32/one_hot/small/offset_b1_n33_c2` | none | 0.09× | 0.49× |
| `metal/f32/one_hot/small/offset_b1_n33_c33` | none | 0.12× | 0.47× |
| `metal/f32/one_hot/small/offset_b1_n33_c7` | none | 0.11× | 0.51× |
| `metal/f32/one_hot/small/offset_b1_n3_c16` | none | 0.07× | 0.46× |
| `metal/f32/one_hot/small/offset_b1_n3_c2` | none | 0.07× | 0.49× |
| `metal/f32/one_hot/small/offset_b1_n3_c33` | none | 0.08× | 0.51× |
| `metal/f32/one_hot/small/offset_b1_n3_c7` | none | 0.07× | 0.52× |
| `metal/f32/one_hot/small/offset_b1_n7_c16` | none | 0.09× | 0.44× |
| `metal/f32/one_hot/small/offset_b1_n7_c2` | none | 0.08× | 0.47× |
| `metal/f32/one_hot/small/offset_b1_n7_c33` | none | 0.09× | 0.53× |
| `metal/f32/one_hot/small/offset_b1_n7_c7` | none | 0.07× | 0.53× |
| `metal/f32/one_hot/small/offset_b2_n16_c16` | none | 0.12× | 0.44× |
| `metal/f32/one_hot/small/offset_b2_n16_c2` | none | 0.09× | 0.54× |
| `metal/f32/one_hot/small/offset_b2_n16_c33` | none | 0.11× | 0.50× |
| `metal/f32/one_hot/small/offset_b2_n16_c7` | none | 0.12× | 0.46× |
| `metal/f32/one_hot/small/offset_b2_n33_c16` | none | 0.06× | 0.43× |
| `metal/f32/one_hot/small/offset_b2_n33_c2` | none | 0.09× | 0.48× |
| `metal/f32/one_hot/small/offset_b2_n33_c33` | none | 0.09× | 0.36× |
| `metal/f32/one_hot/small/offset_b2_n33_c7` | none | 0.10× | 0.47× |
| `metal/f32/one_hot/small/offset_b2_n3_c16` | none | 0.08× | 0.57× |
| `metal/f32/one_hot/small/offset_b2_n3_c2` | none | 0.08× | 0.57× |
| `metal/f32/one_hot/small/offset_b2_n3_c33` | none | 0.08× | 0.52× |
| `metal/f32/one_hot/small/offset_b2_n3_c7` | none | 0.08× | 0.47× |
| `metal/f32/one_hot/small/offset_b2_n7_c16` | none | 0.06× | 0.51× |
| `metal/f32/one_hot/small/offset_b2_n7_c2` | none | 0.07× | 0.52× |
| `metal/f32/one_hot/small/offset_b2_n7_c33` | none | 0.09× | 0.48× |
| `metal/f32/one_hot/small/offset_b2_n7_c7` | none | 0.08× | 0.51× |
| `metal/f32/one_hot/strided` | none | 0.07× | 0.52× |
| `metal/f32/one_hot/strided/b1_n16_c16` | none | 0.09× | 0.46× |
| `metal/f32/one_hot/strided/b1_n16_c2` | none | 0.08× | 0.56× |
| `metal/f32/one_hot/strided/b1_n16_c33` | none | 0.10× | 0.43× |
| `metal/f32/one_hot/strided/b1_n16_c7` | none | 0.10× | 0.50× |
| `metal/f32/one_hot/strided/b1_n33_c16` | none | 0.11× | 0.39× |
| `metal/f32/one_hot/strided/b1_n33_c2` | none | 0.09× | 0.23× |
| `metal/f32/one_hot/strided/b1_n33_c33` | none | 0.11× | 0.45× |
| `metal/f32/one_hot/strided/b1_n33_c7` | none | 0.12× | 0.53× |
| `metal/f32/one_hot/strided/b1_n3_c16` | none | 0.07× | 0.47× |
| `metal/f32/one_hot/strided/b1_n3_c33` | none | 0.07× | 0.49× |
| `metal/f32/one_hot/strided/b1_n3_c7` | none | 0.07× | 0.48× |
| `metal/f32/one_hot/strided/b1_n7_c16` | none | 0.08× | 0.41× |
| `metal/f32/one_hot/strided/b1_n7_c2` | none | 0.07× | 0.51× |
| `metal/f32/one_hot/strided/b1_n7_c33` | none | 0.08× | 0.44× |
| `metal/f32/one_hot/strided/b1_n7_c7` | none | 0.08× | 0.52× |
| `metal/f32/one_hot/strided/b2_n16_c16` | none | 0.13× | 0.47× |
| `metal/f32/one_hot/strided/b2_n16_c2` | none | 0.10× | 0.54× |
| `metal/f32/one_hot/strided/b2_n16_c33` | none | 0.12× | 0.43× |
| `metal/f32/one_hot/strided/b2_n16_c7` | none | 0.11× | 0.54× |
| `metal/f32/one_hot/strided/b2_n33_c16` | none | 0.09× | 0.44× |
| `metal/f32/one_hot/strided/b2_n33_c2` | none | 0.11× | 0.51× |
| `metal/f32/one_hot/strided/b2_n33_c33` | none | 0.13× | 0.41× |
| `metal/f32/one_hot/strided/b2_n33_c7` | none | 0.16× | 0.46× |
| `metal/f32/one_hot/strided/b2_n3_c16` | none | 0.05× | 0.51× |
| `metal/f32/one_hot/strided/b2_n3_c2` | none | 0.06× | 0.48× |
| `metal/f32/one_hot/strided/b2_n3_c33` | none | 0.08× | 0.47× |
| `metal/f32/one_hot/strided/b2_n3_c7` | none | 0.05× | 0.50× |
| `metal/f32/one_hot/strided/b2_n7_c16` | none | 0.09× | 0.51× |
| `metal/f32/one_hot/strided/b2_n7_c2` | none | 0.08× | 0.51× |
| `metal/f32/one_hot/strided/b2_n7_c33` | none | 0.10× | 0.51× |
| `metal/f32/one_hot/strided/b2_n7_c7` | none | 0.09× | 0.50× |
| `metal/f32/one_hot/strided/b4_n16_c16` | none | 0.13× | 0.46× |
| `metal/f32/one_hot/strided/b4_n16_c2` | none | 0.10× | 0.51× |
| `metal/f32/one_hot/strided/b4_n16_c33` | none | 0.14× | 0.31× |
| `metal/f32/one_hot/strided/b4_n16_c7` | none | 0.16× | 0.39× |
| `metal/f32/one_hot/strided/b4_n33_c16` | none | 0.15× | 0.31× |
| `metal/f32/one_hot/strided/b4_n33_c2` | none | 0.13× | 0.46× |
| `metal/f32/one_hot/strided/b4_n33_c33` | none | 0.14× | 0.30× |
| `metal/f32/one_hot/strided/b4_n33_c7` | none | 0.16× | 0.44× |
| `metal/f32/one_hot/strided/b4_n3_c16` | none | 0.09× | 0.56× |
| `metal/f32/one_hot/strided/b4_n3_c2` | none | 0.09× | 0.52× |
| `metal/f32/one_hot/strided/b4_n3_c33` | none | 0.06× | 0.46× |
| `metal/f32/one_hot/strided/b4_n3_c7` | none | 0.11× | 0.50× |
| `metal/f32/one_hot/strided/b4_n7_c16` | none | 0.11× | 0.50× |
| `metal/f32/one_hot/strided/b4_n7_c2` | none | 0.10× | 0.55× |
| `metal/f32/one_hot/strided/b4_n7_c33` | none | 0.12× | 0.42× |
| `metal/f32/one_hot/strided/b4_n7_c7` | none | 0.12× | 0.51× |
| `metal/f32/one_hot/strided/b8_n16_c16` | none | 0.14× | 0.41× |
| `metal/f32/one_hot/strided/b8_n16_c2` | none | 0.13× | 0.55× |
| `metal/f32/one_hot/strided/b8_n16_c33` | none | 0.14× | 0.33× |
| `metal/f32/one_hot/strided/b8_n16_c7` | none | 0.18× | 0.46× |
| `metal/f32/one_hot/strided/b8_n33_c16` | none | 0.12× | 0.30× |
| `metal/f32/one_hot/strided/b8_n33_c2` | none | 0.14× | 0.45× |
| `metal/f32/one_hot/strided/b8_n33_c33` | none | 0.13× | 0.23× |
| `metal/f32/one_hot/strided/b8_n33_c7` | none | 0.18× | 0.40× |
| `metal/f32/one_hot/strided/b8_n3_c16` | none | 0.11× | 0.47× |
| `metal/f32/one_hot/strided/b8_n3_c2` | none | 0.08× | 0.49× |
| `metal/f32/one_hot/strided/b8_n3_c33` | none | 0.10× | 0.46× |
| `metal/f32/one_hot/strided/b8_n3_c7` | none | 0.11× | 0.49× |
| `metal/f32/one_hot/strided/b8_n7_c16` | none | 0.13× | 0.46× |
| `metal/f32/one_hot/strided/b8_n7_c2` | none | 0.11× | 0.54× |
| `metal/f32/one_hot/strided/b8_n7_c33` | none | 0.13× | 0.37× |
| `metal/f32/one_hot/strided/b8_n7_c7` | none | 0.15× | 0.68× |
