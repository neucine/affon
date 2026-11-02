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
| `metal/f32/one_hot/batched` | none | 0.03× | 0.48× |
| `metal/f32/one_hot/batched/b1_r16_c17_k2` | none | 0.10× | 0.47× |
| `metal/f32/one_hot/batched/b1_r16_c17_k33` | none | 0.14× | 0.24× |
| `metal/f32/one_hot/batched/b1_r16_c1_k2` | none | 0.03× | 0.52× |
| `metal/f32/one_hot/batched/b1_r16_c1_k33` | none | 0.05× | 0.48× |
| `metal/f32/one_hot/batched/b1_r16_c3_k2` | none | 0.05× | 0.49× |
| `metal/f32/one_hot/batched/b1_r16_c3_k33` | none | 0.07× | 0.44× |
| `metal/f32/one_hot/batched/b1_r16_c8_k2` | none | 0.07× | 0.49× |
| `metal/f32/one_hot/batched/b1_r16_c8_k33` | none | 0.11× | 0.34× |
| `metal/f32/one_hot/batched/b1_r1_c17_k2` | none | 0.03× | 0.52× |
| `metal/f32/one_hot/batched/b1_r1_c17_k33` | none | 0.04× | 0.47× |
| `metal/f32/one_hot/batched/b1_r1_c1_k33` | none | 0.03× | 0.58× |
| `metal/f32/one_hot/batched/b1_r1_c3_k2` | none | 0.03× | 0.49× |
| `metal/f32/one_hot/batched/b1_r1_c3_k33` | none | 0.03× | 0.55× |
| `metal/f32/one_hot/batched/b1_r1_c8_k2` | none | 0.03× | 0.44× |
| `metal/f32/one_hot/batched/b1_r1_c8_k33` | none | 0.04× | 0.52× |
| `metal/f32/one_hot/batched/b1_r3_c17_k2` | none | 0.05× | 0.53× |
| `metal/f32/one_hot/batched/b1_r3_c17_k33` | none | 0.08× | 0.40× |
| `metal/f32/one_hot/batched/b1_r3_c1_k2` | none | 0.03× | 0.53× |
| `metal/f32/one_hot/batched/b1_r3_c1_k33` | none | 0.03× | 0.49× |
| `metal/f32/one_hot/batched/b1_r3_c3_k2` | none | 0.03× | 0.49× |
| `metal/f32/one_hot/batched/b1_r3_c3_k33` | none | 0.04× | 0.50× |
| `metal/f32/one_hot/batched/b1_r3_c8_k2` | none | 0.04× | 0.61× |
| `metal/f32/one_hot/batched/b1_r3_c8_k33` | none | 0.05× | 0.46× |
| `metal/f32/one_hot/batched/b1_r7_c17_k2` | none | 0.07× | 0.47× |
| `metal/f32/one_hot/batched/b1_r7_c17_k33` | none | 0.12× | 0.37× |
| `metal/f32/one_hot/batched/b1_r7_c1_k2` | none | 0.03× | 0.53× |
| `metal/f32/one_hot/batched/b1_r7_c1_k33` | none | 0.04× | 0.47× |
| `metal/f32/one_hot/batched/b1_r7_c3_k2` | none | 0.03× | 0.48× |
| `metal/f32/one_hot/batched/b1_r7_c3_k33` | none | 0.05× | 0.47× |
| `metal/f32/one_hot/batched/b1_r7_c8_k2` | none | 0.05× | 0.51× |
| `metal/f32/one_hot/batched/b1_r7_c8_k33` | none | 0.08× | 0.42× |
| `metal/f32/one_hot/batched/b2_r16_c17_k2` | none | 0.11× | 0.41× |
| `metal/f32/one_hot/batched/b2_r16_c17_k33` | none | 0.12× | 0.19× |
| `metal/f32/one_hot/batched/b2_r16_c1_k2` | none | 0.04× | 0.47× |
| `metal/f32/one_hot/batched/b2_r16_c1_k33` | none | 0.06× | 0.42× |
| `metal/f32/one_hot/batched/b2_r16_c3_k2` | none | 0.07× | 0.46× |
| `metal/f32/one_hot/batched/b2_r16_c3_k33` | none | 0.09× | 0.42× |
| `metal/f32/one_hot/batched/b2_r16_c8_k2` | none | 0.10× | 0.42× |
| `metal/f32/one_hot/batched/b2_r16_c8_k33` | none | 0.14× | 0.24× |
| `metal/f32/one_hot/batched/b2_r1_c17_k2` | none | 0.04× | 0.51× |
| `metal/f32/one_hot/batched/b2_r1_c17_k33` | none | 0.06× | 0.39× |
| `metal/f32/one_hot/batched/b2_r1_c1_k2` | none | 0.03× | 0.46× |
| `metal/f32/one_hot/batched/b2_r1_c1_k33` | none | 0.03× | 0.58× |
| `metal/f32/one_hot/batched/b2_r1_c3_k2` | none | 0.03× | 0.58× |
| `metal/f32/one_hot/batched/b2_r1_c3_k33` | none | 0.04× | 0.50× |
| `metal/f32/one_hot/batched/b2_r1_c8_k2` | none | 0.03× | 0.54× |
| `metal/f32/one_hot/batched/b2_r1_c8_k33` | none | 0.04× | 0.48× |
| `metal/f32/one_hot/batched/b2_r3_c17_k2` | none | 0.06× | 0.51× |
| `metal/f32/one_hot/batched/b2_r3_c17_k33` | none | 0.11× | 0.38× |
| `metal/f32/one_hot/batched/b2_r3_c1_k2` | none | 0.03× | 0.48× |
| `metal/f32/one_hot/batched/b2_r3_c1_k33` | none | 0.03× | 0.54× |
| `metal/f32/one_hot/batched/b2_r3_c3_k2` | none | 0.03× | 0.51× |
| `metal/f32/one_hot/batched/b2_r3_c3_k33` | none | 0.06× | 0.49× |
| `metal/f32/one_hot/batched/b2_r3_c8_k2` | none | 0.04× | 0.49× |
| `metal/f32/one_hot/batched/b2_r3_c8_k33` | none | 0.08× | 0.51× |
| `metal/f32/one_hot/batched/b2_r7_c17_k2` | none | 0.10× | 0.42× |
| `metal/f32/one_hot/batched/b2_r7_c17_k33` | none | 0.14× | 0.32× |
| `metal/f32/one_hot/batched/b2_r7_c1_k2` | none | 0.03× | 0.56× |
| `metal/f32/one_hot/batched/b2_r7_c1_k33` | none | 0.05× | 0.48× |
| `metal/f32/one_hot/batched/b2_r7_c3_k2` | none | 0.06× | 0.49× |
| `metal/f32/one_hot/batched/b2_r7_c3_k33` | none | 0.07× | 0.39× |
| `metal/f32/one_hot/batched/b2_r7_c8_k2` | none | 0.07× | 0.47× |
| `metal/f32/one_hot/batched/b2_r7_c8_k33` | none | 0.11× | 0.39× |
| `metal/f32/one_hot/batched/b4_r16_c17_k2` | none | 0.19× | 0.26× |
| `metal/f32/one_hot/batched/b4_r16_c17_k33` | none | 0.14× | 0.19× |
| `metal/f32/one_hot/batched/b4_r16_c1_k2` | none | 0.07× | 0.44× |
| `metal/f32/one_hot/batched/b4_r16_c1_k33` | none | 0.10× | 0.37× |
| `metal/f32/one_hot/batched/b4_r16_c3_k2` | none | 0.11× | 0.40× |
| `metal/f32/one_hot/batched/b4_r16_c3_k33` | none | 0.13× | 0.21× |
| `metal/f32/one_hot/batched/b4_r16_c8_k2` | none | 0.14× | 0.29× |
| `metal/f32/one_hot/batched/b4_r16_c8_k33` | none | 0.13× | 0.20× |
| `metal/f32/one_hot/batched/b4_r1_c17_k2` | none | 0.05× | 0.49× |
| `metal/f32/one_hot/batched/b4_r1_c17_k33` | none | 0.08× | 0.37× |
| `metal/f32/one_hot/batched/b4_r1_c1_k2` | none | 0.03× | 0.62× |
| `metal/f32/one_hot/batched/b4_r1_c1_k33` | none | 0.03× | 0.49× |
| `metal/f32/one_hot/batched/b4_r1_c3_k2` | none | 0.03× | 0.58× |
| `metal/f32/one_hot/batched/b4_r1_c3_k33` | none | 0.05× | 0.43× |
| `metal/f32/one_hot/batched/b4_r1_c8_k2` | none | 0.04× | 0.52× |
| `metal/f32/one_hot/batched/b4_r1_c8_k33` | none | 0.07× | 0.42× |
| `metal/f32/one_hot/batched/b4_r3_c17_k2` | none | 0.08× | 0.47× |
| `metal/f32/one_hot/batched/b4_r3_c17_k33` | none | 0.16× | 0.30× |
| `metal/f32/one_hot/batched/b4_r3_c1_k2` | none | 0.03× | 0.52× |
| `metal/f32/one_hot/batched/b4_r3_c1_k33` | none | 0.05× | 0.54× |
| `metal/f32/one_hot/batched/b4_r3_c3_k2` | none | 0.04× | 0.52× |
| `metal/f32/one_hot/batched/b4_r3_c3_k33` | none | 0.07× | 0.52× |
| `metal/f32/one_hot/batched/b4_r3_c8_k2` | none | 0.06× | 0.53× |
| `metal/f32/one_hot/batched/b4_r3_c8_k33` | none | 0.12× | 0.45× |
| `metal/f32/one_hot/batched/b4_r7_c17_k2` | none | 0.14× | 0.32× |
| `metal/f32/one_hot/batched/b4_r7_c17_k33` | none | 0.13× | 0.20× |
| `metal/f32/one_hot/batched/b4_r7_c1_k2` | none | 0.05× | 0.49× |
| `metal/f32/one_hot/batched/b4_r7_c1_k33` | none | 0.06× | 0.40× |
| `metal/f32/one_hot/batched/b4_r7_c3_k2` | none | 0.06× | 0.50× |
| `metal/f32/one_hot/batched/b4_r7_c3_k33` | none | 0.14× | 0.33× |
| `metal/f32/one_hot/batched/b4_r7_c8_k2` | none | 0.11× | 0.37× |
| `metal/f32/one_hot/batched/b4_r7_c8_k33` | none | 0.15× | 0.26× |
| `metal/f32/one_hot/batched/b8_r16_c17_k2` | none | 0.18× | 0.23× |
| `metal/f32/one_hot/batched/b8_r16_c17_k33` | none | 0.14× | 0.16× |
| `metal/f32/one_hot/batched/b8_r16_c1_k2` | none | 0.09× | 0.40× |
| `metal/f32/one_hot/batched/b8_r16_c1_k33` | none | 0.12× | 0.28× |
| `metal/f32/one_hot/batched/b8_r16_c3_k2` | none | 0.13× | 0.37× |
| `metal/f32/one_hot/batched/b8_r16_c3_k33` | none | 0.13× | 0.21× |
| `metal/f32/one_hot/batched/b8_r16_c8_k2` | none | 0.21× | 0.33× |
| `metal/f32/one_hot/batched/b8_r16_c8_k33` | none | 0.14× | 0.17× |
| `metal/f32/one_hot/batched/b8_r1_c17_k2` | none | 0.07× | 0.45× |
| `metal/f32/one_hot/batched/b8_r1_c17_k33` | none | 0.12× | 0.31× |
| `metal/f32/one_hot/batched/b8_r1_c1_k2` | none | 0.03× | 0.51× |
| `metal/f32/one_hot/batched/b8_r1_c1_k33` | none | 0.04× | 0.46× |
| `metal/f32/one_hot/batched/b8_r1_c3_k2` | none | 0.04× | 0.46× |
| `metal/f32/one_hot/batched/b8_r1_c3_k33` | none | 0.06× | 0.42× |
| `metal/f32/one_hot/batched/b8_r1_c8_k2` | none | 0.06× | 0.45× |
| `metal/f32/one_hot/batched/b8_r1_c8_k33` | none | 0.10× | 0.39× |
| `metal/f32/one_hot/batched/b8_r3_c17_k2` | none | 0.11× | 0.35× |
| `metal/f32/one_hot/batched/b8_r3_c17_k33` | none | 0.13× | 0.21× |
| `metal/f32/one_hot/batched/b8_r3_c1_k2` | none | 0.04× | 0.47× |
| `metal/f32/one_hot/batched/b8_r3_c1_k33` | none | 0.06× | 0.41× |
| `metal/f32/one_hot/batched/b8_r3_c3_k2` | none | 0.06× | 0.49× |
| `metal/f32/one_hot/batched/b8_r3_c3_k33` | none | 0.10× | 0.28× |
| `metal/f32/one_hot/batched/b8_r3_c8_k2` | none | 0.10× | 0.41× |
| `metal/f32/one_hot/batched/b8_r3_c8_k33` | none | 0.11× | 0.26× |
| `metal/f32/one_hot/batched/b8_r7_c17_k2` | none | 0.15× | 0.33× |
| `metal/f32/one_hot/batched/b8_r7_c17_k33` | none | 0.14× | 0.18× |
| `metal/f32/one_hot/batched/b8_r7_c1_k2` | none | 0.06× | 0.43× |
| `metal/f32/one_hot/batched/b8_r7_c1_k33` | none | 0.10× | 0.35× |
| `metal/f32/one_hot/batched/b8_r7_c3_k2` | none | 0.09× | 0.41× |
| `metal/f32/one_hot/batched/b8_r7_c3_k33` | none | 0.11× | 0.25× |
| `metal/f32/one_hot/batched/b8_r7_c8_k2` | none | 0.18× | 0.35× |
| `metal/f32/one_hot/batched/b8_r7_c8_k33` | none | 0.13× | 0.19× |
