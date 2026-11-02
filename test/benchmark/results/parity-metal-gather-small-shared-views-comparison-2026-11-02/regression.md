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
| `metal/bf16/gather/large/offset_axis0_r16_c129` | none | 0.03× | 0.66× |
| `metal/bf16/gather/large/offset_axis0_r16_c17` | none | 0.01× | 0.53× |
| `metal/bf16/gather/large/offset_axis0_r16_c3` | none | 0.01× | 0.50× |
| `metal/bf16/gather/large/offset_axis0_r16_c64` | none | 0.04× | 0.59× |
| `metal/bf16/gather/large/offset_axis0_r33_c129` | none | 0.05× | 0.54× |
| `metal/bf16/gather/large/offset_axis0_r33_c17` | none | 0.02× | 0.64× |
| `metal/bf16/gather/large/offset_axis0_r33_c3` | none | 0.02× | 0.65× |
| `metal/bf16/gather/large/offset_axis0_r33_c64` | none | 0.02× | 0.69× |
| `metal/bf16/gather/large/offset_axis0_r3_c129` | none | 0.02× | 0.60× |
| `metal/bf16/gather/large/offset_axis0_r3_c17` | none | 0.03× | 0.61× |
| `metal/bf16/gather/large/offset_axis0_r3_c3` | none | 0.02× | 0.63× |
| `metal/bf16/gather/large/offset_axis0_r3_c64` | none | 0.03× | 0.50× |
| `metal/bf16/gather/large/offset_axis0_r7_c129` | none | 0.03× | 0.60× |
| `metal/bf16/gather/large/offset_axis0_r7_c17` | none | 0.02× | 0.53× |
| `metal/bf16/gather/large/offset_axis0_r7_c3` | none | 0.03× | 0.59× |
| `metal/bf16/gather/large/offset_axis0_r7_c64` | none | 0.01× | 0.59× |
| `metal/bf16/gather/large/offset_axis1_r16_c129` | none | 0.05× | 0.62× |
| `metal/bf16/gather/large/offset_axis1_r16_c17` | none | 0.03× | 0.60× |
| `metal/bf16/gather/large/offset_axis1_r16_c3` | none | 0.02× | 0.66× |
| `metal/bf16/gather/large/offset_axis1_r16_c64` | none | 0.03× | 0.59× |
| `metal/bf16/gather/large/offset_axis1_r33_c129` | none | 0.10× | 0.62× |
| `metal/bf16/gather/large/offset_axis1_r33_c17` | none | 0.04× | 0.65× |
| `metal/bf16/gather/large/offset_axis1_r33_c3` | none | 0.02× | 0.65× |
| `metal/bf16/gather/large/offset_axis1_r33_c64` | none | 0.06× | 0.67× |
| `metal/bf16/gather/large/offset_axis1_r3_c129` | none | 0.01× | 0.54× |
| `metal/bf16/gather/large/offset_axis1_r3_c17` | none | 0.02× | 0.64× |
| `metal/bf16/gather/large/offset_axis1_r3_c3` | none | 0.01× | 0.51× |
| `metal/bf16/gather/large/offset_axis1_r3_c64` | none | 0.01× | 0.59× |
| `metal/bf16/gather/large/offset_axis1_r7_c129` | none | 0.04× | 0.66× |
| `metal/bf16/gather/large/offset_axis1_r7_c17` | none | 0.01× | 0.53× |
| `metal/bf16/gather/large/offset_axis1_r7_c3` | none | 0.01× | 0.57× |
| `metal/bf16/gather/large/offset_axis1_r7_c64` | none | 0.01× | 0.59× |
| `metal/bf16/gather/strided/axis0_r16_c129` | none | 0.06× | 0.66× |
| `metal/bf16/gather/strided/axis0_r16_c17` | none | 0.03× | 0.47× |
| `metal/bf16/gather/strided/axis0_r16_c3` | none | 0.03× | 0.52× |
| `metal/bf16/gather/strided/axis0_r16_c64` | none | 0.05× | 0.62× |
| `metal/bf16/gather/strided/axis0_r33_c129` | none | 0.10× | 0.57× |
| `metal/bf16/gather/strided/axis0_r33_c17` | none | 0.04× | 0.57× |
| `metal/bf16/gather/strided/axis0_r33_c3` | none | 0.02× | 0.63× |
| `metal/bf16/gather/strided/axis0_r33_c64` | none | 0.06× | 0.68× |
| `metal/bf16/gather/strided/axis0_r3_c129` | none | 0.03× | 0.63× |
| `metal/bf16/gather/strided/axis0_r3_c17` | none | 0.02× | 0.53× |
| `metal/bf16/gather/strided/axis0_r3_c3` | none | 0.02× | 0.59× |
| `metal/bf16/gather/strided/axis0_r3_c64` | none | 0.03× | 0.56× |
| `metal/bf16/gather/strided/axis0_r7_c129` | none | 0.04× | 0.59× |
| `metal/bf16/gather/strided/axis0_r7_c17` | none | 0.02× | 0.66× |
| `metal/bf16/gather/strided/axis0_r7_c3` | none | 0.02× | 0.60× |
| `metal/bf16/gather/strided/axis0_r7_c64` | none | 0.03× | 0.62× |
| `metal/bf16/gather/strided/axis1_r16_c129` | none | 0.06× | 0.62× |
| `metal/bf16/gather/strided/axis1_r16_c17` | none | 0.04× | 0.62× |
| `metal/bf16/gather/strided/axis1_r16_c3` | none | 0.03× | 0.63× |
| `metal/bf16/gather/strided/axis1_r16_c64` | none | 0.04× | 0.67× |
| `metal/bf16/gather/strided/axis1_r33_c129` | none | 0.10× | 0.67× |
| `metal/bf16/gather/strided/axis1_r33_c17` | none | 0.05× | 0.68× |
| `metal/bf16/gather/strided/axis1_r33_c3` | none | 0.03× | 0.62× |
| `metal/bf16/gather/strided/axis1_r33_c64` | none | 0.08× | 0.53× |
| `metal/bf16/gather/strided/axis1_r3_c129` | none | 0.02× | 0.59× |
| `metal/bf16/gather/strided/axis1_r3_c17` | none | 0.02× | 0.57× |
| `metal/bf16/gather/strided/axis1_r3_c3` | none | 0.02× | 0.65× |
| `metal/bf16/gather/strided/axis1_r3_c64` | none | 0.03× | 0.60× |
| `metal/bf16/gather/strided/axis1_r7_c129` | none | 0.03× | 0.52× |
| `metal/bf16/gather/strided/axis1_r7_c17` | none | 0.03× | 0.64× |
| `metal/bf16/gather/strided/axis1_r7_c3` | none | 0.03× | 0.57× |
| `metal/bf16/gather/strided/axis1_r7_c64` | none | 0.04× | 0.49× |
| `metal/f32/gather/large/offset_axis0_r16_c129` | none | 0.06× | 0.55× |
| `metal/f32/gather/large/offset_axis0_r16_c17` | none | 0.03× | 0.54× |
| `metal/f32/gather/large/offset_axis0_r16_c3` | none | 0.03× | 0.56× |
| `metal/f32/gather/large/offset_axis0_r16_c64` | none | 0.05× | 0.56× |
| `metal/f32/gather/large/offset_axis0_r33_c129` | none | 0.09× | 0.50× |
| `metal/f32/gather/large/offset_axis0_r33_c17` | none | 0.04× | 0.57× |
| `metal/f32/gather/large/offset_axis0_r33_c3` | none | 0.03× | 0.57× |
| `metal/f32/gather/large/offset_axis0_r33_c64` | none | 0.06× | 0.48× |
| `metal/f32/gather/large/offset_axis0_r3_c129` | none | 0.02× | 0.49× |
| `metal/f32/gather/large/offset_axis0_r3_c17` | none | 0.03× | 0.51× |
| `metal/f32/gather/large/offset_axis0_r3_c3` | none | 0.02× | 0.50× |
| `metal/f32/gather/large/offset_axis0_r3_c64` | none | 0.02× | 0.58× |
| `metal/f32/gather/large/offset_axis0_r7_c129` | none | 0.04× | 0.54× |
| `metal/f32/gather/large/offset_axis0_r7_c17` | none | 0.02× | 0.50× |
| `metal/f32/gather/large/offset_axis0_r7_c3` | none | 0.02× | 0.55× |
| `metal/f32/gather/large/offset_axis0_r7_c64` | none | 0.02× | 0.56× |
| `metal/f32/gather/large/offset_axis1_r16_c129` | none | 0.06× | 0.57× |
| `metal/f32/gather/large/offset_axis1_r16_c17` | none | 0.03× | 0.45× |
| `metal/f32/gather/large/offset_axis1_r16_c3` | none | 0.02× | 0.52× |
| `metal/f32/gather/large/offset_axis1_r16_c64` | none | 0.04× | 0.50× |
| `metal/f32/gather/large/offset_axis1_r33_c129` | none | 0.08× | 0.57× |
| `metal/f32/gather/large/offset_axis1_r33_c17` | none | 0.05× | 0.52× |
| `metal/f32/gather/large/offset_axis1_r33_c3` | none | 0.03× | 0.52× |
| `metal/f32/gather/large/offset_axis1_r33_c64` | none | 0.07× | 0.58× |
| `metal/f32/gather/large/offset_axis1_r3_c129` | none | 0.02× | 0.56× |
| `metal/f32/gather/large/offset_axis1_r3_c17` | none | 0.03× | 0.58× |
| `metal/f32/gather/large/offset_axis1_r3_c3` | none | 0.02× | 0.54× |
| `metal/f32/gather/large/offset_axis1_r3_c64` | none | 0.02× | 0.44× |
| `metal/f32/gather/large/offset_axis1_r7_c129` | none | 0.04× | 0.53× |
| `metal/f32/gather/large/offset_axis1_r7_c17` | none | 0.03× | 0.56× |
| `metal/f32/gather/large/offset_axis1_r7_c3` | none | 0.02× | 0.54× |
| `metal/f32/gather/large/offset_axis1_r7_c64` | none | 0.04× | 0.54× |
| `metal/f32/gather/strided/axis0_r16_c129` | none | 0.05× | 0.49× |
| `metal/f32/gather/strided/axis0_r16_c17` | none | 0.03× | 0.56× |
| `metal/f32/gather/strided/axis0_r16_c3` | none | 0.03× | 0.46× |
| `metal/f32/gather/strided/axis0_r16_c64` | none | 0.04× | 0.55× |
| `metal/f32/gather/strided/axis0_r33_c129` | none | 0.09× | 0.62× |
| `metal/f32/gather/strided/axis0_r33_c17` | none | 0.05× | 0.45× |
| `metal/f32/gather/strided/axis0_r33_c3` | none | 0.03× | 0.54× |
| `metal/f32/gather/strided/axis0_r33_c64` | none | 0.07× | 0.50× |
| `metal/f32/gather/strided/axis0_r3_c129` | none | 0.02× | 0.52× |
| `metal/f32/gather/strided/axis0_r3_c17` | none | 0.02× | 0.63× |
| `metal/f32/gather/strided/axis0_r3_c3` | none | 0.02× | 0.49× |
| `metal/f32/gather/strided/axis0_r3_c64` | none | 0.02× | 0.48× |
| `metal/f32/gather/strided/axis0_r7_c129` | none | 0.04× | 0.39× |
| `metal/f32/gather/strided/axis0_r7_c17` | none | 0.02× | 0.21× |
| `metal/f32/gather/strided/axis0_r7_c3` | none | 0.02× | 0.34× |
| `metal/f32/gather/strided/axis0_r7_c64` | none | 0.03× | 0.39× |
| `metal/f32/gather/strided/axis1_r16_c129` | none | 0.05× | 0.57× |
| `metal/f32/gather/strided/axis1_r16_c17` | none | 0.03× | 0.58× |
| `metal/f32/gather/strided/axis1_r16_c3` | none | 0.02× | 0.55× |
| `metal/f32/gather/strided/axis1_r16_c64` | none | 0.05× | 0.54× |
| `metal/f32/gather/strided/axis1_r33_c129` | none | 0.10× | 0.58× |
| `metal/f32/gather/strided/axis1_r33_c17` | none | 0.05× | 0.55× |
| `metal/f32/gather/strided/axis1_r33_c3` | none | 0.03× | 0.56× |
| `metal/f32/gather/strided/axis1_r33_c64` | none | 0.07× | 0.54× |
| `metal/f32/gather/strided/axis1_r3_c129` | none | 0.03× | 0.56× |
| `metal/f32/gather/strided/axis1_r3_c17` | none | 0.02× | 0.56× |
| `metal/f32/gather/strided/axis1_r3_c3` | none | 0.02× | 0.48× |
| `metal/f32/gather/strided/axis1_r3_c64` | none | 0.02× | 0.51× |
| `metal/f32/gather/strided/axis1_r7_c129` | none | 0.04× | 0.42× |
| `metal/f32/gather/strided/axis1_r7_c17` | none | 0.03× | 0.54× |
| `metal/f32/gather/strided/axis1_r7_c3` | none | 0.02× | 0.54× |
| `metal/f32/gather/strided/axis1_r7_c64` | none | 0.03× | 0.54× |
