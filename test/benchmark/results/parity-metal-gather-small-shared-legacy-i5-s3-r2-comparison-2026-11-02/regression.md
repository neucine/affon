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
| `metal/bf16/gather/large/offset_axis0_r2_c127` | none | 0.02× | 0.41× |
| `metal/bf16/gather/large/offset_axis0_r2_c31` | none | 0.02× | 0.48× |
| `metal/bf16/gather/large/offset_axis0_r2_c63` | none | 0.02× | 0.49× |
| `metal/bf16/gather/large/offset_axis0_r2_c7` | none | 0.02× | 0.44× |
| `metal/bf16/gather/large/offset_axis0_r32_c127` | none | 0.07× | 0.63× |
| `metal/bf16/gather/large/offset_axis0_r32_c31` | none | 0.04× | 0.44× |
| `metal/bf16/gather/large/offset_axis0_r32_c63` | none | 0.05× | 0.51× |
| `metal/bf16/gather/large/offset_axis0_r32_c7` | none | 0.03× | 0.50× |
| `metal/bf16/gather/large/offset_axis0_r65_c127` | none | 0.98× | 1.15× |
| `metal/bf16/gather/large/offset_axis0_r65_c31` | none | 0.05× | 0.52× |
| `metal/bf16/gather/large/offset_axis0_r65_c63` | none | 0.07× | 0.56× |
| `metal/bf16/gather/large/offset_axis0_r65_c7` | none | 0.03× | 0.45× |
| `metal/bf16/gather/large/offset_axis0_r8_c127` | none | 0.03× | 0.43× |
| `metal/bf16/gather/large/offset_axis0_r8_c31` | none | 0.03× | 0.44× |
| `metal/bf16/gather/large/offset_axis0_r8_c63` | none | 0.04× | 0.55× |
| `metal/bf16/gather/large/offset_axis0_r8_c7` | none | 0.02× | 0.37× |
| `metal/bf16/gather/large/offset_axis1_r2_c127` | none | 0.02× | 0.50× |
| `metal/bf16/gather/large/offset_axis1_r2_c31` | none | 0.02× | 0.53× |
| `metal/bf16/gather/large/offset_axis1_r2_c63` | none | 0.02× | 0.46× |
| `metal/bf16/gather/large/offset_axis1_r2_c7` | none | 0.02× | 0.50× |
| `metal/bf16/gather/large/offset_axis1_r32_c127` | none | 0.09× | 0.48× |
| `metal/bf16/gather/large/offset_axis1_r32_c31` | none | 0.04× | 0.54× |
| `metal/bf16/gather/large/offset_axis1_r32_c63` | none | 0.05× | 0.50× |
| `metal/bf16/gather/large/offset_axis1_r32_c7` | none | 0.03× | 0.59× |
| `metal/bf16/gather/large/offset_axis1_r65_c127` | none | 0.83× | 1.33× |
| `metal/bf16/gather/large/offset_axis1_r65_c31` | none | 0.07× | 0.53× |
| `metal/bf16/gather/large/offset_axis1_r65_c63` | none | 0.11× | 0.58× |
| `metal/bf16/gather/large/offset_axis1_r65_c7` | none | 0.05× | 0.58× |
| `metal/bf16/gather/large/offset_axis1_r8_c127` | none | 0.04× | 0.53× |
| `metal/bf16/gather/large/offset_axis1_r8_c31` | none | 0.02× | 0.48× |
| `metal/bf16/gather/large/offset_axis1_r8_c63` | none | 0.03× | 0.62× |
| `metal/bf16/gather/large/offset_axis1_r8_c7` | none | 0.02× | 0.57× |
| `metal/bf16/gather/strided/axis0_r2_c127` | none | 0.02× | 0.55× |
| `metal/bf16/gather/strided/axis0_r2_c31` | none | 0.01× | 0.49× |
| `metal/bf16/gather/strided/axis0_r2_c63` | none | 0.02× | 0.52× |
| `metal/bf16/gather/strided/axis0_r2_c7` | none | 0.02× | 0.66× |
| `metal/bf16/gather/strided/axis0_r32_c127` | none | 0.07× | 0.44× |
| `metal/bf16/gather/strided/axis0_r32_c31` | none | 0.03× | 0.56× |
| `metal/bf16/gather/strided/axis0_r32_c63` | none | 0.05× | 0.60× |
| `metal/bf16/gather/strided/axis0_r32_c7` | none | 0.03× | 0.56× |
| `metal/bf16/gather/strided/axis0_r65_c127` | none | 1.15× | 1.00× |
| `metal/bf16/gather/strided/axis0_r65_c31` | none | 0.08× | 0.48× |
| `metal/bf16/gather/strided/axis0_r65_c63` | none | 0.09× | 0.54× |
| `metal/bf16/gather/strided/axis0_r65_c7` | none | 0.04× | 0.35× |
| `metal/bf16/gather/strided/axis0_r8_c127` | none | 0.03× | 0.49× |
| `metal/bf16/gather/strided/axis0_r8_c31` | none | 0.02× | 0.43× |
| `metal/bf16/gather/strided/axis0_r8_c63` | none | 0.02× | 0.41× |
| `metal/bf16/gather/strided/axis0_r8_c7` | none | 0.02× | 0.41× |
| `metal/bf16/gather/strided/axis1_r2_c127` | none | 0.02× | 0.63× |
| `metal/bf16/gather/strided/axis1_r2_c31` | none | 0.02× | 0.49× |
| `metal/bf16/gather/strided/axis1_r2_c63` | none | 0.02× | 0.56× |
| `metal/bf16/gather/strided/axis1_r2_c7` | none | 0.02× | 0.44× |
| `metal/bf16/gather/strided/axis1_r32_c127` | none | 0.08× | 0.59× |
| `metal/bf16/gather/strided/axis1_r32_c31` | none | 0.05× | 0.66× |
| `metal/bf16/gather/strided/axis1_r32_c63` | none | 0.06× | 0.57× |
| `metal/bf16/gather/strided/axis1_r32_c7` | none | 0.03× | 0.59× |
| `metal/bf16/gather/strided/axis1_r65_c127` | none | 1.02× | 0.91× |
| `metal/bf16/gather/strided/axis1_r65_c31` | none | 0.07× | 0.51× |
| `metal/bf16/gather/strided/axis1_r65_c63` | none | 0.10× | 0.59× |
| `metal/bf16/gather/strided/axis1_r65_c7` | none | 0.03× | 0.75× |
| `metal/bf16/gather/strided/axis1_r8_c127` | none | 0.03× | 0.63× |
| `metal/bf16/gather/strided/axis1_r8_c31` | none | 0.02× | 0.39× |
| `metal/bf16/gather/strided/axis1_r8_c63` | none | 0.03× | 0.53× |
| `metal/bf16/gather/strided/axis1_r8_c7` | none | 0.02× | 0.61× |
| `metal/f32/gather/large/offset_axis0_r2_c127` | none | 0.03× | 0.44× |
| `metal/f32/gather/large/offset_axis0_r2_c31` | none | 0.02× | 0.34× |
| `metal/f32/gather/large/offset_axis0_r2_c63` | none | 0.02× | 0.45× |
| `metal/f32/gather/large/offset_axis0_r2_c7` | none | 0.02× | 0.43× |
| `metal/f32/gather/large/offset_axis0_r32_c127` | none | 0.06× | 0.41× |
| `metal/f32/gather/large/offset_axis0_r32_c31` | none | 0.05× | 0.40× |
| `metal/f32/gather/large/offset_axis0_r32_c63` | none | 0.05× | 0.34× |
| `metal/f32/gather/large/offset_axis0_r32_c7` | none | 0.03× | 0.43× |
| `metal/f32/gather/large/offset_axis0_r65_c127` | none | 0.93× | 0.73× |
| `metal/f32/gather/large/offset_axis0_r65_c31` | none | 0.07× | 0.55× |
| `metal/f32/gather/large/offset_axis0_r65_c63` | none | 0.08× | 0.46× |
| `metal/f32/gather/large/offset_axis0_r65_c7` | none | 0.04× | 0.47× |
| `metal/f32/gather/large/offset_axis0_r8_c127` | none | 0.03× | 0.48× |
| `metal/f32/gather/large/offset_axis0_r8_c31` | none | 0.02× | 0.41× |
| `metal/f32/gather/large/offset_axis0_r8_c63` | none | 0.03× | 0.41× |
| `metal/f32/gather/large/offset_axis0_r8_c7` | none | 0.03× | 0.66× |
| `metal/f32/gather/large/offset_axis1_r2_c127` | none | 0.02× | 0.38× |
| `metal/f32/gather/large/offset_axis1_r2_c31` | none | 0.02× | 0.47× |
| `metal/f32/gather/large/offset_axis1_r2_c63` | none | 0.02× | 0.47× |
| `metal/f32/gather/large/offset_axis1_r2_c7` | none | 0.02× | 0.35× |
| `metal/f32/gather/large/offset_axis1_r32_c127` | none | 0.08× | 0.52× |
| `metal/f32/gather/large/offset_axis1_r32_c31` | none | 0.03× | 0.43× |
| `metal/f32/gather/large/offset_axis1_r32_c63` | none | 0.06× | 0.64× |
| `metal/f32/gather/large/offset_axis1_r32_c7` | none | 0.03× | 0.43× |
| `metal/f32/gather/large/offset_axis1_r65_c127` | none | 0.81× | 1.03× |
| `metal/f32/gather/large/offset_axis1_r65_c31` | none | 0.09× | 0.50× |
| `metal/f32/gather/large/offset_axis1_r65_c63` | none | 0.10× | 0.42× |
| `metal/f32/gather/large/offset_axis1_r65_c7` | none | 0.06× | 0.61× |
| `metal/f32/gather/large/offset_axis1_r8_c127` | none | 0.03× | 0.41× |
| `metal/f32/gather/large/offset_axis1_r8_c31` | none | 0.02× | 0.38× |
| `metal/f32/gather/large/offset_axis1_r8_c63` | none | 0.04× | 0.47× |
| `metal/f32/gather/large/offset_axis1_r8_c7` | none | 0.02× | 0.44× |
| `metal/f32/gather/strided/axis0_r2_c127` | none | 0.02× | 0.45× |
| `metal/f32/gather/strided/axis0_r2_c31` | none | 0.02× | 0.47× |
| `metal/f32/gather/strided/axis0_r2_c63` | none | 0.02× | 0.32× |
| `metal/f32/gather/strided/axis0_r2_c7` | none | 0.02× | 0.50× |
| `metal/f32/gather/strided/axis0_r32_c127` | none | 0.06× | 0.60× |
| `metal/f32/gather/strided/axis0_r32_c31` | none | 0.04× | 0.38× |
| `metal/f32/gather/strided/axis0_r32_c63` | none | 0.05× | 0.59× |
| `metal/f32/gather/strided/axis0_r32_c7` | none | 0.03× | 0.46× |
| `metal/f32/gather/strided/axis0_r65_c127` | none | 0.99× | 1.04× |
| `metal/f32/gather/strided/axis0_r65_c31` | none | 0.06× | 0.46× |
| `metal/f32/gather/strided/axis0_r65_c63` | none | 0.09× | 0.54× |
| `metal/f32/gather/strided/axis0_r65_c7` | none | 0.04× | 0.61× |
| `metal/f32/gather/strided/axis0_r8_c127` | none | 0.04× | 0.31× |
| `metal/f32/gather/strided/axis0_r8_c31` | none | 0.02× | 0.56× |
| `metal/f32/gather/strided/axis0_r8_c63` | none | 0.03× | 0.48× |
| `metal/f32/gather/strided/axis0_r8_c7` | none | 0.02× | 0.45× |
| `metal/f32/gather/strided/axis1_r2_c127` | none | 0.03× | 0.43× |
| `metal/f32/gather/strided/axis1_r2_c31` | none | 0.02× | 0.44× |
| `metal/f32/gather/strided/axis1_r2_c63` | none | 0.02× | 0.54× |
| `metal/f32/gather/strided/axis1_r2_c7` | none | 0.02× | 0.43× |
| `metal/f32/gather/strided/axis1_r32_c127` | none | 0.08× | 0.43× |
| `metal/f32/gather/strided/axis1_r32_c31` | none | 0.05× | 0.59× |
| `metal/f32/gather/strided/axis1_r32_c63` | none | 0.05× | 0.41× |
| `metal/f32/gather/strided/axis1_r32_c7` | none | 0.04× | 0.47× |
| `metal/f32/gather/strided/axis1_r65_c127` | none | 0.86× | 1.02× |
| `metal/f32/gather/strided/axis1_r65_c31` | none | 0.08× | 0.57× |
| `metal/f32/gather/strided/axis1_r65_c63` | none | 0.10× | 0.52× |
| `metal/f32/gather/strided/axis1_r65_c7` | none | 0.04× | 0.43× |
| `metal/f32/gather/strided/axis1_r8_c127` | none | 0.04× | 0.56× |
| `metal/f32/gather/strided/axis1_r8_c31` | none | 0.03× | 0.45× |
| `metal/f32/gather/strided/axis1_r8_c63` | none | 0.03× | 0.49× |
| `metal/f32/gather/strided/axis1_r8_c7` | none | 0.03× | 0.48× |
