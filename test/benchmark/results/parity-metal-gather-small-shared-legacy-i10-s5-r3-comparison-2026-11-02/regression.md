# Parity regression check

Target: `Mac16,1/26.6.2/arm64`

- Cells compared: 64
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
| `metal/bf16/gather/batched` | none | 0.03× | 0.68× |
| `metal/bf16/gather/large` | none | 0.10× | 0.54× |
| `metal/bf16/gather/large/a0_r64_w64_o32` | none | 0.10× | 0.56× |
| `metal/bf16/gather/large/a1_r64_w64_o32` | none | 0.12× | 0.73× |
| `metal/bf16/gather/large/axis0_32x64` | none | 0.10× | 0.65× |
| `metal/bf16/gather/large/axis1_16x128` | none | 0.09× | 0.52× |
| `metal/bf16/gather/large/axis1_64x64` | none | 0.16× | 0.44× |
| `metal/bf16/gather/large/r128_c127_i64` | none | 0.96× | 0.98× |
| `metal/bf16/gather/large/r128_w64_i128x32_a1_offset` | none | 0.21× | 0.69× |
| `metal/bf16/gather/large/r128_w64_i128x32_a1_strided` | none | 0.23× | 0.71× |
| `metal/bf16/gather/large/r64_w33_i32x33_a0_offset` | none | 0.07× | 0.62× |
| `metal/bf16/gather/large/r64_w33_i32x33_a0_strided` | none | 0.09× | 0.67× |
| `metal/bf16/gather/small` | none | 0.02× | 0.53× |
| `metal/bf16/gather/small/a0_r31_w32_o16` | none | 0.04× | 0.61× |
| `metal/bf16/gather/small/a0_r32_w32_o32` | none | 0.08× | 0.62× |
| `metal/bf16/gather/small/a0_r33_w32_o33` | none | 0.07× | 0.66× |
| `metal/bf16/gather/small/a0_r7_w32_o1` | none | 0.02× | 0.55× |
| `metal/bf16/gather/small/a1_r32_w31_o16` | none | 0.05× | 0.65× |
| `metal/bf16/gather/small/a1_r32_w32_o32` | none | 0.07× | 0.67× |
| `metal/bf16/gather/small/a1_r32_w33_o33` | none | 0.07× | 0.66× |
| `metal/bf16/gather/small/a1_r32_w7_o1` | none | 0.03× | 0.58× |
| `metal/bf16/gather/small/axis0_4x8` | none | 0.03× | 0.62× |
| `metal/bf16/gather/small/axis0_8x16` | none | 0.03× | 0.62× |
| `metal/bf16/gather/small/axis1_8x16` | none | 0.03× | 0.57× |
| `metal/bf16/gather/small/r16_w31_i8x31_a0_offset` | none | 0.04× | 0.66× |
| `metal/bf16/gather/small/r16_w31_i8x31_a0_strided` | none | 0.03× | 0.63× |
| `metal/bf16/gather/small/r32_w32_i16x32_a0_offset` | none | 0.05× | 0.53× |
| `metal/bf16/gather/small/r32_w32_i16x32_a0_strided` | none | 0.04× | 0.62× |
| `metal/bf16/gather/small/r32_w64_i32x31_a1_offset` | none | 0.08× | 0.63× |
| `metal/bf16/gather/small/r32_w64_i32x31_a1_strided` | none | 0.08× | 0.64× |
| `metal/bf16/gather/strided` | none | 0.03× | 0.56× |
| `metal/bf16/gather/strided/r64_c127_i32` | none | 0.13× | 0.67× |
| `metal/f32/gather/batched` | none | 0.03× | 0.48× |
| `metal/f32/gather/large` | none | 0.11× | 0.51× |
| `metal/f32/gather/large/a0_r64_w64_o32` | none | 0.10× | 0.58× |
| `metal/f32/gather/large/a1_r64_w64_o32` | none | 0.12× | 0.57× |
| `metal/f32/gather/large/axis0_32x64` | none | 0.10× | 0.60× |
| `metal/f32/gather/large/axis1_16x128` | none | 0.08× | 0.57× |
| `metal/f32/gather/large/axis1_64x64` | none | 0.17× | 0.64× |
| `metal/f32/gather/large/r128_c127_i64` | none | 0.99× | 1.01× |
| `metal/f32/gather/large/r128_w64_i128x32_a1_offset` | none | 0.23× | 0.63× |
| `metal/f32/gather/large/r128_w64_i128x32_a1_strided` | none | 0.23× | 0.66× |
| `metal/f32/gather/large/r64_w33_i32x33_a0_offset` | none | 0.08× | 0.56× |
| `metal/f32/gather/large/r64_w33_i32x33_a0_strided` | none | 0.09× | 0.53× |
| `metal/f32/gather/small` | none | 0.03× | 0.51× |
| `metal/f32/gather/small/a0_r31_w32_o16` | none | 0.05× | 0.58× |
| `metal/f32/gather/small/a0_r32_w32_o32` | none | 0.08× | 0.57× |
| `metal/f32/gather/small/a0_r33_w32_o33` | none | 0.07× | 0.53× |
| `metal/f32/gather/small/a0_r7_w32_o1` | none | 0.02× | 0.49× |
| `metal/f32/gather/small/a1_r32_w31_o16` | none | 0.07× | 0.54× |
| `metal/f32/gather/small/a1_r32_w32_o32` | none | 0.07× | 0.52× |
| `metal/f32/gather/small/a1_r32_w33_o33` | none | 0.08× | 0.51× |
| `metal/f32/gather/small/a1_r32_w7_o1` | none | 0.02× | 0.50× |
| `metal/f32/gather/small/axis0_4x8` | none | 0.03× | 0.47× |
| `metal/f32/gather/small/axis0_8x16` | none | 0.03× | 0.52× |
| `metal/f32/gather/small/axis1_8x16` | none | 0.03× | 0.49× |
| `metal/f32/gather/small/r16_w31_i8x31_a0_offset` | none | 0.03× | 0.51× |
| `metal/f32/gather/small/r16_w31_i8x31_a0_strided` | none | 0.03× | 0.47× |
| `metal/f32/gather/small/r32_w32_i16x32_a0_offset` | none | 0.05× | 0.51× |
| `metal/f32/gather/small/r32_w32_i16x32_a0_strided` | none | 0.05× | 0.50× |
| `metal/f32/gather/small/r32_w64_i32x31_a1_offset` | none | 0.07× | 0.55× |
| `metal/f32/gather/small/r32_w64_i32x31_a1_strided` | none | 0.08× | 0.52× |
| `metal/f32/gather/strided` | none | 0.02× | 0.51× |
| `metal/f32/gather/strided/r64_c127_i32` | none | 0.14× | 0.63× |
