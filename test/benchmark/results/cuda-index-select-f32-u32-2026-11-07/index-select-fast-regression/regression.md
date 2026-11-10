# Parity regression check

Target: `NVIDIA GeForce RTX 3090/7.0.0-38-generic/x86_64`

- Cells compared: 18
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
| `cuda/f32/index_select/large/offset_axis0_r2_c7_i2` | none | 0.98× | 0.94× |
| `cuda/f32/index_select/large/offset_axis0_r8_c127_i5` | none | 0.98× | 1.00× |
| `cuda/f32/index_select/large/offset_axis1_r2_c127_i64` | none | 0.95× | 0.98× |
| `cuda/f32/index_select/large/offset_axis1_r3_c5_i3` | none | 1.02× | 0.99× |
| `cuda/f32/index_select/large/offset_axis1_r65_c127_i64` | none | 1.04× | 0.98× |
| `cuda/f32/index_select/large/r128_c127_i64` | none | 1.03× | 1.01× |
| `cuda/f32/index_select/small` | none | 0.96× | 0.98× |
| `cuda/f32/index_select/small/a0_r31_w32_i15` | none | 0.99× | 0.94× |
| `cuda/f32/index_select/small/a0_r33_w32_i17` | none | 1.02× | 0.99× |
| `cuda/f32/index_select/small/a1_r32_w32_i16` | none | 0.98× | 0.97× |
| `cuda/f32/index_select/small/axis1_len1` | none | 0.98× | 0.99× |
| `cuda/f32/index_select/small/r32_w64_i31_a1_strided` | none | 0.90× | 1.00× |
| `cuda/f32/index_select/strided/strided_axis0_r2_c7_i2` | none | 0.99× | 1.01× |
| `cuda/f32/index_select/strided/strided_axis0_r65_c127_i33` | none | 1.06× | 1.00× |
| `cuda/f32/index_select/strided/strided_axis0_r7_c129_i4` | none | 0.98× | 0.98× |
| `cuda/f32/index_select/strided/strided_axis1_r32_c31_i16` | none | 0.99× | 0.96× |
| `cuda/f32/index_select/strided/strided_axis1_r3_c5_i3` | none | 0.96× | 0.99× |
| `cuda/f32/index_select/strided/strided_axis1_r65_c127_i64` | none | 0.87× | 0.96× |
