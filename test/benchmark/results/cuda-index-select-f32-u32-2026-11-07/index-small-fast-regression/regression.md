# Parity regression check

Target: `NVIDIA GeForce RTX 3090/7.0.0-38-generic/x86_64`

- Cells compared: 33
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
| `cuda/bf16/embedding/batched` | none | 1.02× | 0.99× |
| `cuda/bf16/embedding/batched/batch3` | none | 1.01× | 1.00× |
| `cuda/bf16/embedding/batched/batch8` | none | 0.97× | 1.06× |
| `cuda/bf16/embedding/large/r128_c127_i64` | none | 1.01× | 1.00× |
| `cuda/bf16/embedding/large/r64_w31_i31` | none | 1.04× | 1.02× |
| `cuda/bf16/embedding/large/rows65` | none | 0.99× | 1.03× |
| `cuda/bf16/embedding/small` | none | 1.00× | 1.01× |
| `cuda/bf16/embedding/small/r32_w16_i16` | none | 1.00× | 0.99× |
| `cuda/bf16/embedding/small/r32_w32_i16_strided` | none | 1.01× | 1.04× |
| `cuda/bf16/embedding/strided` | none | 1.00× | 0.98× |
| `cuda/bf16/embedding/strided/r64_c127_i16` | none | 1.07× | 1.05× |
| `cuda/bf16/gather/batched` | none | 1.00× | 0.97× |
| `cuda/bf16/gather/large/offset_axis0_r3_c3` | none | 1.00× | 1.02× |
| `cuda/bf16/gather/large/offset_axis1_r8_c63` | none | 1.02× | 1.00× |
| `cuda/bf16/gather/large/r128_c127_i64` | none | 1.00× | 1.01× |
| `cuda/bf16/gather/small` | none | 1.01× | 0.98× |
| `cuda/bf16/gather/small/a1_r32_w31_o16` | none | 0.99× | 0.97× |
| `cuda/bf16/gather/small/r32_w64_i32x31_a1_strided` | none | 1.01× | 1.08× |
| `cuda/bf16/gather/strided/axis0_r3_c3` | none | 1.02× | 0.98× |
| `cuda/bf16/gather/strided/axis0_r7_c64` | none | 1.00× | 0.98× |
| `cuda/bf16/gather/strided/axis1_r65_c127` | none | 0.91× | 0.98× |
| `cuda/bf16/index_select/batched/dense_axis1_b2_r3_c5_n1` | none | 1.02× | 0.97× |
| `cuda/bf16/index_select/batched/dense_axis2_b4_r33_c129_n64` | none | 1.00× | 0.99× |
| `cuda/bf16/index_select/batched/dense_axis2_b4_r3_c65_n32` | none | 1.02× | 0.99× |
| `cuda/bf16/index_select/large/offset_axis0_r2_c7_i2` | none | 1.01× | 0.97× |
| `cuda/bf16/index_select/large/offset_axis0_r7_c129_i4` | none | 0.98× | 0.97× |
| `cuda/bf16/index_select/large/r128_c127_i64` | none | 1.00× | 1.01× |
| `cuda/bf16/index_select/small` | none | 1.00× | 0.98× |
| `cuda/bf16/index_select/small/a1_r32_w31_i15` | none | 1.00× | 0.98× |
| `cuda/bf16/index_select/small/r32_w64_i31_a1_strided` | none | 1.00× | 1.04× |
| `cuda/bf16/index_select/strided/strided_axis0_r2_c7_i2` | none | 1.02× | 0.95× |
| `cuda/bf16/index_select/strided/strided_axis1_r65_c127_i64` | none | 1.04× | 1.03× |
| `cuda/bf16/index_select/strided/strided_axis1_r65_c7_i4` | none | 0.97× | 0.99× |
