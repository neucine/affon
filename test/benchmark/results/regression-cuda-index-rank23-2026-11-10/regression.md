# Parity regression check

Target: `NVIDIA GeForce RTX 3090/7.0.0-42-generic/x86_64`

- Cells compared: 11
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
| `cuda/bf16/gather/large/a0_r64_w64_o32` | none | 1.05× | 1.02× |
| `cuda/bf16/gather/large/a1_r64_w64_o32` | none | 0.98× | 0.94× |
| `cuda/bf16/gather/large/offset_axis0_r16_c129` | none | 1.00× | 0.98× |
| `cuda/bf16/gather/small/a1_r32_w7_o1` | none | 1.03× | 1.01× |
| `cuda/bf16/gather/strided/axis1_r65_c7` | none | 0.98× | 0.94× |
| `cuda/bf16/index_select/batched/dense_axis1_b2_r16_c129_n8` | none | 0.99× | 0.92× |
| `cuda/bf16/index_select/batched/dense_axis1_b4_r33_c5_n16` | none | 0.97× | 0.98× |
| `cuda/bf16/index_select/large` | none | 0.98× | 1.01× |
| `cuda/bf16/index_select/small` | none | 0.98× | 1.00× |
| `cuda/f32/gather/strided/axis1_r65_c7` | none | 1.05× | 0.99× |
| `cuda/f32/index_select/batched/dense_axis2_b4_r33_c5_n2` | none | 0.99× | 1.00× |
