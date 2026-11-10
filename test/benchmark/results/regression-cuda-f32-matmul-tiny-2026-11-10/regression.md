# Parity regression check

Target: `NVIDIA GeForce RTX 3090/7.0.0-42-generic/x86_64`

- Cells compared: 32
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
| `cuda/f32/matmul/decode_m1/k63_n64_dense` | none | 0.97× | 1.01× |
| `cuda/f32/matmul/decode_m1/k64_n65_dense` | none | 0.95× | 0.98× |
| `cuda/f32/matmul/decode_m1/m1_k127_n127_default` | none | 1.10× | 1.02× |
| `cuda/f32/matmul/decode_m1/m1_k128_n128_dense` | none | 1.09× | 1.08× |
| `cuda/f32/matmul/decode_m1/m1_k31_n31` | none | 0.93× | 0.94× |
| `cuda/f32/matmul/decode_m1/m1_k32_n127_dense` | none | 0.94× | 0.95× |
| `cuda/f32/matmul/decode_m1/m1_k32_n33` | none | 0.94× | 0.96× |
| `cuda/f32/matmul/decode_m1/m1_k33_n32_default` | none | 0.92× | 0.91× |
| `cuda/f32/matmul/small_batch/grid_m2_k127_n129` | none | 0.96× | 1.02× |
| `cuda/f32/matmul/small_batch/grid_m2_k33_n31` | none | 0.91× | 0.93× |
| `cuda/f32/matmul/small_batch/grid_m2_k5_n31` | none | 0.86× | 0.92× |
| `cuda/f32/matmul/small_batch/grid_m2_k65_n7` | none | 0.91× | 0.95× |
| `cuda/f32/matmul/small_batch/grid_m8_k33_n7` | none | 0.90× | 1.22× |
| `cuda/f32/matmul/small_batch/grid_m8_k65_n7` | none | 0.91× | 0.93× |
| `cuda/f32/matmul/small_batch/m2_k31_n31_default` | none | 0.88× | 0.94× |
| `cuda/f32/matmul/small_batch/m2_k64_n128_dense` | none | 0.94× | 0.93× |
| `cuda/f32/matmul/strided/lhs_m16_k129_n17` | none | 1.02× | 0.97× |
| `cuda/f32/matmul/strided/lhs_m1_k129_n17` | none | 0.77× | 0.82× |
| `cuda/f32/matmul/strided/lhs_m4_k129_n17` | none | 0.79× | 0.82× |
| `cuda/f32/matmul/strided/lhs_m4_k129_n65` | none | 0.77× | 0.79× |
| `cuda/f32/matmul/strided/rhs_m16_k129_n17` | none | 1.02× | 0.98× |
| `cuda/f32/matmul/strided/rhs_m1_k128_n128` | none | 0.86× | 0.87× |
| `cuda/f32/matmul/strided/rhs_m1_k128_n32` | none | 0.81× | 0.82× |
| `cuda/f32/matmul/strided/rhs_m1_k128_n64` | none | 0.79× | 0.83× |
| `cuda/f32/matmul/transposed/lhs_m1_k129_n17` | none | 1.07× | 1.02× |
| `cuda/f32/matmul/transposed/lhs_m1_k64_n65` | none | 1.00× | 0.99× |
| `cuda/f32/matmul/transposed/rhs_m1_k128_n64` | none | 1.07× | 1.09× |
| `cuda/f32/matmul/transposed/rhs_m1_k31_n17` | none | 0.90× | 0.94× |
| `cuda/f32/matmul/transposed/rhs_m1_k32_n8` | none | 1.09× | 0.95× |
| `cuda/f32/matmul/transposed/rhs_m1_k64_n17` | none | 0.93× | 0.96× |
| `cuda/f32/matmul/transposed/rhs_m1_k8_n32` | none | 0.91× | 1.13× |
| `cuda/f32/matmul/transposed/rhs_m33_k8_n8` | none | 1.04× | 1.04× |
