# Parity regression check

Target: `NVIDIA GeForce RTX 3090/7.0.0-38-generic/x86_64`

- Cells compared: 19
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
| `cuda/bf16/matmul/decode_m1/k63_n64_default` | none | 1.08× | 1.05× |
| `cuda/bf16/matmul/large_matrix/grid_m128_k127_n129` | none | 1.00× | 0.99× |
| `cuda/bf16/matmul/prefill` | none | 0.96× | 1.03× |
| `cuda/bf16/matmul/prefill/model_768` | none | 0.97× | 0.99× |
| `cuda/bf16/matmul/small_batch/grid_m2_k33_n31` | none | 0.99× | 1.00× |
| `cuda/bf16/matmul/small_batch/grid_m2_k5_n7` | none | 0.94× | 0.92× |
| `cuda/bf16/matmul/small_batch/grid_m8_k127_n129` | none | 1.07× | 1.02× |
| `cuda/bf16/matmul/small_batch/grid_m8_k65_n63` | none | 0.83× | 0.99× |
| `cuda/bf16/matmul/strided/lhs_m16_k129_n17` | none | 1.04× | 0.95× |
| `cuda/bf16/matmul/transposed/lhs_m16_k129_n17` | none | 1.01× | 0.99× |
| `cuda/f32/matmul/decode_m1/k63_n64_default` | none | 1.09× | 1.08× |
| `cuda/f32/matmul/large_matrix/grid_m128_k127_n129` | none | 0.98× | 1.02× |
| `cuda/f32/matmul/prefill` | none | 1.01× | 1.00× |
| `cuda/f32/matmul/small_batch/grid_m2_k33_n31` | none | 1.01× | 1.00× |
| `cuda/f32/matmul/small_batch/grid_m2_k5_n7` | none | 0.93× | 0.98× |
| `cuda/f32/matmul/small_batch/grid_m8_k127_n129` | none | 0.97× | 1.02× |
| `cuda/f32/matmul/small_batch/grid_m8_k65_n63` | none | 1.01× | 1.01× |
| `cuda/f32/matmul/strided/lhs_m16_k129_n17` | none | 1.02× | 1.03× |
| `cuda/f32/matmul/transposed/lhs_m16_k129_n17` | none | 1.03× | 1.00× |
