# Parity regression check

Target: `NVIDIA GeForce RTX 3090/7.0.0-42-generic/x86_64`

- Cells compared: 8
- Added cells: 0
- Removed cells: 0
- Changed case definitions: 0
- New correctness failures: 0
- Diagnostic timing suspects: 1
- Target matches: yes
- Measurement protocol matches: yes

Timing changes are diagnostic and do not fail this check. Timing is compared only when target and protocol match. No timing baseline is accepted by this comparison.

| Cell | Correctness regression | Operator relative timing | Program relative timing |
| --- | --- | ---: | ---: |
| `cuda/f32/matmul/decode_m1/k63_n64_default` | none | 1.12× | 1.59× |
| `cuda/f32/matmul/decode_m1/k63_n64_dense` | none | 1.07× | 1.05× |
| `cuda/f32/matmul/decode_m1/k65_n65_default` | none | 1.13× | 1.12× |
| `cuda/f32/matmul/decode_m1/k65_n65_dense` | none | 1.16× | 1.11× |
| `cuda/f32/matmul/decode_m1/m1_k128_n128_default` | none | 1.34× | 1.24× |
| `cuda/f32/matmul/decode_m1/m1_k128_n128_dense` | none | 1.34× | 1.28× |
| `cuda/f32/matmul/decode_m1/m1_k31_n32_default` | none | 1.01× | 1.00× |
| `cuda/f32/matmul/decode_m1/m1_k31_n32_dense` | none | 1.00× | 0.99× |
