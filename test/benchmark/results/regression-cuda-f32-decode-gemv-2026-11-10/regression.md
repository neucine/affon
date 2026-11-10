# Parity regression check

Target: `NVIDIA GeForce RTX 3090/7.0.0-42-generic/x86_64`

- Cells compared: 8
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
| `cuda/f32/matmul/decode_m1/k63_n64_default` | none | 0.91× | 1.33× |
| `cuda/f32/matmul/decode_m1/k63_n64_dense` | none | 0.90× | 0.92× |
| `cuda/f32/matmul/decode_m1/k65_n65_default` | none | 0.93× | 0.94× |
| `cuda/f32/matmul/decode_m1/k65_n65_dense` | none | 0.76× | 0.94× |
| `cuda/f32/matmul/decode_m1/m1_k128_n128_default` | none | 0.94× | 0.93× |
| `cuda/f32/matmul/decode_m1/m1_k128_n128_dense` | none | 0.93× | 0.92× |
| `cuda/f32/matmul/decode_m1/m1_k31_n32_default` | none | 0.86× | 0.91× |
| `cuda/f32/matmul/decode_m1/m1_k31_n32_dense` | none | 0.92× | 0.89× |
