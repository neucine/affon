# Parity regression check

Target: `Mac16,1/26.6.2/arm64`

- Cells compared: 16
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
| `metal/bf16/matmul/decode_m1/m1_k31_n31` | none | 0.90× | 1.00× |
| `metal/bf16/matmul/decode_m1/m1_k31_n32` | none | 0.95× | 0.99× |
| `metal/bf16/matmul/decode_m1/m1_k31_n33` | none | 0.93× | 0.95× |
| `metal/bf16/matmul/decode_m1/m1_k31_n65` | none | 0.87× | 0.95× |
| `metal/bf16/matmul/decode_m1/m1_k32_n31` | none | 0.89× | 1.00× |
| `metal/bf16/matmul/decode_m1/m1_k32_n32` | none | 1.00× | 0.98× |
| `metal/bf16/matmul/decode_m1/m1_k32_n33` | none | 0.93× | 0.97× |
| `metal/bf16/matmul/decode_m1/m1_k32_n65` | none | 1.00× | 0.87× |
| `metal/bf16/matmul/decode_m1/m1_k33_n31` | none | 1.02× | 1.00× |
| `metal/bf16/matmul/decode_m1/m1_k33_n32` | none | 0.89× | 0.98× |
| `metal/bf16/matmul/decode_m1/m1_k33_n33` | none | 0.97× | 0.96× |
| `metal/bf16/matmul/decode_m1/m1_k33_n65` | none | 0.93× | 0.95× |
| `metal/bf16/matmul/decode_m1/m1_k64_n31` | none | 0.94× | 1.02× |
| `metal/bf16/matmul/decode_m1/m1_k64_n32` | none | 0.99× | 0.98× |
| `metal/bf16/matmul/decode_m1/m1_k64_n33` | none | 0.95× | 1.01× |
| `metal/bf16/matmul/decode_m1/m1_k64_n65` | none | 1.02× | 0.97× |
