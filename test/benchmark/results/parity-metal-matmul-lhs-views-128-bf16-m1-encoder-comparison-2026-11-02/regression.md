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
| `metal/bf16/matmul/strided/lhs_m1_k129_n17` | none | 1.09× | 1.00× |
| `metal/bf16/matmul/strided/lhs_m1_k129_n65` | none | 1.02× | 1.01× |
| `metal/bf16/matmul/strided/lhs_m1_k31_n17` | none | 1.01× | 0.89× |
| `metal/bf16/matmul/strided/lhs_m1_k31_n65` | none | 0.97× | 0.93× |
| `metal/bf16/matmul/strided/lhs_m1_k64_n17` | none | 0.98× | 0.98× |
| `metal/bf16/matmul/strided/lhs_m1_k64_n65` | none | 0.96× | 0.97× |
| `metal/bf16/matmul/strided/lhs_m1_k7_n17` | none | 0.94× | 0.96× |
| `metal/bf16/matmul/strided/lhs_m1_k7_n65` | none | 0.97× | 1.02× |
| `metal/bf16/matmul/transposed/lhs_m1_k129_n17` | none | 1.02× | 1.02× |
| `metal/bf16/matmul/transposed/lhs_m1_k129_n65` | none | 1.09× | 0.99× |
| `metal/bf16/matmul/transposed/lhs_m1_k31_n17` | none | 0.93× | 0.94× |
| `metal/bf16/matmul/transposed/lhs_m1_k31_n65` | none | 0.98× | 0.96× |
| `metal/bf16/matmul/transposed/lhs_m1_k64_n17` | none | 1.00× | 0.98× |
| `metal/bf16/matmul/transposed/lhs_m1_k64_n65` | none | 0.98× | 1.02× |
| `metal/bf16/matmul/transposed/lhs_m1_k7_n17` | none | 1.06× | 1.00× |
| `metal/bf16/matmul/transposed/lhs_m1_k7_n65` | none | 1.02× | 0.97× |
