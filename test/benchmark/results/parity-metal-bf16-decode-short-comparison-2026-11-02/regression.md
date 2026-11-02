# Parity regression check

Target: `Mac16,1/26.6.2/arm64`

- Cells compared: 16
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
| `metal/bf16/matmul/decode_m1/m1_k31_n31` | none | 0.78× | 1.06× |
| `metal/bf16/matmul/decode_m1/m1_k31_n32` | none | 1.02× | 1.66× |
| `metal/bf16/matmul/decode_m1/m1_k31_n33` | none | 1.02× | 1.27× |
| `metal/bf16/matmul/decode_m1/m1_k31_n65` | none | 1.15× | 1.21× |
| `metal/bf16/matmul/decode_m1/m1_k32_n31` | none | 1.42× | 0.99× |
| `metal/bf16/matmul/decode_m1/m1_k32_n32` | none | 1.14× | 1.04× |
| `metal/bf16/matmul/decode_m1/m1_k32_n33` | none | 0.99× | 1.25× |
| `metal/bf16/matmul/decode_m1/m1_k32_n65` | none | 0.90× | 1.02× |
| `metal/bf16/matmul/decode_m1/m1_k33_n31` | none | 0.97× | 0.95× |
| `metal/bf16/matmul/decode_m1/m1_k33_n32` | none | 1.18× | 0.99× |
| `metal/bf16/matmul/decode_m1/m1_k33_n33` | none | 1.33× | 0.80× |
| `metal/bf16/matmul/decode_m1/m1_k33_n65` | none | 1.07× | 1.10× |
| `metal/bf16/matmul/decode_m1/m1_k64_n31` | none | 1.03× | 1.10× |
| `metal/bf16/matmul/decode_m1/m1_k64_n32` | none | 1.23× | 1.08× |
| `metal/bf16/matmul/decode_m1/m1_k64_n33` | none | 0.98× | 1.19× |
| `metal/bf16/matmul/decode_m1/m1_k64_n65` | none | 0.92× | 0.98× |
