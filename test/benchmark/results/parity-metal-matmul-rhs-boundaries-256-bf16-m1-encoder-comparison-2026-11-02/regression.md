# Parity regression check

Target: `Mac16,1/26.6.2/arm64`

- Cells compared: 4
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
| `metal/bf16/matmul/strided/rhs_m1_k128_n128` | none | 1.16× | 1.01× |
| `metal/bf16/matmul/strided/rhs_m1_k32_n32` | none | 0.89× | 1.01× |
| `metal/bf16/matmul/transposed/rhs_m1_k63_n64` | none | 1.00× | 0.98× |
| `metal/bf16/matmul/transposed/rhs_m1_k8_n8` | none | 1.09× | 0.92× |
