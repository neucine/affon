# Parity regression check

Target: `Mac16,1/26.6.2/arm64`

- Cells compared: 6
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
| `metal/bf16/contiguous/ragged/r33_c129` | none | 0.97× | 0.88× |
| `metal/bf16/matmul/prefill/model_768` | none | 1.93× | 2.00× |
| `metal/f32/contiguous/materialized` | none | 0.88× | 0.88× |
| `metal/f32/contiguous/strided/r64_c127` | none | 0.94× | 0.90× |
| `metal/f32/matmul/decode_m1/m1_k128_n128_default` | none | 0.96× | 1.10× |
| `metal/f32/matmul/prefill` | none | 1.19× | 1.03× |
