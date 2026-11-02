# Parity regression check

Target: `Mac16,1/26.6.2/arm64`

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
| `metal/bf16/dot/dot_or_vector/n1` | none | 0.95× | 0.87× |
| `metal/bf16/dot/dot_or_vector/n17` | none | 0.97× | 0.94× |
| `metal/bf16/dot/dot_or_vector/n2048` | none | 1.91× | 1.83× |
| `metal/bf16/dot/dot_or_vector/n25` | none | 0.95× | 0.92× |
| `metal/bf16/dot/dot_or_vector/n257` | none | 1.05× | 1.26× |
| `metal/bf16/dot/dot_or_vector/n33` | none | 1.00× | 0.94× |
| `metal/bf16/dot/dot_or_vector/n41` | none | 0.95× | 0.96× |
| `metal/bf16/dot/dot_or_vector/n9` | none | 1.00× | 0.85× |
