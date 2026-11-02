# Parity regression check

Target: `Mac16,1/26.6.2/arm64`

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
| `metal/bf16/dot/dot_or_vector/offset_n1` | none | 0.93× | 1.00× |
| `metal/bf16/dot/dot_or_vector/offset_n128` | none | 1.26× | 1.08× |
| `metal/bf16/dot/dot_or_vector/offset_n31` | none | 0.87× | 0.86× |
| `metal/bf16/dot/dot_or_vector/offset_n9` | none | 0.99× | 0.88× |
| `metal/bf16/dot/strided/n1` | none | 0.96× | 0.87× |
| `metal/bf16/dot/strided/n17` | none | 0.96× | 1.03× |
| `metal/bf16/dot/strided/n31` | none | 0.94× | 1.02× |
| `metal/bf16/dot/strided/n9` | none | 0.97× | 1.02× |
