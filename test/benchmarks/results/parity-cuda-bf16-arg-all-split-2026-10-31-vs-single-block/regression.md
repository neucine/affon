# Parity regression check

Target: `NVIDIA GeForce RTX 3090/7.0.0-38-generic/x86_64`

- Cells compared: 6
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
| `cuda/bf16/argmax_all/long_axis` | none | 1.01× | 0.99× |
| `cuda/bf16/argmax_all/outer_axis` | none | 0.39× | 0.48× |
| `cuda/bf16/argmax_all/short_axis` | none | 0.99× | 1.04× |
| `cuda/bf16/argmin_all/long_axis` | none | 1.00× | 1.00× |
| `cuda/bf16/argmin_all/outer_axis` | none | 0.39× | 0.47× |
| `cuda/bf16/argmin_all/short_axis` | none | 1.00× | 1.01× |
