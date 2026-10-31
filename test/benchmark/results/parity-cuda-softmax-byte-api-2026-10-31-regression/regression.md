# Parity regression check

Target: `NVIDIA GeForce RTX 3090/7.0.0-38-generic/x86_64`

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
| `cuda/bf16/log_softmax/long_row` | none | 1.02× | 1.02× |
| `cuda/bf16/softmax/long_row` | none | 1.01× | 1.07× |
| `cuda/f32/log_softmax/long_row` | none | 1.02× | 0.99× |
| `cuda/f32/softmax/long_row` | none | 1.00× | 1.01× |
