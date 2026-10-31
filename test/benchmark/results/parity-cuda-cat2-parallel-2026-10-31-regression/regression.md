# Parity regression check

Target: `NVIDIA GeForce RTX 3090/7.0.0-38-generic/x86_64`

- Cells compared: 5
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
| `cuda/bf16/cat/materialized` | none | 0.45× | 0.33× |
| `cuda/bf16/cat/ragged` | none | 0.43× | 0.42× |
| `cuda/bf16/cat/strided` | none | 0.34× | 0.36× |
| `cuda/f32/cat/materialized` | none | 0.24× | 0.23× |
| `cuda/f32/cat/ragged` | none | 0.28× | 0.29× |
