# Parity regression check

Target: `NVIDIA GeForce RTX 3090/7.0.0-38-generic/x86_64`

- Cells compared: 3
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
| `cuda/bf16/cross_entropy_indexed/batched` | none | 0.25× | 0.29× |
| `cuda/bf16/cross_entropy_indexed/large` | none | 0.03× | 0.04× |
| `cuda/bf16/cross_entropy_indexed/small` | none | 0.63× | 0.66× |
