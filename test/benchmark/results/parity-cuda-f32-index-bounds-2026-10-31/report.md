# Parity batch report

Target: `NVIDIA GeForce RTX 3090/7.0.0-38-generic/x86_64`
Catalog SHA-256: `27758a6e70bd968432d47a1673b88dd024625381aae3ecab64a3b6dd6006076b`
Binary SHA-256: `56147b7d2f11254870028f06637090c1cb9bd2ae7db0eb42257da960d6b02e2d`
Timing: 3 iterations × 3 samples × 2 rounds

## Checks

- Cases run: 4
- Numerical check passed: 4
- Operator check and timing passed: 4
- Program check and timing passed: 4
- Numerical failures: 0
- Operator failures: 0
- Program failures: 0

## Cases

| Case | Family | Shape | Layout | Numerical | Operator | Affon / PyTorch ms | Program | Affon / PyTorch ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `cuda/bf16/index_select/small` | indexing | 8 × 16 | dense | correct | [passed](operator_peer/cuda__bf16__index_select__small.json) | 0.02112 / 0.01304 | [passed](program_peer/cuda__bf16__index_select__small.json) | 0.02333 / 0.01606 |
| `cuda/f32/index_select/small` | indexing | 8 × 16 | dense | correct | [passed](operator_peer/cuda__f32__index_select__small.json) | 0.02134 / 0.01314 | [passed](program_peer/cuda__f32__index_select__small.json) | 0.02555 / 0.02589 |
| `cuda/bf16/index_select/large` | indexing | 128 × 64 | dense | correct | [passed](operator_peer/cuda__bf16__index_select__large.json) | 0.06343 / 0.03472 | [passed](program_peer/cuda__bf16__index_select__large.json) | 0.06615 / 0.03674 |
| `cuda/f32/index_select/large` | indexing | 128 × 64 | dense | correct | [passed](operator_peer/cuda__f32__index_select__large.json) | 0.06494 / 0.04095 | [passed](program_peer/cuda__f32__index_select__large.json) | 0.07037 / 0.03935 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
