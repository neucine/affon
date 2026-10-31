# Parity batch report

Target: `NVIDIA GeForce RTX 3090/7.0.0-38-generic/x86_64`
Catalog SHA-256: `27758a6e70bd968432d47a1673b88dd024625381aae3ecab64a3b6dd6006076b`
Binary SHA-256: `1424890d815adbc7da93cdb76906f8f2c562e99893f6cea066747dd77d4ab4ab`
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
| `cuda/bf16/index_select/small` | indexing | 8 × 16 | dense | correct | [passed](operator_peer/cuda__bf16__index_select__small.json) | 0.02078 / 0.01327 | [passed](program_peer/cuda__bf16__index_select__small.json) | 0.02304 / 0.01814 |
| `cuda/f32/index_select/small` | indexing | 8 × 16 | dense | correct | [passed](operator_peer/cuda__f32__index_select__small.json) | 0.02044 / 0.01321 | [passed](program_peer/cuda__f32__index_select__small.json) | 0.02533 / 0.01608 |
| `cuda/bf16/index_select/large` | indexing | 128 × 64 | dense | correct | [passed](operator_peer/cuda__bf16__index_select__large.json) | 0.06431 / 0.03969 | [passed](program_peer/cuda__bf16__index_select__large.json) | 0.06655 / 0.03692 |
| `cuda/f32/index_select/large` | indexing | 128 × 64 | dense | correct | [passed](operator_peer/cuda__f32__index_select__large.json) | 0.06524 / 0.03559 | [passed](program_peer/cuda__f32__index_select__large.json) | 0.06996 / 0.03972 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
