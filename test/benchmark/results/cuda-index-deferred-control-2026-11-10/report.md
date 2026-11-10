# Parity batch report

Target: `NVIDIA GeForce RTX 3090/7.0.0-42-generic/x86_64`  
Catalog SHA-256: `0cf581ae6b3aba05d744feca1b774ab020c7fc4f3faf5b2675f3483bf9a5f603`  
Binary SHA-256: `42a8c02d83f1ffcb4ddeac1c144f2330ec8a7ce96db672bb8fc89c9321e71b2a`  
Timing: 10 iterations × 5 samples × 3 rounds

## Checks

- Cases run: 6
- Numerical check passed: 6
- Operator check and timing passed: 6
- Program check and timing passed: 6
- Numerical failures: 0
- Operator failures: 0
- Program failures: 0
- Operator checks pending: 0
- Program checks pending: 0

## Cases

| Case | Family | Shape | Layout | Numerical | Operator | Affon / PyTorch ms | Program | Affon / PyTorch ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `cuda/bf16/gather/large/axis1_16x128` | indexing | 16 × 128 | dense | correct | [passed](operator_peer/cuda__bf16__gather__large__axis1_16x128.json) | 0.03579 / 0.03622 | [passed](program_peer/cuda__bf16__gather__large__axis1_16x128.json) | 0.04123 / 0.03142 |
| `cuda/bf16/gather/large/offset_axis0_r16_c129` | indexing | 16 × 129 | offset | correct | [passed](operator_peer/cuda__bf16__gather__large__offset_axis0_r16_c129.json) | 0.03098 / 0.02703 | [passed](program_peer/cuda__bf16__gather__large__offset_axis0_r16_c129.json) | 0.03625 / 0.02816 |
| `cuda/bf16/gather/small/axis1_8x16` | indexing | 8 × 16 | dense | correct | [passed](operator_peer/cuda__bf16__gather__small__axis1_8x16.json) | 0.02522 / 0.01428 | [passed](program_peer/cuda__bf16__gather__small__axis1_8x16.json) | 0.03035 / 0.01618 |
| `cuda/bf16/index_select/large/axis1_len64` | indexing | 128 × 64 | dense | correct | [passed](operator_peer/cuda__bf16__index_select__large__axis1_len64.json) | 0.07974 / 0.09015 | [passed](program_peer/cuda__bf16__index_select__large__axis1_len64.json) | 0.08556 / 0.08531 |
| `cuda/bf16/index_select/large/offset_axis1_r16_c129_i65` | indexing | 16 × 129 | offset | correct | [passed](operator_peer/cuda__bf16__index_select__large__offset_axis1_r16_c129_i65.json) | 0.03119 / 0.02615 | [passed](program_peer/cuda__bf16__index_select__large__offset_axis1_r16_c129_i65.json) | 0.0365 / 0.03267 |
| `cuda/bf16/index_select/small/axis1_len8` | indexing | 16 × 32 | dense | correct | [passed](operator_peer/cuda__bf16__index_select__small__axis1_len8.json) | 0.0257 / 0.02014 | [passed](program_peer/cuda__bf16__index_select__small__axis1_len8.json) | 0.03093 / 0.01656 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
