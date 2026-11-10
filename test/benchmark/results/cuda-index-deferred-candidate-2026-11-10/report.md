# Parity batch report

Target: `NVIDIA GeForce RTX 3090/7.0.0-42-generic/x86_64`  
Catalog SHA-256: `0cf581ae6b3aba05d744feca1b774ab020c7fc4f3faf5b2675f3483bf9a5f603`  
Binary SHA-256: `b3db0353778c60e0dbe45469cdc58ffb600e5050608b3c61598a2959470686d4`  
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
| `cuda/bf16/gather/large/axis1_16x128` | indexing | 16 × 128 | dense | correct | [passed](operator_peer/cuda__bf16__gather__large__axis1_16x128.json) | 0.03448 / 0.03598 | [passed](program_peer/cuda__bf16__gather__large__axis1_16x128.json) | 0.03713 / 0.03366 |
| `cuda/bf16/gather/large/offset_axis0_r16_c129` | indexing | 16 × 129 | offset | correct | [passed](operator_peer/cuda__bf16__gather__large__offset_axis0_r16_c129.json) | 0.02936 / 0.02965 | [passed](program_peer/cuda__bf16__gather__large__offset_axis0_r16_c129.json) | 0.03203 / 0.02802 |
| `cuda/bf16/gather/small/axis1_8x16` | indexing | 8 × 16 | dense | correct | [passed](operator_peer/cuda__bf16__gather__small__axis1_8x16.json) | 0.02362 / 0.01283 | [passed](program_peer/cuda__bf16__gather__small__axis1_8x16.json) | 0.02586 / 0.0158 |
| `cuda/bf16/index_select/large/axis1_len64` | indexing | 128 × 64 | dense | correct | [passed](operator_peer/cuda__bf16__index_select__large__axis1_len64.json) | 0.07825 / 0.086 | [passed](program_peer/cuda__bf16__index_select__large__axis1_len64.json) | 0.08038 / 0.08759 |
| `cuda/bf16/index_select/large/offset_axis1_r16_c129_i65` | indexing | 16 × 129 | offset | correct | [passed](operator_peer/cuda__bf16__index_select__large__offset_axis1_r16_c129_i65.json) | 0.0295 / 0.02589 | [passed](program_peer/cuda__bf16__index_select__large__offset_axis1_r16_c129_i65.json) | 0.03209 / 0.02911 |
| `cuda/bf16/index_select/small/axis1_len8` | indexing | 16 × 32 | dense | correct | [passed](operator_peer/cuda__bf16__index_select__small__axis1_len8.json) | 0.02395 / 0.02122 | [passed](program_peer/cuda__bf16__index_select__small__axis1_len8.json) | 0.02639 / 0.01717 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
