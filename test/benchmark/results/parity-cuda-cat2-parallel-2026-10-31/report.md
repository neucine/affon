# Parity batch report

Target: `NVIDIA GeForce RTX 3090/7.0.0-38-generic/x86_64`  
Catalog SHA-256: `6667a957df54b1a5bbc961c670058bf081a3613745b028216a54770c778b2790`  
Binary SHA-256: `1dac33dfde56dcaa7867202fea07861c77f923f43ad182a95b9d02e871c30046`  
Timing: 10 iterations × 5 samples × 3 rounds

## Checks

- Cases run: 5
- Numerical check passed: 5
- Operator check and timing passed: 5
- Program check and timing passed: 5
- Numerical failures: 0
- Operator failures: 0
- Program failures: 0

## Cases

| Case | Family | Shape | Layout | Numerical | Operator | Affon / PyTorch ms | Program | Affon / PyTorch ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `cuda/bf16/cat/materialized` | structural | 16 × 32 | dense | correct | [passed](operator_peer/cuda__bf16__cat__materialized.json) | 0.03963 / 0.02067 | [passed](program_peer/cuda__bf16__cat__materialized.json) | 0.0424 / 0.02502 |
| `cuda/f32/cat/materialized` | structural | 16 × 32 | dense | correct | [passed](operator_peer/cuda__f32__cat__materialized.json) | 0.0406 / 0.02123 | [passed](program_peer/cuda__f32__cat__materialized.json) | 0.04561 / 0.02726 |
| `cuda/bf16/cat/ragged` | structural | 17 × 65 | dense | correct | [passed](operator_peer/cuda__bf16__cat__ragged.json) | 0.06028 / 0.03337 | [passed](program_peer/cuda__bf16__cat__ragged.json) | 0.06236 / 0.03649 |
| `cuda/f32/cat/ragged` | structural | 17 × 65 | dense | correct | [passed](operator_peer/cuda__f32__cat__ragged.json) | 0.06084 / 0.03687 | [passed](program_peer/cuda__f32__cat__ragged.json) | 0.06553 / 0.04022 |
| `cuda/bf16/cat/strided` | structural | 16 × 32 | strided | correct | [passed](operator_peer/cuda__bf16__cat__strided.json) | 0.04015 / 0.02472 | [passed](program_peer/cuda__bf16__cat__strided.json) | 0.04273 / 0.02783 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
