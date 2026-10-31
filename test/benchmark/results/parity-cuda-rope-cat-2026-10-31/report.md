# Parity batch report

Target: `NVIDIA GeForce RTX 3090/7.0.0-38-generic/x86_64`
Catalog SHA-256: `6667a957df54b1a5bbc961c670058bf081a3613745b028216a54770c778b2790`
Binary SHA-256: `a7016a051fc45cdb985aa05b4c27060cb979a41c6f1087266ecdf1a49ed78bca`
Timing: 3 iterations × 3 samples × 2 rounds

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
| `cuda/bf16/cat/materialized` | structural | 16 × 32 | dense | correct | [passed](operator_peer/cuda__bf16__cat__materialized.json) | 0.1168 / 0.02247 | [passed](program_peer/cuda__bf16__cat__materialized.json) | 0.1202 / 0.02545 |
| `cuda/f32/cat/materialized` | structural | 16 × 32 | dense | correct | [passed](operator_peer/cuda__f32__cat__materialized.json) | 0.1861 / 0.02452 | [passed](program_peer/cuda__f32__cat__materialized.json) | 0.1923 / 0.033 |
| `cuda/bf16/cat/ragged` | structural | 17 × 65 | dense | correct | [passed](operator_peer/cuda__bf16__cat__ragged.json) | 0.1418 / 0.03345 | [passed](program_peer/cuda__bf16__cat__ragged.json) | 0.145 / 0.03641 |
| `cuda/f32/cat/ragged` | structural | 17 × 65 | dense | correct | [passed](operator_peer/cuda__f32__cat__ragged.json) | 0.2164 / 0.03599 | [passed](program_peer/cuda__f32__cat__ragged.json) | 0.2216 / 0.03984 |
| `cuda/bf16/cat/strided` | structural | 16 × 32 | strided | correct | [passed](operator_peer/cuda__bf16__cat__strided.json) | 0.1182 / 0.03088 | [passed](program_peer/cuda__bf16__cat__strided.json) | 0.1216 / 0.02827 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
