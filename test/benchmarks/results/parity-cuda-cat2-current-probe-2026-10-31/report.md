# Parity batch report

Target: `NVIDIA GeForce RTX 3090/7.0.0-38-generic/x86_64`  
Catalog SHA-256: `6667a957df54b1a5bbc961c670058bf081a3613745b028216a54770c778b2790`  
Binary SHA-256: `7cb6bad6ea4bdd1ab503d9437d11521e3232b764a4d340b796b90370eec5cebc`  
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
| `cuda/bf16/cat/materialized` | structural | 16 × 32 | dense | correct | [passed](operator_peer/cuda__bf16__cat__materialized.json) | 0.1151 / 0.02044 | [passed](program_peer/cuda__bf16__cat__materialized.json) | 0.1189 / 0.02342 |
| `cuda/f32/cat/materialized` | structural | 16 × 32 | dense | correct | [passed](operator_peer/cuda__f32__cat__materialized.json) | 0.1865 / 0.02095 | [passed](program_peer/cuda__f32__cat__materialized.json) | 0.1915 / 0.0266 |
| `cuda/bf16/cat/ragged` | structural | 17 × 65 | dense | correct | [passed](operator_peer/cuda__bf16__cat__ragged.json) | 0.1409 / 0.03342 | [passed](program_peer/cuda__bf16__cat__ragged.json) | 0.1438 / 0.03636 |
| `cuda/f32/cat/ragged` | structural | 17 × 65 | dense | correct | [passed](operator_peer/cuda__f32__cat__ragged.json) | 0.2171 / 0.03164 | [passed](program_peer/cuda__f32__cat__ragged.json) | 0.2219 / 0.0336 |
| `cuda/bf16/cat/strided` | structural | 16 × 32 | strided | correct | [passed](operator_peer/cuda__bf16__cat__strided.json) | 0.1175 / 0.02466 | [passed](program_peer/cuda__bf16__cat__strided.json) | 0.1202 / 0.02779 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
