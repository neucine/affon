# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `fbf3b559b92693991cfb2628aedd5fa080abfa05366d0126d60e6d4b7bab41c4`  
Binary SHA-256: `94e3c1795a7cbcae6e6202b022ce83d5f4d5a69f03d81ab1e9de230af9c7bc4e`  
Timing: 10 iterations × 5 samples × 3 rounds

## Checks

- Cases run: 3
- Numerical check passed: 3
- Operator check and timing passed: 3
- Program check and timing passed: 3
- Numerical failures: 0
- Operator failures: 0
- Program failures: 0

## Cases

| Case | Family | Shape | Layout | Numerical | Operator | Affon / PyTorch ms | Program | Affon / PyTorch ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `metal/f32/stack/strided/r64_c127` | structural | 64 × 127 | strided | correct | [passed](operator_peer/metal__f32__stack__strided__r64_c127.json) | 0.4623 / 0.4027 | [passed](program_peer/metal__f32__stack__strided__r64_c127.json) | 0.4786 / 0.4233 |
| `metal/f32/cat/strided/r64_c127` | structural | 64 × 127 | strided | correct | [passed](operator_peer/metal__f32__cat__strided__r64_c127.json) | 0.4244 / 0.393 | [passed](program_peer/metal__f32__cat__strided__r64_c127.json) | 0.4274 / 0.4159 |
| `metal/f32/contiguous/strided/r64_c127` | structural | 64 × 127 | strided | correct | [passed](operator_peer/metal__f32__contiguous__strided__r64_c127.json) | 0.3738 / 0.2748 | [passed](program_peer/metal__f32__contiguous__strided__r64_c127.json) | 0.3694 / 0.2777 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
