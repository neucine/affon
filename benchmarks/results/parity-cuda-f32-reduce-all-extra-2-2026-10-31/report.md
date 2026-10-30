# Parity batch report

Target: `NVIDIA GeForce RTX 3090/7.0.0-38-generic/x86_64`  
Catalog SHA-256: `65d81a133b07bd4348e1612b85e36b143b33de936d38822e5144874b7be63104`  
Binary SHA-256: `5935499eea7d4fd6b5cfbcce670dedcac747c411e5515df9cf3dd7f0d7a956d0`  
Timing: 10 iterations × 5 samples × 3 rounds

## Checks

- Cases run: 2
- Numerical check passed: 2
- Operator check and timing passed: 2
- Program check and timing passed: 2
- Numerical failures: 0
- Operator failures: 0
- Program failures: 0

## Cases

| Case | Family | Shape | Layout | Numerical | Operator | Affon / PyTorch ms | Program | Affon / PyTorch ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `cuda/f32/sum_all/strided/r8_c512` | reduction | 8 × 512 | strided | correct | [passed](operator_peer/cuda__f32__sum_all__strided__r8_c512.json) | 0.02677 / 0.01592 | [passed](program_peer/cuda__f32__sum_all__strided__r8_c512.json) | 0.03184 / 0.01868 |
| `cuda/f32/argmin_all/offset/r8_c512` | reduction | 8 × 512 | offset | correct | [passed](operator_peer/cuda__f32__argmin_all__offset__r8_c512.json) | 0.0277 / 0.02038 | [passed](program_peer/cuda__f32__argmin_all__offset__r8_c512.json) | 0.03261 / 0.02376 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
