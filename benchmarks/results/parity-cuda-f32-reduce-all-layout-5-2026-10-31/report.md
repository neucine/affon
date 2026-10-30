# Parity batch report

Target: `NVIDIA GeForce RTX 3090/7.0.0-38-generic/x86_64`  
Catalog SHA-256: `c19bf62e3a37e7736e0d66e82ee0502463a80e220b4b5127cee9ebc16b4ab611`  
Binary SHA-256: `5935499eea7d4fd6b5cfbcce670dedcac747c411e5515df9cf3dd7f0d7a956d0`  
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
| `cuda/f32/argmax_all/strided/r8_c512` | reduction | 8 × 512 | strided | correct | [passed](operator_peer/cuda__f32__argmax_all__strided__r8_c512.json) | 0.02822 / 0.01599 | [passed](program_peer/cuda__f32__argmax_all__strided__r8_c512.json) | 0.03285 / 0.01955 |
| `cuda/f32/min_all/offset/r8_c512` | reduction | 8 × 512 | offset | correct | [passed](operator_peer/cuda__f32__min_all__offset__r8_c512.json) | 0.02836 / 0.02085 | [passed](program_peer/cuda__f32__min_all__offset__r8_c512.json) | 0.0332 / 0.02416 |
| `cuda/f32/mean_all/strided/r8_c512` | reduction | 8 × 512 | strided | correct | [passed](operator_peer/cuda__f32__mean_all__strided__r8_c512.json) | 0.02726 / 0.01636 | [passed](program_peer/cuda__f32__mean_all__strided__r8_c512.json) | 0.03188 / 0.01919 |
| `cuda/f32/variance_all/offset/r8_c512` | reduction | 8 × 512 | offset | correct | [passed](operator_peer/cuda__f32__variance_all__offset__r8_c512.json) | 0.02711 / 0.02334 | [passed](program_peer/cuda__f32__variance_all__offset__r8_c512.json) | 0.03189 / 0.0212 |
| `cuda/f32/std_all/strided/r8_c512` | reduction | 8 × 512 | strided | correct | [passed](operator_peer/cuda__f32__std_all__strided__r8_c512.json) | 0.02713 / 0.0184 | [passed](program_peer/cuda__f32__std_all__strided__r8_c512.json) | 0.0319 / 0.02177 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
