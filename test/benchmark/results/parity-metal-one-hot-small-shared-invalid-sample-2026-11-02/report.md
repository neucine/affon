# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `08d18cc297c72848aed9a703d3a4881cb723f12119b787118d4300be0abda6da`  
Binary SHA-256: `3178c8fac99312ffa4cde0111d37fcfe3869579640a313a1955be26f088bddcf`  
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
| `metal/f32/one_hot/invalid_index/b1_n3_c3` | indexing | 1 × 3 | dense | correct | [passed](operator_peer/metal__f32__one_hot__invalid_index__b1_n3_c3.json) | 0.0091 / 0.2752 | [passed](program_peer/metal__f32__one_hot__invalid_index__b1_n3_c3.json) | 0.2101 / 0.241 |
| `metal/f32/one_hot/invalid_index/b8_n33_c32` | indexing | 8 × 33 | dense | correct | [passed](operator_peer/metal__f32__one_hot__invalid_index__b8_n33_c32.json) | 0.1936 / 0.3327 | [passed](program_peer/metal__f32__one_hot__invalid_index__b8_n33_c32.json) | 0.417 / 0.34 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
