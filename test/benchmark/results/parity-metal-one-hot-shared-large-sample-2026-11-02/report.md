# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `dcde4251afc8541a4711dd1130bd063e4bf6e8c7d39365304043866e40692b3b`  
Binary SHA-256: `224e9094dc414008d3ad00e0e6591d3748f673a1ba2c5d03d73c60ce9e1c24d9`  
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
| `metal/f32/one_hot/batched/b8_r16_c8_k33` | indexing | 8 × 16 × 8 | dense | correct | [passed](operator_peer/metal__f32__one_hot__batched__b8_r16_c8_k33.json) | 0.7656 / 0.7546 | [passed](program_peer/metal__f32__one_hot__batched__b8_r16_c8_k33.json) | 0.9856 / 0.7686 |
| `metal/f32/one_hot/batched/b8_r16_c17_k33` | indexing | 8 × 16 × 17 | dense | correct | [passed](operator_peer/metal__f32__one_hot__batched__b8_r16_c17_k33.json) | 1.602 / 1.619 | [passed](program_peer/metal__f32__one_hot__batched__b8_r16_c17_k33.json) | 2.151 / 1.539 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
