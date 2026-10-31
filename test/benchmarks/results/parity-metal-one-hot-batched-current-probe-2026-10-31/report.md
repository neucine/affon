# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `62d3a21686813c4ea51b16140a661679de8894843d988612d1acc716edb06adc`  
Binary SHA-256: `8cf45c727d1adf3054252a616442e3900674aa8612990d5ddcb3e0b39e128490`  
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
| `metal/f32/one_hot/batched/b4_r16_c17_k33` | indexing | 4 × 16 × 17 | dense | correct | [passed](operator_peer/metal__f32__one_hot__batched__b4_r16_c17_k33.json) | 1.039 / 0.7297 | [passed](program_peer/metal__f32__one_hot__batched__b4_r16_c17_k33.json) | 1.003 / 0.7369 |
| `metal/f32/one_hot/batched/b8_r16_c17_k33` | indexing | 8 × 16 × 17 | dense | correct | [passed](operator_peer/metal__f32__one_hot__batched__b8_r16_c17_k33.json) | 2.057 / 1.273 | [passed](program_peer/metal__f32__one_hot__batched__b8_r16_c17_k33.json) | 2.09 / 1.283 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
