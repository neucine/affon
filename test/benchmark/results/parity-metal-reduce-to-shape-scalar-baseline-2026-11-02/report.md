# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `b93b0379d3f529ad9e9ec65f2443a80c2afa380e601f496f8b4757f68d4b5b7a`  
Binary SHA-256: `880455d53f20a11059886477b295503cf1e8928a20ce407ee5beb0548962e4ff`  
Timing: 30 iterations × 5 samples × 3 rounds

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
| `metal/f32/reduce_to_shape/ragged/dense_r16_c32` | reduction | 16 × 32 | dense | correct | [passed](operator_peer/metal__f32__reduce_to_shape__ragged__dense_r16_c32.json) | 0.7761 / 0.2071 | [passed](program_peer/metal__f32__reduce_to_shape__ragged__dense_r16_c32.json) | 0.7643 / 0.2057 |
| `metal/f32/reduce_to_shape/ragged/dense_r128_c5` | reduction | 128 × 5 | dense | correct | [passed](operator_peer/metal__f32__reduce_to_shape__ragged__dense_r128_c5.json) | 1.015 / 0.2025 | [passed](program_peer/metal__f32__reduce_to_shape__ragged__dense_r128_c5.json) | 0.9642 / 0.208 |
| `metal/f32/reduce_to_shape/ragged/offset_r16_c32` | reduction | 16 × 32 | offset | correct | [passed](operator_peer/metal__f32__reduce_to_shape__ragged__offset_r16_c32.json) | 0.7875 / 0.2068 | [passed](program_peer/metal__f32__reduce_to_shape__ragged__offset_r16_c32.json) | 0.7949 / 0.2128 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
