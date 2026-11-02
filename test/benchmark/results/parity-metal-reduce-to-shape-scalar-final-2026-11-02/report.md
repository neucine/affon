# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `b93b0379d3f529ad9e9ec65f2443a80c2afa380e601f496f8b4757f68d4b5b7a`  
Binary SHA-256: `165614e444d0927f60b4ff553be291135a2315f15d13f69574187782656e462d`  
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
| `metal/f32/reduce_to_shape/ragged/dense_r16_c32` | reduction | 16 × 32 | dense | correct | [passed](operator_peer/metal__f32__reduce_to_shape__ragged__dense_r16_c32.json) | 0.1959 / 0.2367 | [passed](program_peer/metal__f32__reduce_to_shape__ragged__dense_r16_c32.json) | 0.2325 / 0.2136 |
| `metal/f32/reduce_to_shape/ragged/dense_r128_c5` | reduction | 128 × 5 | dense | correct | [passed](operator_peer/metal__f32__reduce_to_shape__ragged__dense_r128_c5.json) | 0.196 / 0.1925 | [passed](program_peer/metal__f32__reduce_to_shape__ragged__dense_r128_c5.json) | 0.2217 / 0.1925 |
| `metal/f32/reduce_to_shape/ragged/offset_r16_c32` | reduction | 16 × 32 | offset | correct | [passed](operator_peer/metal__f32__reduce_to_shape__ragged__offset_r16_c32.json) | 0.1938 / 0.1893 | [passed](program_peer/metal__f32__reduce_to_shape__ragged__offset_r16_c32.json) | 0.2308 / 0.195 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
