# Parity batch report

Target: `NVIDIA GeForce RTX 3090/7.0.0-42-generic/x86_64`  
Catalog SHA-256: `4642da50439cd81ccbd947ef1bae2c5aa7312405b491393045a248bcbcce6c8f`  
Binary SHA-256: `becb09bf27d4371837102c15edc60614ce07243e3a0085d34627b5bedc47ea98`  
Timing: 10 iterations × 5 samples × 3 rounds

## Checks

- Cases run: 6
- Numerical check passed: 6
- Operator check and timing passed: 6
- Program check and timing passed: 6
- Numerical failures: 0
- Operator failures: 0
- Program failures: 0
- Operator checks pending: 0
- Program checks pending: 0

## Cases

| Case | Family | Shape | Layout | Numerical | Operator | Affon / PyTorch ms | Program | Affon / PyTorch ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `cuda/f32/reduce_to_shape/long_axis/dense_r128_c129` | reduction | 128 × 129 | dense | correct | [passed](operator_peer/cuda__f32__reduce_to_shape__long_axis__dense_r128_c129.json) | 0.0306 / 0.01464 | [passed](program_peer/cuda__f32__reduce_to_shape__long_axis__dense_r128_c129.json) | 0.03552 / 0.01784 |
| `cuda/f32/reduce_to_shape/long_axis/dense_r128_c5` | reduction | 128 × 5 | dense | correct | [passed](operator_peer/cuda__f32__reduce_to_shape__long_axis__dense_r128_c5.json) | 0.03049 / 0.0147 | [passed](program_peer/cuda__f32__reduce_to_shape__long_axis__dense_r128_c5.json) | 0.0355 / 0.01751 |
| `cuda/f32/reduce_to_shape/outer_axis/dense_r128_c129` | reduction | 128 × 129 | dense | correct | [passed](operator_peer/cuda__f32__reduce_to_shape__outer_axis__dense_r128_c129.json) | 0.01693 / 0.0215 | [passed](program_peer/cuda__f32__reduce_to_shape__outer_axis__dense_r128_c129.json) | 0.0223 / 0.02462 |
| `cuda/f32/reduce_to_shape/ragged/dense_r128_c129` | reduction | 128 × 129 | dense | correct | [passed](operator_peer/cuda__f32__reduce_to_shape__ragged__dense_r128_c129.json) | 0.03354 / 0.01808 | [passed](program_peer/cuda__f32__reduce_to_shape__ragged__dense_r128_c129.json) | 0.03821 / 0.01429 |
| `cuda/f32/reduce_to_shape/scalar` | reduction | 3 × 5 | dense | correct | [passed](operator_peer/cuda__f32__reduce_to_shape__scalar.json) | 0.0161 / 0.01107 | [passed](program_peer/cuda__f32__reduce_to_shape__scalar.json) | 0.02092 / 0.01396 |
| `cuda/f32/reduce_to_shape/short_axis/dense_r16_c65` | reduction | 16 × 65 | dense | correct | [passed](operator_peer/cuda__f32__reduce_to_shape__short_axis__dense_r16_c65.json) | 0.01601 / 0.01182 | [passed](program_peer/cuda__f32__reduce_to_shape__short_axis__dense_r16_c65.json) | 0.02083 / 0.0154 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
