# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `5b276e60963a5aac9a5c59f80f4b4dc5e0b445be35e060eb8fdca87b21cbfac8`  
Binary SHA-256: `caaf0cf1877c6ac144065c22ecaa111cb654e4124d17d211e8c4fe9a562643fe`  
Timing: 10 iterations × 5 samples × 3 rounds

## Checks

- Cases run: 1
- Numerical check passed: 1
- Operator check and timing passed: 1
- Program check and timing passed: 1
- Numerical failures: 0
- Operator failures: 0
- Program failures: 0

## Cases

| Case | Family | Shape | Layout | Numerical | Operator | Affon / PyTorch ms | Program | Affon / PyTorch ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `metal/f32/reduce_to_shape/ragged/dense_r33_c5` | reduction | 33 × 5 | dense | correct | [passed](operator_peer/metal__f32__reduce_to_shape__ragged__dense_r33_c5.json) | 0.1932 / 0.2848 | [passed](program_peer/metal__f32__reduce_to_shape__ragged__dense_r33_c5.json) | 0.2074 / 0.4474 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
