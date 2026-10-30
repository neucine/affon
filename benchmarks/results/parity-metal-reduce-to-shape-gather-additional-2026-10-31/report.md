# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `e03307aa55e221742156870e21194ece49a7a254b19620dd1f11bf75c1f913fc`  
Binary SHA-256: `3c4af2ad0335c3d217f5ba2aaa95a9074736951287eae037f6fd5fce8c6a4615`  
Timing: 5 iterations × 5 samples × 3 rounds

## Checks

- Cases run: 6
- Numerical check passed: 6
- Operator check and timing passed: 6
- Program check and timing passed: 6
- Numerical failures: 0
- Operator failures: 0
- Program failures: 0

## Cases

| Case | Family | Shape | Layout | Numerical | Operator | Affon / PyTorch ms | Program | Affon / PyTorch ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `metal/f32/reduce_to_shape/short_axis` | reduction | 3 × 5 | dense | correct | [passed](operator_peer/metal__f32__reduce_to_shape__short_axis.json) | 0.253 / 0.2104 | [passed](program_peer/metal__f32__reduce_to_shape__short_axis.json) | 0.2536 / 0.2061 |
| `metal/f32/reduce_to_shape/long_axis` | reduction | 3 × 5 | dense | correct | [passed](operator_peer/metal__f32__reduce_to_shape__long_axis.json) | 0.2542 / 0.2066 | [passed](program_peer/metal__f32__reduce_to_shape__long_axis.json) | 0.2738 / 0.2094 |
| `metal/f32/reduce_to_shape/outer_axis` | reduction | 3 × 5 | dense | correct | [passed](operator_peer/metal__f32__reduce_to_shape__outer_axis.json) | 0.2486 / 0.2237 | [passed](program_peer/metal__f32__reduce_to_shape__outer_axis.json) | 0.2552 / 0.215 |
| `metal/f32/reduce_to_shape/ragged` | reduction | 3 × 5 | dense | correct | [passed](operator_peer/metal__f32__reduce_to_shape__ragged.json) | 0.3122 / 0.1965 | [passed](program_peer/metal__f32__reduce_to_shape__ragged.json) | 0.3186 / 0.2012 |
| `metal/f32/reduce_to_shape/strided` | reduction | 33 × 65 | strided | correct | [passed](operator_peer/metal__f32__reduce_to_shape__strided.json) | 0.3664 / 0.2113 | [passed](program_peer/metal__f32__reduce_to_shape__strided.json) | 0.3794 / 0.203 |
| `metal/f32/reduce_to_shape/scalar` | reduction | 3 × 5 | dense | correct | [passed](operator_peer/metal__f32__reduce_to_shape__scalar.json) | 0.3046 / 0.2058 | [passed](program_peer/metal__f32__reduce_to_shape__scalar.json) | 0.2344 / 0.1957 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
