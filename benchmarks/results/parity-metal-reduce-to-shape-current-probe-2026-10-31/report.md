# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `621e7d4907027df7281727ba371433208462af870b3b71034aa54e62984beee3`  
Binary SHA-256: `16d14cf1b9cd8279ae8a5766519b8506a63a19322c9728d6ba0b391f478b6bfe`  
Timing: 10 iterations × 5 samples × 3 rounds

## Checks

- Cases run: 4
- Numerical check passed: 4
- Operator check and timing passed: 4
- Program check and timing passed: 4
- Numerical failures: 0
- Operator failures: 0
- Program failures: 0

## Cases

| Case | Family | Shape | Layout | Numerical | Operator | Affon / PyTorch ms | Program | Affon / PyTorch ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `metal/f32/reduce_to_shape/short_axis/dense_r128_c129` | reduction | 128 × 129 | dense | correct | [passed](operator_peer/metal__f32__reduce_to_shape__short_axis__dense_r128_c129.json) | 18.28 / 0.2136 | [passed](program_peer/metal__f32__reduce_to_shape__short_axis__dense_r128_c129.json) | 18.31 / 0.1971 |
| `metal/f32/reduce_to_shape/outer_axis/dense_r128_c129` | reduction | 128 × 129 | dense | correct | [passed](operator_peer/metal__f32__reduce_to_shape__outer_axis__dense_r128_c129.json) | 17.31 / 0.1992 | [passed](program_peer/metal__f32__reduce_to_shape__outer_axis__dense_r128_c129.json) | 17.33 / 0.2169 |
| `metal/f32/reduce_to_shape/short_axis/offset_r128_c129` | reduction | 128 × 129 | offset | correct | [passed](operator_peer/metal__f32__reduce_to_shape__short_axis__offset_r128_c129.json) | 18.28 / 0.2167 | [passed](program_peer/metal__f32__reduce_to_shape__short_axis__offset_r128_c129.json) | 18.32 / 0.2132 |
| `metal/f32/reduce_to_shape/outer_axis/offset_r128_c129` | reduction | 128 × 129 | offset | correct | [passed](operator_peer/metal__f32__reduce_to_shape__outer_axis__offset_r128_c129.json) | 17.31 / 0.2129 | [passed](program_peer/metal__f32__reduce_to_shape__outer_axis__offset_r128_c129.json) | 17.33 / 0.2119 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
