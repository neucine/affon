# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `621e7d4907027df7281727ba371433208462af870b3b71034aa54e62984beee3`  
Binary SHA-256: `3c4af2ad0335c3d217f5ba2aaa95a9074736951287eae037f6fd5fce8c6a4615`  
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
| `metal/f32/reduce_to_shape/short_axis/dense_r128_c129` | reduction | 128 × 129 | dense | correct | [passed](operator_peer/metal__f32__reduce_to_shape__short_axis__dense_r128_c129.json) | 0.3044 / 0.2066 | [passed](program_peer/metal__f32__reduce_to_shape__short_axis__dense_r128_c129.json) | 0.2938 / 0.2053 |
| `metal/f32/reduce_to_shape/outer_axis/dense_r128_c129` | reduction | 128 × 129 | dense | correct | [passed](operator_peer/metal__f32__reduce_to_shape__outer_axis__dense_r128_c129.json) | 0.3422 / 0.2069 | [passed](program_peer/metal__f32__reduce_to_shape__outer_axis__dense_r128_c129.json) | 0.3011 / 0.2099 |
| `metal/f32/reduce_to_shape/short_axis/offset_r128_c129` | reduction | 128 × 129 | offset | correct | [passed](operator_peer/metal__f32__reduce_to_shape__short_axis__offset_r128_c129.json) | 0.2941 / 0.2031 | [passed](program_peer/metal__f32__reduce_to_shape__short_axis__offset_r128_c129.json) | 0.2952 / 0.2073 |
| `metal/f32/reduce_to_shape/outer_axis/offset_r128_c129` | reduction | 128 × 129 | offset | correct | [passed](operator_peer/metal__f32__reduce_to_shape__outer_axis__offset_r128_c129.json) | 0.3357 / 0.2085 | [passed](program_peer/metal__f32__reduce_to_shape__outer_axis__offset_r128_c129.json) | 0.2914 / 0.2048 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
