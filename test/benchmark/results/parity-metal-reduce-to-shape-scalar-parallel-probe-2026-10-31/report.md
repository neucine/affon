# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `424813f8368d7eecef2a8687db42fc60fdbf11a2b0a345be3c411dc777157003`  
Binary SHA-256: `bbaf2688179046b86e82f22d7971334adf887d5c6ba64b904b94b3dd72073601`  
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
| `metal/f32/reduce_to_shape/ragged/dense_r128_c129` | reduction | 128 × 129 | dense | correct | [passed](operator_peer/metal__f32__reduce_to_shape__ragged__dense_r128_c129.json) | 0.3625 / 0.2067 | [passed](program_peer/metal__f32__reduce_to_shape__ragged__dense_r128_c129.json) | 0.3741 / 0.2058 |
| `metal/f32/reduce_to_shape/ragged/offset_r128_c129` | reduction | 128 × 129 | offset | correct | [passed](operator_peer/metal__f32__reduce_to_shape__ragged__offset_r128_c129.json) | 0.2974 / 0.2059 | [passed](program_peer/metal__f32__reduce_to_shape__ragged__offset_r128_c129.json) | 0.2965 / 0.2122 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
