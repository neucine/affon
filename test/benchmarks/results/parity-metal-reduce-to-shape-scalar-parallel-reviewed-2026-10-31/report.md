# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `f2cdfb22b3d7b1d49e5f7720ffe5727149d7f50e574da4e202c1a5667ae8f439`  
Binary SHA-256: `bbaf2688179046b86e82f22d7971334adf887d5c6ba64b904b94b3dd72073601`  
Timing: 10 iterations × 5 samples × 3 rounds

## Checks

- Cases run: 16
- Numerical check passed: 16
- Operator check and timing passed: 16
- Program check and timing passed: 16
- Numerical failures: 0
- Operator failures: 0
- Program failures: 0

## Cases

| Case | Family | Shape | Layout | Numerical | Operator | Affon / PyTorch ms | Program | Affon / PyTorch ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `metal/f32/reduce_to_shape/ragged/dense_r16_c65` | reduction | 16 × 65 | dense | correct | [passed](operator_peer/metal__f32__reduce_to_shape__ragged__dense_r16_c65.json) | 0.2583 / 0.2053 | [passed](program_peer/metal__f32__reduce_to_shape__ragged__dense_r16_c65.json) | 0.2821 / 0.2074 |
| `metal/f32/reduce_to_shape/ragged/dense_r16_c129` | reduction | 16 × 129 | dense | correct | [passed](operator_peer/metal__f32__reduce_to_shape__ragged__dense_r16_c129.json) | 0.2848 / 0.1956 | [passed](program_peer/metal__f32__reduce_to_shape__ragged__dense_r16_c129.json) | 0.3034 / 0.2007 |
| `metal/f32/reduce_to_shape/ragged/dense_r33_c32` | reduction | 33 × 32 | dense | correct | [passed](operator_peer/metal__f32__reduce_to_shape__ragged__dense_r33_c32.json) | 0.2347 / 0.2069 | [passed](program_peer/metal__f32__reduce_to_shape__ragged__dense_r33_c32.json) | 0.2702 / 0.2046 |
| `metal/f32/reduce_to_shape/ragged/dense_r33_c65` | reduction | 33 × 65 | dense | correct | [passed](operator_peer/metal__f32__reduce_to_shape__ragged__dense_r33_c65.json) | 0.2995 / 0.2051 | [passed](program_peer/metal__f32__reduce_to_shape__ragged__dense_r33_c65.json) | 0.3059 / 0.2055 |
| `metal/f32/reduce_to_shape/ragged/dense_r33_c129` | reduction | 33 × 129 | dense | correct | [passed](operator_peer/metal__f32__reduce_to_shape__ragged__dense_r33_c129.json) | 0.3268 / 0.2045 | [passed](program_peer/metal__f32__reduce_to_shape__ragged__dense_r33_c129.json) | 0.2346 / 0.2028 |
| `metal/f32/reduce_to_shape/ragged/dense_r128_c32` | reduction | 128 × 32 | dense | correct | [passed](operator_peer/metal__f32__reduce_to_shape__ragged__dense_r128_c32.json) | 0.2781 / 0.1957 | [passed](program_peer/metal__f32__reduce_to_shape__ragged__dense_r128_c32.json) | 0.3621 / 0.201 |
| `metal/f32/reduce_to_shape/ragged/dense_r128_c65` | reduction | 128 × 65 | dense | correct | [passed](operator_peer/metal__f32__reduce_to_shape__ragged__dense_r128_c65.json) | 0.2813 / 0.1988 | [passed](program_peer/metal__f32__reduce_to_shape__ragged__dense_r128_c65.json) | 0.2573 / 0.2028 |
| `metal/f32/reduce_to_shape/ragged/dense_r128_c129` | reduction | 128 × 129 | dense | correct | [passed](operator_peer/metal__f32__reduce_to_shape__ragged__dense_r128_c129.json) | 0.2945 / 0.2082 | [passed](program_peer/metal__f32__reduce_to_shape__ragged__dense_r128_c129.json) | 0.2897 / 0.2036 |
| `metal/f32/reduce_to_shape/ragged/offset_r16_c65` | reduction | 16 × 65 | offset | correct | [passed](operator_peer/metal__f32__reduce_to_shape__ragged__offset_r16_c65.json) | 0.2217 / 0.2 | [passed](program_peer/metal__f32__reduce_to_shape__ragged__offset_r16_c65.json) | 0.24 / 0.1966 |
| `metal/f32/reduce_to_shape/ragged/offset_r16_c129` | reduction | 16 × 129 | offset | correct | [passed](operator_peer/metal__f32__reduce_to_shape__ragged__offset_r16_c129.json) | 0.282 / 0.2012 | [passed](program_peer/metal__f32__reduce_to_shape__ragged__offset_r16_c129.json) | 0.3022 / 0.2054 |
| `metal/f32/reduce_to_shape/ragged/offset_r33_c32` | reduction | 33 × 32 | offset | correct | [passed](operator_peer/metal__f32__reduce_to_shape__ragged__offset_r33_c32.json) | 0.232 / 0.2007 | [passed](program_peer/metal__f32__reduce_to_shape__ragged__offset_r33_c32.json) | 0.2784 / 0.2059 |
| `metal/f32/reduce_to_shape/ragged/offset_r33_c65` | reduction | 33 × 65 | offset | correct | [passed](operator_peer/metal__f32__reduce_to_shape__ragged__offset_r33_c65.json) | 0.2448 / 0.2055 | [passed](program_peer/metal__f32__reduce_to_shape__ragged__offset_r33_c65.json) | 0.3072 / 0.2148 |
| `metal/f32/reduce_to_shape/ragged/offset_r33_c129` | reduction | 33 × 129 | offset | correct | [passed](operator_peer/metal__f32__reduce_to_shape__ragged__offset_r33_c129.json) | 0.2668 / 0.2012 | [passed](program_peer/metal__f32__reduce_to_shape__ragged__offset_r33_c129.json) | 0.3151 / 0.2098 |
| `metal/f32/reduce_to_shape/ragged/offset_r128_c32` | reduction | 128 × 32 | offset | correct | [passed](operator_peer/metal__f32__reduce_to_shape__ragged__offset_r128_c32.json) | 0.3494 / 0.2008 | [passed](program_peer/metal__f32__reduce_to_shape__ragged__offset_r128_c32.json) | 0.3592 / 0.2056 |
| `metal/f32/reduce_to_shape/ragged/offset_r128_c65` | reduction | 128 × 65 | offset | correct | [passed](operator_peer/metal__f32__reduce_to_shape__ragged__offset_r128_c65.json) | 0.3106 / 0.2073 | [passed](program_peer/metal__f32__reduce_to_shape__ragged__offset_r128_c65.json) | 0.2557 / 0.2118 |
| `metal/f32/reduce_to_shape/ragged/offset_r128_c129` | reduction | 128 × 129 | offset | correct | [passed](operator_peer/metal__f32__reduce_to_shape__ragged__offset_r128_c129.json) | 0.2959 / 0.2125 | [passed](program_peer/metal__f32__reduce_to_shape__ragged__offset_r128_c129.json) | 0.2926 / 0.2129 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
