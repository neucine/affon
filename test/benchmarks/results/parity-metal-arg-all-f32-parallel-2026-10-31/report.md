# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `4d6acf06d9b74758475653e5d3211583baa4fb82cf2b5e2ab5f390aa7d3ee992`  
Binary SHA-256: `caa71ec6ea98005c433e270979a30a1f097a30c25835d18dc74edbe829c525a9`  
Timing: 10 iterations × 5 samples × 3 rounds

## Checks

- Cases run: 10
- Numerical check passed: 10
- Operator check and timing passed: 10
- Program check and timing passed: 10
- Numerical failures: 0
- Operator failures: 0
- Program failures: 0

## Cases

| Case | Family | Shape | Layout | Numerical | Operator | Affon / PyTorch ms | Program | Affon / PyTorch ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `metal/f32/argmin_all/short_axis` | reduction | 4 × 8 | dense | correct | [passed](operator_peer/metal__f32__argmin_all__short_axis.json) | 0.3561 / 0.2318 | [passed](program_peer/metal__f32__argmin_all__short_axis.json) | 0.2595 / 0.201 |
| `metal/f32/argmin_all/long_axis` | reduction | 8 × 512 | dense | correct | [passed](operator_peer/metal__f32__argmin_all__long_axis.json) | 0.2276 / 0.2104 | [passed](program_peer/metal__f32__argmin_all__long_axis.json) | 0.3361 / 0.2147 |
| `metal/f32/argmin_all/outer_axis` | reduction | 128 × 256 | dense | correct | [passed](operator_peer/metal__f32__argmin_all__outer_axis.json) | 0.3909 / 0.2098 | [passed](program_peer/metal__f32__argmin_all__outer_axis.json) | 0.3932 / 0.2175 |
| `metal/f32/argmin_all/strided` | reduction | 16 × 64 | strided | correct | [passed](operator_peer/metal__f32__argmin_all__strided.json) | 0.2553 / 0.2062 | [passed](program_peer/metal__f32__argmin_all__strided.json) | 0.2956 / 0.2113 |
| `metal/f32/argmin_all/ragged` | reduction | 17 × 65 | dense | correct | [passed](operator_peer/metal__f32__argmin_all__ragged.json) | 0.2956 / 0.2049 | [passed](program_peer/metal__f32__argmin_all__ragged.json) | 0.255 / 0.2093 |
| `metal/f32/argmax_all/short_axis` | reduction | 4 × 8 | dense | correct | [passed](operator_peer/metal__f32__argmax_all__short_axis.json) | 0.3324 / 0.2029 | [passed](program_peer/metal__f32__argmax_all__short_axis.json) | 0.2565 / 0.206 |
| `metal/f32/argmax_all/long_axis` | reduction | 8 × 512 | dense | correct | [passed](operator_peer/metal__f32__argmax_all__long_axis.json) | 0.25 / 0.2012 | [passed](program_peer/metal__f32__argmax_all__long_axis.json) | 0.2525 / 0.2064 |
| `metal/f32/argmax_all/outer_axis` | reduction | 128 × 256 | dense | correct | [passed](operator_peer/metal__f32__argmax_all__outer_axis.json) | 0.3807 / 0.2147 | [passed](program_peer/metal__f32__argmax_all__outer_axis.json) | 0.3871 / 0.2171 |
| `metal/f32/argmax_all/strided` | reduction | 16 × 64 | strided | correct | [passed](operator_peer/metal__f32__argmax_all__strided.json) | 0.2799 / 0.2029 | [passed](program_peer/metal__f32__argmax_all__strided.json) | 0.2884 / 0.2088 |
| `metal/f32/argmax_all/ragged` | reduction | 17 × 65 | dense | correct | [passed](operator_peer/metal__f32__argmax_all__ragged.json) | 0.2422 / 0.1955 | [passed](program_peer/metal__f32__argmax_all__ragged.json) | 0.2943 / 0.2071 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
