# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `32bbf4c5ea24a4fd7a07d38cd8b11313cdb1598ef773338d51d300634edea4ce`  
Binary SHA-256: `b7bf4a8c6b1ff9397eb19467936e6537c8d738710e6daccf8fb40f105439c5ae`  
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
| `metal/f32/variance_axis/strided/strided_r128_c65` | reduction | 128 × 65 | strided | correct | [passed](operator_peer/metal__f32__variance_axis__strided__strided_r128_c65.json) | 0.2531 / 0.2707 | [passed](program_peer/metal__f32__variance_axis__strided__strided_r128_c65.json) | 0.2577 / 0.2281 |
| `metal/f32/variance_axis/ragged/offset_r128_c65` | reduction | 128 × 65 | offset | correct | [passed](operator_peer/metal__f32__variance_axis__ragged__offset_r128_c65.json) | 0.2333 / 0.2081 | [passed](program_peer/metal__f32__variance_axis__ragged__offset_r128_c65.json) | 0.2591 / 0.2177 |
| `metal/f32/std_axis/strided/strided_r128_c65` | reduction | 128 × 65 | strided | correct | [passed](operator_peer/metal__f32__std_axis__strided__strided_r128_c65.json) | 0.2256 / 0.1964 | [passed](program_peer/metal__f32__std_axis__strided__strided_r128_c65.json) | 0.2425 / 0.2237 |
| `metal/f32/std_axis/ragged/offset_r128_c65` | reduction | 128 × 65 | offset | correct | [passed](operator_peer/metal__f32__std_axis__ragged__offset_r128_c65.json) | 0.2303 / 0.2011 | [passed](program_peer/metal__f32__std_axis__ragged__offset_r128_c65.json) | 0.2516 / 0.2132 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
