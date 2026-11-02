# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `deaace5f0077f7eb481a1d9786fa7f719d5827a197343410b15d6167ff225d92`  
Binary SHA-256: `e96991ca2b8a6024b51810fcb209036fa53b58aaefd00ba881e5f7c6d4e11d1d`  
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
| `metal/f32/sum_axis/outer_axis/r128_c64` | reduction | 128 × 64 | dense | correct | [passed](operator_peer/metal__f32__sum_axis__outer_axis__r128_c64.json) | 0.2376 / 0.2303 | [passed](program_peer/metal__f32__sum_axis__outer_axis__r128_c64.json) | 0.2192 / 0.2433 |
| `metal/f32/mean_axis/outer_axis/r128_c64` | reduction | 128 × 64 | dense | correct | [passed](operator_peer/metal__f32__mean_axis__outer_axis__r128_c64.json) | 0.2369 / 0.2274 | [passed](program_peer/metal__f32__mean_axis__outer_axis__r128_c64.json) | 0.2523 / 0.2529 |
| `metal/f32/sum_axis/strided/inner_strided_r129_c257` | reduction | 129 × 257 | strided | correct | [passed](operator_peer/metal__f32__sum_axis__strided__inner_strided_r129_c257.json) | 0.2425 / 0.2559 | [passed](program_peer/metal__f32__sum_axis__strided__inner_strided_r129_c257.json) | 0.2885 / 0.2151 |
| `metal/f32/mean_axis/strided/inner_strided_r129_c257` | reduction | 129 × 257 | strided | correct | [passed](operator_peer/metal__f32__mean_axis__strided__inner_strided_r129_c257.json) | 0.2403 / 0.2542 | [passed](program_peer/metal__f32__mean_axis__strided__inner_strided_r129_c257.json) | 0.2539 / 0.262 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
