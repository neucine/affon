# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `d25dbf6e3622f0579ceb1b6e52f1c6863151bff4e2dcd1db799b4f16068673b3`  
Binary SHA-256: `c265a163bdb6745717ab2527586c650058f307a8ce2779e718c638de472b7dc2`  
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
| `metal/f32/argmax_axis/outer_axis/r128_c64` | reduction | 128 × 64 | dense | correct | [passed](operator_peer/metal__f32__argmax_axis__outer_axis__r128_c64.json) | 0.2403 / 0.2535 | [passed](program_peer/metal__f32__argmax_axis__outer_axis__r128_c64.json) | 0.2644 / 0.2071 |
| `metal/f32/argmin_axis/outer_axis/r128_c64` | reduction | 128 × 64 | dense | correct | [passed](operator_peer/metal__f32__argmin_axis__outer_axis__r128_c64.json) | 0.2349 / 0.2548 | [passed](program_peer/metal__f32__argmin_axis__outer_axis__r128_c64.json) | 0.2764 / 0.2097 |
| `metal/f32/argmax_axis/strided/r64_c127` | reduction | 64 × 127 | strided | correct | [passed](operator_peer/metal__f32__argmax_axis__strided__r64_c127.json) | 0.2478 / 0.2594 | [passed](program_peer/metal__f32__argmax_axis__strided__r64_c127.json) | 0.265 / 0.2184 |
| `metal/f32/argmin_axis/strided/r64_c127` | reduction | 64 × 127 | strided | correct | [passed](operator_peer/metal__f32__argmin_axis__strided__r64_c127.json) | 0.2614 / 0.2092 | [passed](program_peer/metal__f32__argmin_axis__strided__r64_c127.json) | 0.255 / 0.217 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
