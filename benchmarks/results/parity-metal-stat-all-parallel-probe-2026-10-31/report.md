# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `fa4d459bf961b0b7587fe3504ae0008c0f68e5b4693bee3f3b5768e96f1312bd`  
Binary SHA-256: `7b81a697ea697ba00f0dde5f0a0b6b8df74513955d5c5d5c62cac2da79d8bbce`  
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
| `metal/f32/variance_all/long_axis` | reduction | 8 × 512 | dense | correct | [passed](operator_peer/metal__f32__variance_all__long_axis.json) | 0.365 / 0.2167 | [passed](program_peer/metal__f32__variance_all__long_axis.json) | 0.3321 / 0.2035 |
| `metal/f32/variance_all/strided/r64_c127` | reduction | 64 × 127 | strided | correct | [passed](operator_peer/metal__f32__variance_all__strided__r64_c127.json) | 0.2912 / 0.246 | [passed](program_peer/metal__f32__variance_all__strided__r64_c127.json) | 0.292 / 0.2432 |
| `metal/f32/std_all/long_axis` | reduction | 8 × 512 | dense | correct | [passed](operator_peer/metal__f32__std_all__long_axis.json) | 0.2466 / 0.2088 | [passed](program_peer/metal__f32__std_all__long_axis.json) | 0.2477 / 0.2073 |
| `metal/f32/std_all/strided/r64_c127` | reduction | 64 × 127 | strided | correct | [passed](operator_peer/metal__f32__std_all__strided__r64_c127.json) | 0.2976 / 0.2278 | [passed](program_peer/metal__f32__std_all__strided__r64_c127.json) | 0.2964 / 0.2372 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
