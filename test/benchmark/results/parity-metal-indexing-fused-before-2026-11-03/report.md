# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `c8e2ebe3c6856bff4a13527b3b4f386da8519402d325ec775dd121c97c766e1a`  
Binary SHA-256: `d24f90ccff39ef4c21d44538663e442ba03ac87bc73007a8fce8f0c49cdaab81`  
Timing: 10 iterations × 5 samples × 3 rounds

## Checks

- Cases run: 9
- Numerical check passed: 9
- Operator check and timing passed: 9
- Program check and timing passed: 9
- Numerical failures: 0
- Operator failures: 0
- Program failures: 0

## Cases

| Case | Family | Shape | Layout | Numerical | Operator | Affon / PyTorch ms | Program | Affon / PyTorch ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `metal/f32/index_select/small` | indexing | 8 × 16 | dense | correct | [passed](operator_peer/metal__f32__index_select__small.json) | 0.3887 / 0.2069 | [passed](program_peer/metal__f32__index_select__small.json) | 0.4123 / 0.2081 |
| `metal/f32/index_select/large` | indexing | 128 × 64 | dense | correct | [passed](operator_peer/metal__f32__index_select__large.json) | 0.3903 / 0.2244 | [passed](program_peer/metal__f32__index_select__large.json) | 0.4718 / 0.2328 |
| `metal/f32/embedding/small` | indexing | 8 × 16 | dense | correct | [passed](operator_peer/metal__f32__embedding__small.json) | 0.3602 / 0.2006 | [passed](program_peer/metal__f32__embedding__small.json) | 0.3607 / 0.2041 |
| `metal/f32/embedding/large` | indexing | 128 × 64 | dense | correct | [passed](operator_peer/metal__f32__embedding__large.json) | 0.4789 / 0.227 | [passed](program_peer/metal__f32__embedding__large.json) | 0.4019 / 0.2328 |
| `metal/f32/gather/small` | indexing | 4 × 8 | dense | correct | [passed](operator_peer/metal__f32__gather__small.json) | 0.358 / 0.2136 | [passed](program_peer/metal__f32__gather__small.json) | 0.375 / 0.2124 |
| `metal/f32/gather/large` | indexing | 32 × 64 | dense | correct | [passed](operator_peer/metal__f32__gather__large.json) | 0.4191 / 0.2316 | [passed](program_peer/metal__f32__gather__large.json) | 0.4219 / 0.2405 |
| `metal/bf16/index_select/strided` | indexing | 8 × 16 | strided | correct | [passed](operator_peer/metal__bf16__index_select__strided.json) | 0.3601 / 0.2077 | [passed](program_peer/metal__bf16__index_select__strided.json) | 0.3788 / 0.2138 |
| `metal/bf16/embedding/strided` | indexing | 8 × 16 | strided | correct | [passed](operator_peer/metal__bf16__embedding__strided.json) | 0.3425 / 0.1989 | [passed](program_peer/metal__bf16__embedding__strided.json) | 0.3757 / 0.2031 |
| `metal/bf16/gather/strided` | indexing | 4 × 8 | strided | correct | [passed](operator_peer/metal__bf16__gather__strided.json) | 0.3673 / 0.209 | [passed](program_peer/metal__bf16__gather__strided.json) | 0.3777 / 0.2126 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
