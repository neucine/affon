# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `c8e2ebe3c6856bff4a13527b3b4f386da8519402d325ec775dd121c97c766e1a`  
Binary SHA-256: `6c3f47a626a96b3d0ac3b009cfb85db3f6ebe8416e9751d0da0ec04092b02a4d`  
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
| `metal/f32/index_select/small` | indexing | 8 × 16 | dense | correct | [passed](operator_peer/metal__f32__index_select__small.json) | 0.2361 / 0.2267 | [passed](program_peer/metal__f32__index_select__small.json) | 0.4185 / 0.238 |
| `metal/f32/index_select/large` | indexing | 128 × 64 | dense | correct | [passed](operator_peer/metal__f32__index_select__large.json) | 0.2594 / 0.2329 | [passed](program_peer/metal__f32__index_select__large.json) | 0.4621 / 0.2694 |
| `metal/f32/embedding/small` | indexing | 8 × 16 | dense | correct | [passed](operator_peer/metal__f32__embedding__small.json) | 0.2354 / 0.2195 | [passed](program_peer/metal__f32__embedding__small.json) | 0.3926 / 0.2262 |
| `metal/f32/embedding/large` | indexing | 128 × 64 | dense | correct | [passed](operator_peer/metal__f32__embedding__large.json) | 0.2837 / 0.2517 | [passed](program_peer/metal__f32__embedding__large.json) | 0.4503 / 0.2695 |
| `metal/f32/gather/small` | indexing | 4 × 8 | dense | correct | [passed](operator_peer/metal__f32__gather__small.json) | 0.2371 / 0.2387 | [passed](program_peer/metal__f32__gather__small.json) | 0.407 / 0.2501 |
| `metal/f32/gather/large` | indexing | 32 × 64 | dense | correct | [passed](operator_peer/metal__f32__gather__large.json) | 0.2812 / 0.2691 | [passed](program_peer/metal__f32__gather__large.json) | 0.4605 / 0.2907 |
| `metal/bf16/index_select/strided` | indexing | 8 × 16 | strided | correct | [passed](operator_peer/metal__bf16__index_select__strided.json) | 0.2378 / 0.2487 | [passed](program_peer/metal__bf16__index_select__strided.json) | 0.4459 / 0.2607 |
| `metal/bf16/embedding/strided` | indexing | 8 × 16 | strided | correct | [passed](operator_peer/metal__bf16__embedding__strided.json) | 0.2386 / 0.2462 | [passed](program_peer/metal__bf16__embedding__strided.json) | 0.4402 / 0.2684 |
| `metal/bf16/gather/strided` | indexing | 4 × 8 | strided | correct | [passed](operator_peer/metal__bf16__gather__strided.json) | 0.2354 / 0.2474 | [passed](program_peer/metal__bf16__gather__strided.json) | 0.4477 / 0.2571 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
