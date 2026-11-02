# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `c8e2ebe3c6856bff4a13527b3b4f386da8519402d325ec775dd121c97c766e1a`  
Binary SHA-256: `758f3755cc9bacc25744f4e74c9ec2670232ae07f065e8b28b3e53246dc50dac`  
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
| `metal/f32/index_select/small` | indexing | 8 × 16 | dense | correct | [passed](operator_peer/metal__f32__index_select__small.json) | 0.2389 / 0.2245 | [passed](program_peer/metal__f32__index_select__small.json) | 0.2319 / 0.2136 |
| `metal/f32/index_select/large` | indexing | 128 × 64 | dense | correct | [passed](operator_peer/metal__f32__index_select__large.json) | 0.2788 / 0.2574 | [passed](program_peer/metal__f32__index_select__large.json) | 0.2852 / 0.228 |
| `metal/f32/embedding/small` | indexing | 8 × 16 | dense | correct | [passed](operator_peer/metal__f32__embedding__small.json) | 0.2349 / 0.2151 | [passed](program_peer/metal__f32__embedding__small.json) | 0.2323 / 0.2146 |
| `metal/f32/embedding/large` | indexing | 128 × 64 | dense | correct | [passed](operator_peer/metal__f32__embedding__large.json) | 0.2712 / 0.2307 | [passed](program_peer/metal__f32__embedding__large.json) | 0.304 / 0.2253 |
| `metal/f32/gather/small` | indexing | 4 × 8 | dense | correct | [passed](operator_peer/metal__f32__gather__small.json) | 0.2349 / 0.2364 | [passed](program_peer/metal__f32__gather__small.json) | 0.2496 / 0.2172 |
| `metal/f32/gather/large` | indexing | 32 × 64 | dense | correct | [passed](operator_peer/metal__f32__gather__large.json) | 0.2789 / 0.2661 | [passed](program_peer/metal__f32__gather__large.json) | 0.2925 / 0.2341 |
| `metal/bf16/index_select/strided` | indexing | 8 × 16 | strided | correct | [passed](operator_peer/metal__bf16__index_select__strided.json) | 0.2412 / 0.2301 | [passed](program_peer/metal__bf16__index_select__strided.json) | 0.2647 / 0.2106 |
| `metal/bf16/embedding/strided` | indexing | 8 × 16 | strided | correct | [passed](operator_peer/metal__bf16__embedding__strided.json) | 0.1922 / 0.2036 | [passed](program_peer/metal__bf16__embedding__strided.json) | 0.2692 / 0.2133 |
| `metal/bf16/gather/strided` | indexing | 4 × 8 | strided | correct | [passed](operator_peer/metal__bf16__gather__strided.json) | 0.2297 / 0.241 | [passed](program_peer/metal__bf16__gather__strided.json) | 0.2642 / 0.2174 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
