# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `d94cfbb7160b559144803594e4a7423ba8940e70d9b47476780de0a50ac73fd0`  
Binary SHA-256: `d24f90ccff39ef4c21d44538663e442ba03ac87bc73007a8fce8f0c49cdaab81`  
Timing: 10 iterations × 5 samples × 3 rounds

## Checks

- Cases run: 6
- Numerical check passed: 6
- Operator check and timing passed: 6
- Program check and timing passed: 6
- Numerical failures: 0
- Operator failures: 0
- Program failures: 0

## Cases

| Case | Family | Shape | Layout | Numerical | Operator | Affon / PyTorch ms | Program | Affon / PyTorch ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `metal/f32/matmul/prefill` | matrix | 128 × 768 × 768 | dense | correct | [passed](operator_peer/metal__f32__matmul__prefill.json) | 1.429 / 1.413 | [passed](program_peer/metal__f32__matmul__prefill.json) | 4.546 / 1.571 |
| `metal/bf16/matmul/prefill/model_768` | matrix | 128 × 768 × 768 | dense | correct | [passed](operator_peer/metal__bf16__matmul__prefill__model_768.json) | 1.395 / 1.367 | [passed](program_peer/metal__bf16__matmul__prefill__model_768.json) | 1.45 / 1.517 |
| `metal/f32/contiguous/materialized` | structural | 16 × 32 | dense | correct | [passed](operator_peer/metal__f32__contiguous__materialized.json) | 0.238 / 0.1813 | [passed](program_peer/metal__f32__contiguous__materialized.json) | 0.2655 / 0.1953 |
| `metal/f32/matmul/decode_m1/m1_k128_n128_default` | matrix | 1 × 128 × 128 | dense | correct | [passed](operator_peer/metal__f32__matmul__decode_m1__m1_k128_n128_default.json) | 0.2386 / 0.1811 | [passed](program_peer/metal__f32__matmul__decode_m1__m1_k128_n128_default.json) | 0.2658 / 0.2065 |
| `metal/f32/contiguous/strided/r64_c127` | structural | 64 × 127 | strided | correct | [passed](operator_peer/metal__f32__contiguous__strided__r64_c127.json) | 0.3495 / 0.2613 | [passed](program_peer/metal__f32__contiguous__strided__r64_c127.json) | 0.3799 / 0.2733 |
| `metal/bf16/contiguous/ragged/r33_c129` | structural | 33 × 129 | dense | correct | [passed](operator_peer/metal__bf16__contiguous__ragged__r33_c129.json) | 0.3166 / 0.2264 | [passed](program_peer/metal__bf16__contiguous__ragged__r33_c129.json) | 0.3703 / 0.232 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
