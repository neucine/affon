# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `d94cfbb7160b559144803594e4a7423ba8940e70d9b47476780de0a50ac73fd0`  
Binary SHA-256: `c74e6e6cce5896f5d5539b09cd44755e2f3a07277750b94f7ab4c7a102c35d9a`  
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
| `metal/f32/matmul/prefill` | matrix | 128 × 768 × 768 | dense | correct | [passed](operator_peer/metal__f32__matmul__prefill.json) | 1.839 / 1.528 | [passed](program_peer/metal__f32__matmul__prefill.json) | 4.641 / 1.552 |
| `metal/bf16/matmul/prefill/model_768` | matrix | 128 × 768 × 768 | dense | correct | [passed](operator_peer/metal__bf16__matmul__prefill__model_768.json) | 2.933 / 1.493 | [passed](program_peer/metal__bf16__matmul__prefill__model_768.json) | 3.029 / 1.581 |
| `metal/f32/contiguous/materialized` | structural | 16 × 32 | dense | correct | [passed](operator_peer/metal__f32__contiguous__materialized.json) | 0.2425 / 0.2098 | [passed](program_peer/metal__f32__contiguous__materialized.json) | 0.2515 / 0.2105 |
| `metal/f32/matmul/decode_m1/m1_k128_n128_default` | matrix | 1 × 128 × 128 | dense | correct | [passed](operator_peer/metal__f32__matmul__decode_m1__m1_k128_n128_default.json) | 0.2729 / 0.2153 | [passed](program_peer/metal__f32__matmul__decode_m1__m1_k128_n128_default.json) | 0.3193 / 0.2257 |
| `metal/f32/contiguous/strided/r64_c127` | structural | 64 × 127 | strided | correct | [passed](operator_peer/metal__f32__contiguous__strided__r64_c127.json) | 0.3677 / 0.2931 | [passed](program_peer/metal__f32__contiguous__strided__r64_c127.json) | 0.3766 / 0.3013 |
| `metal/bf16/contiguous/ragged/r33_c129` | structural | 33 × 129 | dense | correct | [passed](operator_peer/metal__bf16__contiguous__ragged__r33_c129.json) | 0.3358 / 0.2465 | [passed](program_peer/metal__bf16__contiguous__ragged__r33_c129.json) | 0.363 / 0.2595 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
