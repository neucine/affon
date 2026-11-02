# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `1705b123a373d85429e11ee1ab05a4c0044f087318876c34daefb8207a9be476`  
Binary SHA-256: `c74e6e6cce5896f5d5539b09cd44755e2f3a07277750b94f7ab4c7a102c35d9a`  
Timing: 10 iterations × 5 samples × 3 rounds

## Checks

- Cases run: 3
- Numerical check passed: 3
- Operator check and timing passed: 3
- Program check and timing passed: 3
- Numerical failures: 0
- Operator failures: 0
- Program failures: 0

## Cases

| Case | Family | Shape | Layout | Numerical | Operator | Affon / PyTorch ms | Program | Affon / PyTorch ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `metal/f32/matmul/prefill` | matrix | 128 × 768 × 768 | dense | correct | [passed](operator_peer/metal__f32__matmul__prefill.json) | 1.453 / 1.384 | [passed](program_peer/metal__f32__matmul__prefill.json) | 3.066 / 1.453 |
| `metal/bf16/matmul/prefill/model_768` | matrix | 128 × 768 × 768 | dense | correct | [passed](operator_peer/metal__bf16__matmul__prefill__model_768.json) | 1.427 / 1.398 | [passed](program_peer/metal__bf16__matmul__prefill__model_768.json) | 1.464 / 1.386 |
| `metal/f32/matmul/decode_m1/m1_k128_n128_default` | matrix | 1 × 128 × 128 | dense | correct | [passed](operator_peer/metal__f32__matmul__decode_m1__m1_k128_n128_default.json) | 0.2436 / 0.2286 | [passed](program_peer/metal__f32__matmul__decode_m1__m1_k128_n128_default.json) | 0.3551 / 0.2038 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
