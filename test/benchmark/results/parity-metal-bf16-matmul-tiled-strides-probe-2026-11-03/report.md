# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `6e1d3ec9ac09314fcf67bac726712409f06ca7bff58a09e19d7a0d320753419a`  
Binary SHA-256: `bb82bece2d6701e6cba06ae26b37eababedfc8fa29c28a6ab0f6db3bf5f52fcd`  
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
| `metal/bf16/matmul/prefill/model_768` | matrix | 128 × 768 × 768 | dense | correct | [passed](operator_peer/metal__bf16__matmul__prefill__model_768.json) | 1.429 / 1.523 | [passed](program_peer/metal__bf16__matmul__prefill__model_768.json) | 3.27 / 1.469 |
| `metal/bf16/matmul/strided/lhs_m64_k129_n17` | matrix | 64 × 129 × 17 | strided | correct | [passed](operator_peer/metal__bf16__matmul__strided__lhs_m64_k129_n17.json) | 0.2826 / 0.2678 | [passed](program_peer/metal__bf16__matmul__strided__lhs_m64_k129_n17.json) | 0.279 / 0.2809 |
| `metal/bf16/matmul/transposed/rhs_m65_k128_n64` | matrix | 65 × 128 × 64 | transposed | correct | [passed](operator_peer/metal__bf16__matmul__transposed__rhs_m65_k128_n64.json) | 0.3126 / 0.2706 | [passed](program_peer/metal__bf16__matmul__transposed__rhs_m65_k128_n64.json) | 0.3228 / 0.2983 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
