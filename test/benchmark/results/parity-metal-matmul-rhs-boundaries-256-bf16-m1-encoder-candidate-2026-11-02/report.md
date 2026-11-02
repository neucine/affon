# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `3c1f6d8774d7dcbbce43b92f9c8a6c44300d325d1483add63449461c430db04b`  
Binary SHA-256: `8522698ce689fb48c552a41c4b152287fbec36f469250cc8d198a9b99cd03a8d`  
Timing: 30 iterations × 5 samples × 3 rounds

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
| `metal/bf16/matmul/transposed/rhs_m1_k8_n8` | matrix | 1 × 8 × 8 | transposed | correct | [passed](operator_peer/metal__bf16__matmul__transposed__rhs_m1_k8_n8.json) | 0.1951 / 0.2063 | [passed](program_peer/metal__bf16__matmul__transposed__rhs_m1_k8_n8.json) | 0.1938 / 0.2092 |
| `metal/bf16/matmul/transposed/rhs_m1_k63_n64` | matrix | 1 × 63 × 64 | transposed | correct | [passed](operator_peer/metal__bf16__matmul__transposed__rhs_m1_k63_n64.json) | 0.2234 / 0.1968 | [passed](program_peer/metal__bf16__matmul__transposed__rhs_m1_k63_n64.json) | 0.2355 / 0.2057 |
| `metal/bf16/matmul/strided/rhs_m1_k32_n32` | matrix | 1 × 32 × 32 | strided | correct | [passed](operator_peer/metal__bf16__matmul__strided__rhs_m1_k32_n32.json) | 0.2086 / 0.2082 | [passed](program_peer/metal__bf16__matmul__strided__rhs_m1_k32_n32.json) | 0.234 / 0.21 |
| `metal/bf16/matmul/strided/rhs_m1_k128_n128` | matrix | 1 × 128 × 128 | strided | correct | [passed](operator_peer/metal__bf16__matmul__strided__rhs_m1_k128_n128.json) | 0.2375 / 0.2126 | [passed](program_peer/metal__bf16__matmul__strided__rhs_m1_k128_n128.json) | 0.2196 / 0.2132 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
