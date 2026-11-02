# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `950550b1938839a6a8ebbf9d8f656321ae989236f618f4ac3ca14f9120e037b8`  
Binary SHA-256: `8522698ce689fb48c552a41c4b152287fbec36f469250cc8d198a9b99cd03a8d`  
Timing: 5 iterations × 5 samples × 3 rounds

## Checks

- Cases run: 12
- Numerical check passed: 12
- Operator check and timing passed: 12
- Program check and timing passed: 12
- Numerical failures: 0
- Operator failures: 0
- Program failures: 0

## Cases

| Case | Family | Shape | Layout | Numerical | Operator | Affon / PyTorch ms | Program | Affon / PyTorch ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `metal/bf16/matmul/transposed/rhs_m1_k7_n17` | matrix | 1 × 7 × 17 | transposed | correct | [passed](operator_peer/metal__bf16__matmul__transposed__rhs_m1_k7_n17.json) | 0.221 / 0.2059 | [passed](program_peer/metal__bf16__matmul__transposed__rhs_m1_k7_n17.json) | 0.251 / 0.2048 |
| `metal/bf16/matmul/transposed/rhs_m1_k7_n65` | matrix | 1 × 7 × 65 | transposed | correct | [passed](operator_peer/metal__bf16__matmul__transposed__rhs_m1_k7_n65.json) | 0.2194 / 0.206 | [passed](program_peer/metal__bf16__matmul__transposed__rhs_m1_k7_n65.json) | 0.2538 / 0.2042 |
| `metal/bf16/matmul/transposed/rhs_m1_k31_n17` | matrix | 1 × 31 × 17 | transposed | correct | [passed](operator_peer/metal__bf16__matmul__transposed__rhs_m1_k31_n17.json) | 0.2376 / 0.1952 | [passed](program_peer/metal__bf16__matmul__transposed__rhs_m1_k31_n17.json) | 0.2606 / 0.208 |
| `metal/bf16/matmul/transposed/rhs_m1_k31_n65` | matrix | 1 × 31 × 65 | transposed | correct | [passed](operator_peer/metal__bf16__matmul__transposed__rhs_m1_k31_n65.json) | 0.202 / 0.2088 | [passed](program_peer/metal__bf16__matmul__transposed__rhs_m1_k31_n65.json) | 0.2684 / 0.2076 |
| `metal/bf16/matmul/transposed/rhs_m1_k64_n17` | matrix | 1 × 64 × 17 | transposed | correct | [passed](operator_peer/metal__bf16__matmul__transposed__rhs_m1_k64_n17.json) | 0.2318 / 0.2021 | [passed](program_peer/metal__bf16__matmul__transposed__rhs_m1_k64_n17.json) | 0.299 / 0.2046 |
| `metal/bf16/matmul/transposed/rhs_m1_k64_n65` | matrix | 1 × 64 × 65 | transposed | correct | [passed](operator_peer/metal__bf16__matmul__transposed__rhs_m1_k64_n65.json) | 0.2928 / 0.2093 | [passed](program_peer/metal__bf16__matmul__transposed__rhs_m1_k64_n65.json) | 0.2958 / 0.2269 |
| `metal/bf16/matmul/strided/rhs_m1_k7_n17` | matrix | 1 × 7 × 17 | strided | correct | [passed](operator_peer/metal__bf16__matmul__strided__rhs_m1_k7_n17.json) | 0.2406 / 0.2122 | [passed](program_peer/metal__bf16__matmul__strided__rhs_m1_k7_n17.json) | 0.2062 / 0.214 |
| `metal/bf16/matmul/strided/rhs_m1_k7_n65` | matrix | 1 × 7 × 65 | strided | correct | [passed](operator_peer/metal__bf16__matmul__strided__rhs_m1_k7_n65.json) | 0.2444 / 0.2092 | [passed](program_peer/metal__bf16__matmul__strided__rhs_m1_k7_n65.json) | 0.2508 / 0.2107 |
| `metal/bf16/matmul/strided/rhs_m1_k31_n17` | matrix | 1 × 31 × 17 | strided | correct | [passed](operator_peer/metal__bf16__matmul__strided__rhs_m1_k31_n17.json) | 0.2792 / 0.2104 | [passed](program_peer/metal__bf16__matmul__strided__rhs_m1_k31_n17.json) | 0.2894 / 0.2176 |
| `metal/bf16/matmul/strided/rhs_m1_k31_n65` | matrix | 1 × 31 × 65 | strided | correct | [passed](operator_peer/metal__bf16__matmul__strided__rhs_m1_k31_n65.json) | 0.2662 / 0.2094 | [passed](program_peer/metal__bf16__matmul__strided__rhs_m1_k31_n65.json) | 0.2966 / 0.2131 |
| `metal/bf16/matmul/strided/rhs_m1_k64_n17` | matrix | 1 × 64 × 17 | strided | correct | [passed](operator_peer/metal__bf16__matmul__strided__rhs_m1_k64_n17.json) | 0.2454 / 0.2102 | [passed](program_peer/metal__bf16__matmul__strided__rhs_m1_k64_n17.json) | 0.3214 / 0.2135 |
| `metal/bf16/matmul/strided/rhs_m1_k64_n65` | matrix | 1 × 64 × 65 | strided | correct | [passed](operator_peer/metal__bf16__matmul__strided__rhs_m1_k64_n65.json) | 0.3152 / 0.2129 | [passed](program_peer/metal__bf16__matmul__strided__rhs_m1_k64_n65.json) | 0.3136 / 0.2026 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
