# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `950550b1938839a6a8ebbf9d8f656321ae989236f618f4ac3ca14f9120e037b8`  
Binary SHA-256: `a9a10dd4a4dd6ab577715a114b1261d475aeddc676e7750489ffff1b83de8cef`  
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
| `metal/bf16/matmul/transposed/rhs_m1_k7_n17` | matrix | 1 × 7 × 17 | transposed | correct | [passed](operator_peer/metal__bf16__matmul__transposed__rhs_m1_k7_n17.json) | 0.2486 / 0.2358 | [passed](program_peer/metal__bf16__matmul__transposed__rhs_m1_k7_n17.json) | 0.2594 / 0.2086 |
| `metal/bf16/matmul/transposed/rhs_m1_k7_n65` | matrix | 1 × 7 × 65 | transposed | correct | [passed](operator_peer/metal__bf16__matmul__transposed__rhs_m1_k7_n65.json) | 0.2484 / 0.2049 | [passed](program_peer/metal__bf16__matmul__transposed__rhs_m1_k7_n65.json) | 0.2584 / 0.1944 |
| `metal/bf16/matmul/transposed/rhs_m1_k31_n17` | matrix | 1 × 31 × 17 | transposed | correct | [passed](operator_peer/metal__bf16__matmul__transposed__rhs_m1_k31_n17.json) | 0.2572 / 0.2007 | [passed](program_peer/metal__bf16__matmul__transposed__rhs_m1_k31_n17.json) | 0.2832 / 0.2047 |
| `metal/bf16/matmul/transposed/rhs_m1_k31_n65` | matrix | 1 × 31 × 65 | transposed | correct | [passed](operator_peer/metal__bf16__matmul__transposed__rhs_m1_k31_n65.json) | 0.262 / 0.2059 | [passed](program_peer/metal__bf16__matmul__transposed__rhs_m1_k31_n65.json) | 0.2904 / 0.1973 |
| `metal/bf16/matmul/transposed/rhs_m1_k64_n17` | matrix | 1 × 64 × 17 | transposed | correct | [passed](operator_peer/metal__bf16__matmul__transposed__rhs_m1_k64_n17.json) | 0.2962 / 0.2071 | [passed](program_peer/metal__bf16__matmul__transposed__rhs_m1_k64_n17.json) | 0.3088 / 0.1982 |
| `metal/bf16/matmul/transposed/rhs_m1_k64_n65` | matrix | 1 × 64 × 65 | transposed | correct | [passed](operator_peer/metal__bf16__matmul__transposed__rhs_m1_k64_n65.json) | 0.2338 / 0.1967 | [passed](program_peer/metal__bf16__matmul__transposed__rhs_m1_k64_n65.json) | 0.3088 / 0.209 |
| `metal/bf16/matmul/strided/rhs_m1_k7_n17` | matrix | 1 × 7 × 17 | strided | correct | [passed](operator_peer/metal__bf16__matmul__strided__rhs_m1_k7_n17.json) | 0.2448 / 0.202 | [passed](program_peer/metal__bf16__matmul__strided__rhs_m1_k7_n17.json) | 0.265 / 0.2168 |
| `metal/bf16/matmul/strided/rhs_m1_k7_n65` | matrix | 1 × 7 × 65 | strided | correct | [passed](operator_peer/metal__bf16__matmul__strided__rhs_m1_k7_n65.json) | 0.2528 / 0.2042 | [passed](program_peer/metal__bf16__matmul__strided__rhs_m1_k7_n65.json) | 0.265 / 0.2151 |
| `metal/bf16/matmul/strided/rhs_m1_k31_n17` | matrix | 1 × 31 × 17 | strided | correct | [passed](operator_peer/metal__bf16__matmul__strided__rhs_m1_k31_n17.json) | 0.2846 / 0.2075 | [passed](program_peer/metal__bf16__matmul__strided__rhs_m1_k31_n17.json) | 0.3072 / 0.214 |
| `metal/bf16/matmul/strided/rhs_m1_k31_n65` | matrix | 1 × 31 × 65 | strided | correct | [passed](operator_peer/metal__bf16__matmul__strided__rhs_m1_k31_n65.json) | 0.2942 / 0.212 | [passed](program_peer/metal__bf16__matmul__strided__rhs_m1_k31_n65.json) | 0.3046 / 0.221 |
| `metal/bf16/matmul/strided/rhs_m1_k64_n17` | matrix | 1 × 64 × 17 | strided | correct | [passed](operator_peer/metal__bf16__matmul__strided__rhs_m1_k64_n17.json) | 0.2488 / 0.2087 | [passed](program_peer/metal__bf16__matmul__strided__rhs_m1_k64_n17.json) | 0.3314 / 0.2135 |
| `metal/bf16/matmul/strided/rhs_m1_k64_n65` | matrix | 1 × 64 × 65 | strided | correct | [passed](operator_peer/metal__bf16__matmul__strided__rhs_m1_k64_n65.json) | 0.3182 / 0.2119 | [passed](program_peer/metal__bf16__matmul__strided__rhs_m1_k64_n65.json) | 0.3292 / 0.2159 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
