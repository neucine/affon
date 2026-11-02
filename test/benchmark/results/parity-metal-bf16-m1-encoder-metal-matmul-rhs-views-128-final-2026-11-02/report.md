# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `950550b1938839a6a8ebbf9d8f656321ae989236f618f4ac3ca14f9120e037b8`  
Binary SHA-256: `8522698ce689fb48c552a41c4b152287fbec36f469250cc8d198a9b99cd03a8d`  
Timing: 30 iterations × 5 samples × 3 rounds

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
| `metal/bf16/matmul/transposed/rhs_m1_k7_n17` | matrix | 1 × 7 × 17 | transposed | correct | [passed](operator_peer/metal__bf16__matmul__transposed__rhs_m1_k7_n17.json) | 0.2122 / 0.1993 | [passed](program_peer/metal__bf16__matmul__transposed__rhs_m1_k7_n17.json) | 0.2048 / 0.2067 |
| `metal/bf16/matmul/transposed/rhs_m1_k7_n65` | matrix | 1 × 7 × 65 | transposed | correct | [passed](operator_peer/metal__bf16__matmul__transposed__rhs_m1_k7_n65.json) | 0.1968 / 0.2057 | [passed](program_peer/metal__bf16__matmul__transposed__rhs_m1_k7_n65.json) | 0.1855 / 0.2041 |
| `metal/bf16/matmul/transposed/rhs_m1_k31_n17` | matrix | 1 × 31 × 17 | transposed | correct | [passed](operator_peer/metal__bf16__matmul__transposed__rhs_m1_k31_n17.json) | 0.189 / 0.2043 | [passed](program_peer/metal__bf16__matmul__transposed__rhs_m1_k31_n17.json) | 0.1932 / 0.2069 |
| `metal/bf16/matmul/transposed/rhs_m1_k31_n65` | matrix | 1 × 31 × 65 | transposed | correct | [passed](operator_peer/metal__bf16__matmul__transposed__rhs_m1_k31_n65.json) | 0.1807 / 0.1973 | [passed](program_peer/metal__bf16__matmul__transposed__rhs_m1_k31_n65.json) | 0.1922 / 0.2046 |
| `metal/bf16/matmul/transposed/rhs_m1_k64_n17` | matrix | 1 × 64 × 17 | transposed | correct | [passed](operator_peer/metal__bf16__matmul__transposed__rhs_m1_k64_n17.json) | 0.1846 / 0.2087 | [passed](program_peer/metal__bf16__matmul__transposed__rhs_m1_k64_n17.json) | 0.2179 / 0.2066 |
| `metal/bf16/matmul/transposed/rhs_m1_k64_n65` | matrix | 1 × 64 × 65 | transposed | correct | [passed](operator_peer/metal__bf16__matmul__transposed__rhs_m1_k64_n65.json) | 0.1898 / 0.2057 | [passed](program_peer/metal__bf16__matmul__transposed__rhs_m1_k64_n65.json) | 0.2124 / 0.2045 |
| `metal/bf16/matmul/strided/rhs_m1_k7_n17` | matrix | 1 × 7 × 17 | strided | correct | [passed](operator_peer/metal__bf16__matmul__strided__rhs_m1_k7_n17.json) | 0.1809 / 0.2148 | [passed](program_peer/metal__bf16__matmul__strided__rhs_m1_k7_n17.json) | 0.2028 / 0.2164 |
| `metal/bf16/matmul/strided/rhs_m1_k7_n65` | matrix | 1 × 7 × 65 | strided | correct | [passed](operator_peer/metal__bf16__matmul__strided__rhs_m1_k7_n65.json) | 0.1749 / 0.2145 | [passed](program_peer/metal__bf16__matmul__strided__rhs_m1_k7_n65.json) | 0.1961 / 0.1991 |
| `metal/bf16/matmul/strided/rhs_m1_k31_n17` | matrix | 1 × 31 × 17 | strided | correct | [passed](operator_peer/metal__bf16__matmul__strided__rhs_m1_k31_n17.json) | 0.2002 / 0.2158 | [passed](program_peer/metal__bf16__matmul__strided__rhs_m1_k31_n17.json) | 0.216 / 0.2147 |
| `metal/bf16/matmul/strided/rhs_m1_k31_n65` | matrix | 1 × 31 × 65 | strided | correct | [passed](operator_peer/metal__bf16__matmul__strided__rhs_m1_k31_n65.json) | 0.1912 / 0.2126 | [passed](program_peer/metal__bf16__matmul__strided__rhs_m1_k31_n65.json) | 0.2087 / 0.2146 |
| `metal/bf16/matmul/strided/rhs_m1_k64_n17` | matrix | 1 × 64 × 17 | strided | correct | [passed](operator_peer/metal__bf16__matmul__strided__rhs_m1_k64_n17.json) | 0.2191 / 0.2039 | [passed](program_peer/metal__bf16__matmul__strided__rhs_m1_k64_n17.json) | 0.2318 / 0.2004 |
| `metal/bf16/matmul/strided/rhs_m1_k64_n65` | matrix | 1 × 64 × 65 | strided | correct | [passed](operator_peer/metal__bf16__matmul__strided__rhs_m1_k64_n65.json) | 0.2263 / 0.2092 | [passed](program_peer/metal__bf16__matmul__strided__rhs_m1_k64_n65.json) | 0.2375 / 0.214 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
