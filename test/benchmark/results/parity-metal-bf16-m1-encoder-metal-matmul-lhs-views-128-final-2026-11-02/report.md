# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `45ff34dc1f1c8c897fe069a697b5880c2271992a2bc152209a5b75e6cfd88328`  
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
| `metal/bf16/matmul/transposed/lhs_m1_k7_n17` | matrix | 1 × 7 × 17 | transposed | correct | [passed](operator_peer/metal__bf16__matmul__transposed__lhs_m1_k7_n17.json) | 0.182 / 0.2094 | [passed](program_peer/metal__bf16__matmul__transposed__lhs_m1_k7_n17.json) | 0.1873 / 0.2093 |
| `metal/bf16/matmul/transposed/lhs_m1_k7_n65` | matrix | 1 × 7 × 65 | transposed | correct | [passed](operator_peer/metal__bf16__matmul__transposed__lhs_m1_k7_n65.json) | 0.1799 / 0.2042 | [passed](program_peer/metal__bf16__matmul__transposed__lhs_m1_k7_n65.json) | 0.1877 / 0.2078 |
| `metal/bf16/matmul/transposed/lhs_m1_k31_n17` | matrix | 1 × 31 × 17 | transposed | correct | [passed](operator_peer/metal__bf16__matmul__transposed__lhs_m1_k31_n17.json) | 0.1941 / 0.2074 | [passed](program_peer/metal__bf16__matmul__transposed__lhs_m1_k31_n17.json) | 0.205 / 0.2073 |
| `metal/bf16/matmul/transposed/lhs_m1_k31_n65` | matrix | 1 × 31 × 65 | transposed | correct | [passed](operator_peer/metal__bf16__matmul__transposed__lhs_m1_k31_n65.json) | 0.1937 / 0.2089 | [passed](program_peer/metal__bf16__matmul__transposed__lhs_m1_k31_n65.json) | 0.2095 / 0.2054 |
| `metal/bf16/matmul/transposed/lhs_m1_k64_n17` | matrix | 1 × 64 × 17 | transposed | correct | [passed](operator_peer/metal__bf16__matmul__transposed__lhs_m1_k64_n17.json) | 0.2204 / 0.2035 | [passed](program_peer/metal__bf16__matmul__transposed__lhs_m1_k64_n17.json) | 0.2301 / 0.2082 |
| `metal/bf16/matmul/transposed/lhs_m1_k64_n65` | matrix | 1 × 64 × 65 | transposed | correct | [passed](operator_peer/metal__bf16__matmul__transposed__lhs_m1_k64_n65.json) | 0.2297 / 0.2059 | [passed](program_peer/metal__bf16__matmul__transposed__lhs_m1_k64_n65.json) | 0.2355 / 0.2077 |
| `metal/bf16/matmul/strided/lhs_m1_k7_n17` | matrix | 1 × 7 × 17 | strided | correct | [passed](operator_peer/metal__bf16__matmul__strided__lhs_m1_k7_n17.json) | 0.1921 / 0.2097 | [passed](program_peer/metal__bf16__matmul__strided__lhs_m1_k7_n17.json) | 0.1846 / 0.2104 |
| `metal/bf16/matmul/strided/lhs_m1_k7_n65` | matrix | 1 × 7 × 65 | strided | correct | [passed](operator_peer/metal__bf16__matmul__strided__lhs_m1_k7_n65.json) | 0.2086 / 0.2104 | [passed](program_peer/metal__bf16__matmul__strided__lhs_m1_k7_n65.json) | 0.1869 / 0.2122 |
| `metal/bf16/matmul/strided/lhs_m1_k31_n17` | matrix | 1 × 31 × 17 | strided | correct | [passed](operator_peer/metal__bf16__matmul__strided__lhs_m1_k31_n17.json) | 0.1895 / 0.2078 | [passed](program_peer/metal__bf16__matmul__strided__lhs_m1_k31_n17.json) | 0.2027 / 0.2081 |
| `metal/bf16/matmul/strided/lhs_m1_k31_n65` | matrix | 1 × 31 × 65 | strided | correct | [passed](operator_peer/metal__bf16__matmul__strided__lhs_m1_k31_n65.json) | 0.2002 / 0.2096 | [passed](program_peer/metal__bf16__matmul__strided__lhs_m1_k31_n65.json) | 0.214 / 0.2071 |
| `metal/bf16/matmul/strided/lhs_m1_k64_n17` | matrix | 1 × 64 × 17 | strided | correct | [passed](operator_peer/metal__bf16__matmul__strided__lhs_m1_k64_n17.json) | 0.2267 / 0.2039 | [passed](program_peer/metal__bf16__matmul__strided__lhs_m1_k64_n17.json) | 0.2299 / 0.2109 |
| `metal/bf16/matmul/strided/lhs_m1_k64_n65` | matrix | 1 × 64 × 65 | strided | correct | [passed](operator_peer/metal__bf16__matmul__strided__lhs_m1_k64_n65.json) | 0.2259 / 0.2077 | [passed](program_peer/metal__bf16__matmul__strided__lhs_m1_k64_n65.json) | 0.2374 / 0.2065 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
