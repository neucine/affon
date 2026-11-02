# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `f1b92c340847b9bddae6749ef94e920b428923c3eabb83903bc407c4ab3577f0`  
Binary SHA-256: `8522698ce689fb48c552a41c4b152287fbec36f469250cc8d198a9b99cd03a8d`  
Timing: 30 iterations × 5 samples × 3 rounds

## Checks

- Cases run: 16
- Numerical check passed: 16
- Operator check and timing passed: 16
- Program check and timing passed: 16
- Numerical failures: 0
- Operator failures: 0
- Program failures: 0

## Cases

| Case | Family | Shape | Layout | Numerical | Operator | Affon / PyTorch ms | Program | Affon / PyTorch ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `metal/bf16/matmul/transposed/lhs_m1_k7_n17` | matrix | 1 × 7 × 17 | transposed | correct | [passed](operator_peer/metal__bf16__matmul__transposed__lhs_m1_k7_n17.json) | 0.2046 / 0.2025 | [passed](program_peer/metal__bf16__matmul__transposed__lhs_m1_k7_n17.json) | 0.1937 / 0.2051 |
| `metal/bf16/matmul/transposed/lhs_m1_k7_n65` | matrix | 1 × 7 × 65 | transposed | correct | [passed](operator_peer/metal__bf16__matmul__transposed__lhs_m1_k7_n65.json) | 0.1937 / 0.2097 | [passed](program_peer/metal__bf16__matmul__transposed__lhs_m1_k7_n65.json) | 0.1857 / 0.2099 |
| `metal/bf16/matmul/transposed/lhs_m1_k31_n17` | matrix | 1 × 31 × 17 | transposed | correct | [passed](operator_peer/metal__bf16__matmul__transposed__lhs_m1_k31_n17.json) | 0.1853 / 0.2052 | [passed](program_peer/metal__bf16__matmul__transposed__lhs_m1_k31_n17.json) | 0.2096 / 0.202 |
| `metal/bf16/matmul/transposed/lhs_m1_k31_n65` | matrix | 1 × 31 × 65 | transposed | correct | [passed](operator_peer/metal__bf16__matmul__transposed__lhs_m1_k31_n65.json) | 0.2052 / 0.2069 | [passed](program_peer/metal__bf16__matmul__transposed__lhs_m1_k31_n65.json) | 0.2207 / 0.201 |
| `metal/bf16/matmul/transposed/lhs_m1_k64_n17` | matrix | 1 × 64 × 17 | transposed | correct | [passed](operator_peer/metal__bf16__matmul__transposed__lhs_m1_k64_n17.json) | 0.2285 / 0.2048 | [passed](program_peer/metal__bf16__matmul__transposed__lhs_m1_k64_n17.json) | 0.2364 / 0.2035 |
| `metal/bf16/matmul/transposed/lhs_m1_k64_n65` | matrix | 1 × 64 × 65 | transposed | correct | [passed](operator_peer/metal__bf16__matmul__transposed__lhs_m1_k64_n65.json) | 0.2281 / 0.2 | [passed](program_peer/metal__bf16__matmul__transposed__lhs_m1_k64_n65.json) | 0.2387 / 0.2033 |
| `metal/bf16/matmul/transposed/lhs_m1_k129_n17` | matrix | 1 × 129 × 17 | transposed | correct | [passed](operator_peer/metal__bf16__matmul__transposed__lhs_m1_k129_n17.json) | 0.186 / 0.2047 | [passed](program_peer/metal__bf16__matmul__transposed__lhs_m1_k129_n17.json) | 0.1996 / 0.2011 |
| `metal/bf16/matmul/transposed/lhs_m1_k129_n65` | matrix | 1 × 129 × 65 | transposed | correct | [passed](operator_peer/metal__bf16__matmul__transposed__lhs_m1_k129_n65.json) | 0.2108 / 0.2094 | [passed](program_peer/metal__bf16__matmul__transposed__lhs_m1_k129_n65.json) | 0.2085 / 0.2082 |
| `metal/bf16/matmul/strided/lhs_m1_k7_n17` | matrix | 1 × 7 × 17 | strided | correct | [passed](operator_peer/metal__bf16__matmul__strided__lhs_m1_k7_n17.json) | 0.1791 / 0.2105 | [passed](program_peer/metal__bf16__matmul__strided__lhs_m1_k7_n17.json) | 0.1984 / 0.21 |
| `metal/bf16/matmul/strided/lhs_m1_k7_n65` | matrix | 1 × 7 × 65 | strided | correct | [passed](operator_peer/metal__bf16__matmul__strided__lhs_m1_k7_n65.json) | 0.1822 / 0.2092 | [passed](program_peer/metal__bf16__matmul__strided__lhs_m1_k7_n65.json) | 0.1929 / 0.2099 |
| `metal/bf16/matmul/strided/lhs_m1_k31_n17` | matrix | 1 × 31 × 17 | strided | correct | [passed](operator_peer/metal__bf16__matmul__strided__lhs_m1_k31_n17.json) | 0.1974 / 0.2104 | [passed](program_peer/metal__bf16__matmul__strided__lhs_m1_k31_n17.json) | 0.2025 / 0.2104 |
| `metal/bf16/matmul/strided/lhs_m1_k31_n65` | matrix | 1 × 31 × 65 | strided | correct | [passed](operator_peer/metal__bf16__matmul__strided__lhs_m1_k31_n65.json) | 0.1996 / 0.2119 | [passed](program_peer/metal__bf16__matmul__strided__lhs_m1_k31_n65.json) | 0.2189 / 0.2109 |
| `metal/bf16/matmul/strided/lhs_m1_k64_n17` | matrix | 1 × 64 × 17 | strided | correct | [passed](operator_peer/metal__bf16__matmul__strided__lhs_m1_k64_n17.json) | 0.2219 / 0.2114 | [passed](program_peer/metal__bf16__matmul__strided__lhs_m1_k64_n17.json) | 0.2295 / 0.2049 |
| `metal/bf16/matmul/strided/lhs_m1_k64_n65` | matrix | 1 × 64 × 65 | strided | correct | [passed](operator_peer/metal__bf16__matmul__strided__lhs_m1_k64_n65.json) | 0.2224 / 0.205 | [passed](program_peer/metal__bf16__matmul__strided__lhs_m1_k64_n65.json) | 0.2348 / 0.2075 |
| `metal/bf16/matmul/strided/lhs_m1_k129_n17` | matrix | 1 × 129 × 17 | strided | correct | [passed](operator_peer/metal__bf16__matmul__strided__lhs_m1_k129_n17.json) | 0.1928 / 0.2001 | [passed](program_peer/metal__bf16__matmul__strided__lhs_m1_k129_n17.json) | 0.1953 / 0.2077 |
| `metal/bf16/matmul/strided/lhs_m1_k129_n65` | matrix | 1 × 129 × 65 | strided | correct | [passed](operator_peer/metal__bf16__matmul__strided__lhs_m1_k129_n65.json) | 0.1914 / 0.209 | [passed](program_peer/metal__bf16__matmul__strided__lhs_m1_k129_n65.json) | 0.2094 / 0.2068 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
