# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `f1b92c340847b9bddae6749ef94e920b428923c3eabb83903bc407c4ab3577f0`  
Binary SHA-256: `3552c52bc08639089d28f433463f3fb1e1502ad44033b793a0d38ceebf3bb968`  
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
| `metal/bf16/matmul/transposed/lhs_m1_k7_n17` | matrix | 1 × 7 × 17 | transposed | correct | [passed](operator_peer/metal__bf16__matmul__transposed__lhs_m1_k7_n17.json) | 0.1953 / 0.2042 | [passed](program_peer/metal__bf16__matmul__transposed__lhs_m1_k7_n17.json) | 0.1951 / 0.2076 |
| `metal/bf16/matmul/transposed/lhs_m1_k7_n65` | matrix | 1 × 7 × 65 | transposed | correct | [passed](operator_peer/metal__bf16__matmul__transposed__lhs_m1_k7_n65.json) | 0.1864 / 0.2057 | [passed](program_peer/metal__bf16__matmul__transposed__lhs_m1_k7_n65.json) | 0.1839 / 0.2021 |
| `metal/bf16/matmul/transposed/lhs_m1_k31_n17` | matrix | 1 × 31 × 17 | transposed | correct | [passed](operator_peer/metal__bf16__matmul__transposed__lhs_m1_k31_n17.json) | 0.2003 / 0.2059 | [passed](program_peer/metal__bf16__matmul__transposed__lhs_m1_k31_n17.json) | 0.2283 / 0.2063 |
| `metal/bf16/matmul/transposed/lhs_m1_k31_n65` | matrix | 1 × 31 × 65 | transposed | correct | [passed](operator_peer/metal__bf16__matmul__transposed__lhs_m1_k31_n65.json) | 0.2087 / 0.2062 | [passed](program_peer/metal__bf16__matmul__transposed__lhs_m1_k31_n65.json) | 0.234 / 0.2038 |
| `metal/bf16/matmul/transposed/lhs_m1_k64_n17` | matrix | 1 × 64 × 17 | transposed | correct | [passed](operator_peer/metal__bf16__matmul__transposed__lhs_m1_k64_n17.json) | 0.2279 / 0.204 | [passed](program_peer/metal__bf16__matmul__transposed__lhs_m1_k64_n17.json) | 0.239 / 0.2019 |
| `metal/bf16/matmul/transposed/lhs_m1_k64_n65` | matrix | 1 × 64 × 65 | transposed | correct | [passed](operator_peer/metal__bf16__matmul__transposed__lhs_m1_k64_n65.json) | 0.2373 / 0.2043 | [passed](program_peer/metal__bf16__matmul__transposed__lhs_m1_k64_n65.json) | 0.2354 / 0.2051 |
| `metal/bf16/matmul/transposed/lhs_m1_k129_n17` | matrix | 1 × 129 × 17 | transposed | correct | [passed](operator_peer/metal__bf16__matmul__transposed__lhs_m1_k129_n17.json) | 0.1838 / 0.2053 | [passed](program_peer/metal__bf16__matmul__transposed__lhs_m1_k129_n17.json) | 0.1967 / 0.2017 |
| `metal/bf16/matmul/transposed/lhs_m1_k129_n65` | matrix | 1 × 129 × 65 | transposed | correct | [passed](operator_peer/metal__bf16__matmul__transposed__lhs_m1_k129_n65.json) | 0.189 / 0.2046 | [passed](program_peer/metal__bf16__matmul__transposed__lhs_m1_k129_n65.json) | 0.2064 / 0.204 |
| `metal/bf16/matmul/strided/lhs_m1_k7_n17` | matrix | 1 × 7 × 17 | strided | correct | [passed](operator_peer/metal__bf16__matmul__strided__lhs_m1_k7_n17.json) | 0.1883 / 0.2074 | [passed](program_peer/metal__bf16__matmul__strided__lhs_m1_k7_n17.json) | 0.2057 / 0.2083 |
| `metal/bf16/matmul/strided/lhs_m1_k7_n65` | matrix | 1 × 7 × 65 | strided | correct | [passed](operator_peer/metal__bf16__matmul__strided__lhs_m1_k7_n65.json) | 0.1891 / 0.211 | [passed](program_peer/metal__bf16__matmul__strided__lhs_m1_k7_n65.json) | 0.1913 / 0.2115 |
| `metal/bf16/matmul/strided/lhs_m1_k31_n17` | matrix | 1 × 31 × 17 | strided | correct | [passed](operator_peer/metal__bf16__matmul__strided__lhs_m1_k31_n17.json) | 0.1947 / 0.2092 | [passed](program_peer/metal__bf16__matmul__strided__lhs_m1_k31_n17.json) | 0.2287 / 0.2123 |
| `metal/bf16/matmul/strided/lhs_m1_k31_n65` | matrix | 1 × 31 × 65 | strided | correct | [passed](operator_peer/metal__bf16__matmul__strided__lhs_m1_k31_n65.json) | 0.2067 / 0.2119 | [passed](program_peer/metal__bf16__matmul__strided__lhs_m1_k31_n65.json) | 0.2305 / 0.2064 |
| `metal/bf16/matmul/strided/lhs_m1_k64_n17` | matrix | 1 × 64 × 17 | strided | correct | [passed](operator_peer/metal__bf16__matmul__strided__lhs_m1_k64_n17.json) | 0.2254 / 0.2096 | [passed](program_peer/metal__bf16__matmul__strided__lhs_m1_k64_n17.json) | 0.2344 / 0.206 |
| `metal/bf16/matmul/strided/lhs_m1_k64_n65` | matrix | 1 × 64 × 65 | strided | correct | [passed](operator_peer/metal__bf16__matmul__strided__lhs_m1_k64_n65.json) | 0.2297 / 0.2036 | [passed](program_peer/metal__bf16__matmul__strided__lhs_m1_k64_n65.json) | 0.2421 / 0.2079 |
| `metal/bf16/matmul/strided/lhs_m1_k129_n17` | matrix | 1 × 129 × 17 | strided | correct | [passed](operator_peer/metal__bf16__matmul__strided__lhs_m1_k129_n17.json) | 0.1787 / 0.202 | [passed](program_peer/metal__bf16__matmul__strided__lhs_m1_k129_n17.json) | 0.1955 / 0.2075 |
| `metal/bf16/matmul/strided/lhs_m1_k129_n65` | matrix | 1 × 129 × 65 | strided | correct | [passed](operator_peer/metal__bf16__matmul__strided__lhs_m1_k129_n65.json) | 0.1875 / 0.209 | [passed](program_peer/metal__bf16__matmul__strided__lhs_m1_k129_n65.json) | 0.2071 / 0.2075 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
