# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `8f32ffd41ff174a12c2785e42a4528e7b99dca3f9031315b8089ae1c48b98d70`  
Binary SHA-256: `8522698ce689fb48c552a41c4b152287fbec36f469250cc8d198a9b99cd03a8d`  
Timing: 30 iterations × 5 samples × 3 rounds

## Checks

- Cases run: 24
- Numerical check passed: 24
- Operator check and timing passed: 24
- Program check and timing passed: 24
- Numerical failures: 0
- Operator failures: 0
- Program failures: 0

## Cases

| Case | Family | Shape | Layout | Numerical | Operator | Affon / PyTorch ms | Program | Affon / PyTorch ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `metal/bf16/matmul/transposed/rhs_m1_k8_n8` | matrix | 1 × 8 × 8 | transposed | correct | [passed](operator_peer/metal__bf16__matmul__transposed__rhs_m1_k8_n8.json) | 0.176 / 0.2048 | [passed](program_peer/metal__bf16__matmul__transposed__rhs_m1_k8_n8.json) | 0.1918 / 0.2089 |
| `metal/bf16/matmul/transposed/rhs_m1_k8_n32` | matrix | 1 × 8 × 32 | transposed | correct | [passed](operator_peer/metal__bf16__matmul__transposed__rhs_m1_k8_n32.json) | 0.1977 / 0.207 | [passed](program_peer/metal__bf16__matmul__transposed__rhs_m1_k8_n32.json) | 0.1987 / 0.2097 |
| `metal/bf16/matmul/transposed/rhs_m1_k8_n64` | matrix | 1 × 8 × 64 | transposed | correct | [passed](operator_peer/metal__bf16__matmul__transposed__rhs_m1_k8_n64.json) | 0.18 / 0.209 | [passed](program_peer/metal__bf16__matmul__transposed__rhs_m1_k8_n64.json) | 0.1845 / 0.2026 |
| `metal/bf16/matmul/transposed/rhs_m1_k8_n128` | matrix | 1 × 8 × 128 | transposed | correct | [passed](operator_peer/metal__bf16__matmul__transposed__rhs_m1_k8_n128.json) | 0.1797 / 0.2083 | [passed](program_peer/metal__bf16__matmul__transposed__rhs_m1_k8_n128.json) | 0.1881 / 0.2092 |
| `metal/bf16/matmul/transposed/rhs_m1_k32_n8` | matrix | 1 × 32 × 8 | transposed | correct | [passed](operator_peer/metal__bf16__matmul__transposed__rhs_m1_k32_n8.json) | 0.1791 / 0.2093 | [passed](program_peer/metal__bf16__matmul__transposed__rhs_m1_k32_n8.json) | 0.1878 / 0.2102 |
| `metal/bf16/matmul/transposed/rhs_m1_k32_n32` | matrix | 1 × 32 × 32 | transposed | correct | [passed](operator_peer/metal__bf16__matmul__transposed__rhs_m1_k32_n32.json) | 0.1914 / 0.2074 | [passed](program_peer/metal__bf16__matmul__transposed__rhs_m1_k32_n32.json) | 0.1966 / 0.2109 |
| `metal/bf16/matmul/transposed/rhs_m1_k32_n64` | matrix | 1 × 32 × 64 | transposed | correct | [passed](operator_peer/metal__bf16__matmul__transposed__rhs_m1_k32_n64.json) | 0.1808 / 0.2071 | [passed](program_peer/metal__bf16__matmul__transposed__rhs_m1_k32_n64.json) | 0.204 / 0.2081 |
| `metal/bf16/matmul/transposed/rhs_m1_k32_n128` | matrix | 1 × 32 × 128 | transposed | correct | [passed](operator_peer/metal__bf16__matmul__transposed__rhs_m1_k32_n128.json) | 0.1859 / 0.2075 | [passed](program_peer/metal__bf16__matmul__transposed__rhs_m1_k32_n128.json) | 0.1998 / 0.2084 |
| `metal/bf16/matmul/transposed/rhs_m1_k63_n8` | matrix | 1 × 63 × 8 | transposed | correct | [passed](operator_peer/metal__bf16__matmul__transposed__rhs_m1_k63_n8.json) | 0.193 / 0.206 | [passed](program_peer/metal__bf16__matmul__transposed__rhs_m1_k63_n8.json) | 0.2063 / 0.2096 |
| `metal/bf16/matmul/transposed/rhs_m1_k63_n32` | matrix | 1 × 63 × 32 | transposed | correct | [passed](operator_peer/metal__bf16__matmul__transposed__rhs_m1_k63_n32.json) | 0.2239 / 0.2043 | [passed](program_peer/metal__bf16__matmul__transposed__rhs_m1_k63_n32.json) | 0.2276 / 0.2084 |
| `metal/bf16/matmul/transposed/rhs_m1_k63_n64` | matrix | 1 × 63 × 64 | transposed | correct | [passed](operator_peer/metal__bf16__matmul__transposed__rhs_m1_k63_n64.json) | 0.2079 / 0.2036 | [passed](program_peer/metal__bf16__matmul__transposed__rhs_m1_k63_n64.json) | 0.2348 / 0.2054 |
| `metal/bf16/matmul/transposed/rhs_m1_k63_n128` | matrix | 1 × 63 × 128 | transposed | correct | [passed](operator_peer/metal__bf16__matmul__transposed__rhs_m1_k63_n128.json) | 0.2281 / 0.204 | [passed](program_peer/metal__bf16__matmul__transposed__rhs_m1_k63_n128.json) | 0.2364 / 0.2067 |
| `metal/bf16/matmul/strided/rhs_m1_k8_n8` | matrix | 1 × 8 × 8 | strided | correct | [passed](operator_peer/metal__bf16__matmul__strided__rhs_m1_k8_n8.json) | 0.2045 / 0.2116 | [passed](program_peer/metal__bf16__matmul__strided__rhs_m1_k8_n8.json) | 0.1942 / 0.2133 |
| `metal/bf16/matmul/strided/rhs_m1_k8_n32` | matrix | 1 × 8 × 32 | strided | correct | [passed](operator_peer/metal__bf16__matmul__strided__rhs_m1_k8_n32.json) | 0.1826 / 0.2147 | [passed](program_peer/metal__bf16__matmul__strided__rhs_m1_k8_n32.json) | 0.1986 / 0.2135 |
| `metal/bf16/matmul/strided/rhs_m1_k8_n64` | matrix | 1 × 8 × 64 | strided | correct | [passed](operator_peer/metal__bf16__matmul__strided__rhs_m1_k8_n64.json) | 0.2042 / 0.214 | [passed](program_peer/metal__bf16__matmul__strided__rhs_m1_k8_n64.json) | 0.1906 / 0.2151 |
| `metal/bf16/matmul/strided/rhs_m1_k8_n128` | matrix | 1 × 8 × 128 | strided | correct | [passed](operator_peer/metal__bf16__matmul__strided__rhs_m1_k8_n128.json) | 0.182 / 0.2131 | [passed](program_peer/metal__bf16__matmul__strided__rhs_m1_k8_n128.json) | 0.1928 / 0.2147 |
| `metal/bf16/matmul/strided/rhs_m1_k32_n8` | matrix | 1 × 32 × 8 | strided | correct | [passed](operator_peer/metal__bf16__matmul__strided__rhs_m1_k32_n8.json) | 0.1948 / 0.2135 | [passed](program_peer/metal__bf16__matmul__strided__rhs_m1_k32_n8.json) | 0.1973 / 0.2181 |
| `metal/bf16/matmul/strided/rhs_m1_k32_n32` | matrix | 1 × 32 × 32 | strided | correct | [passed](operator_peer/metal__bf16__matmul__strided__rhs_m1_k32_n32.json) | 0.2255 / 0.2132 | [passed](program_peer/metal__bf16__matmul__strided__rhs_m1_k32_n32.json) | 0.23 / 0.2126 |
| `metal/bf16/matmul/strided/rhs_m1_k32_n64` | matrix | 1 × 32 × 64 | strided | correct | [passed](operator_peer/metal__bf16__matmul__strided__rhs_m1_k32_n64.json) | 0.1958 / 0.213 | [passed](program_peer/metal__bf16__matmul__strided__rhs_m1_k32_n64.json) | 0.2113 / 0.214 |
| `metal/bf16/matmul/strided/rhs_m1_k32_n128` | matrix | 1 × 32 × 128 | strided | correct | [passed](operator_peer/metal__bf16__matmul__strided__rhs_m1_k32_n128.json) | 0.1981 / 0.2124 | [passed](program_peer/metal__bf16__matmul__strided__rhs_m1_k32_n128.json) | 0.2076 / 0.2161 |
| `metal/bf16/matmul/strided/rhs_m1_k63_n8` | matrix | 1 × 63 × 8 | strided | correct | [passed](operator_peer/metal__bf16__matmul__strided__rhs_m1_k63_n8.json) | 0.2278 / 0.2104 | [passed](program_peer/metal__bf16__matmul__strided__rhs_m1_k63_n8.json) | 0.2316 / 0.2127 |
| `metal/bf16/matmul/strided/rhs_m1_k63_n32` | matrix | 1 × 63 × 32 | strided | correct | [passed](operator_peer/metal__bf16__matmul__strided__rhs_m1_k63_n32.json) | 0.234 / 0.2107 | [passed](program_peer/metal__bf16__matmul__strided__rhs_m1_k63_n32.json) | 0.243 / 0.2102 |
| `metal/bf16/matmul/strided/rhs_m1_k63_n64` | matrix | 1 × 63 × 64 | strided | correct | [passed](operator_peer/metal__bf16__matmul__strided__rhs_m1_k63_n64.json) | 0.236 / 0.211 | [passed](program_peer/metal__bf16__matmul__strided__rhs_m1_k63_n64.json) | 0.2453 / 0.2141 |
| `metal/bf16/matmul/strided/rhs_m1_k63_n128` | matrix | 1 × 63 × 128 | strided | correct | [passed](operator_peer/metal__bf16__matmul__strided__rhs_m1_k63_n128.json) | 0.2347 / 0.2127 | [passed](program_peer/metal__bf16__matmul__strided__rhs_m1_k63_n128.json) | 0.2384 / 0.2152 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
