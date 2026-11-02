# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `8f32ffd41ff174a12c2785e42a4528e7b99dca3f9031315b8089ae1c48b98d70`  
Binary SHA-256: `a9a10dd4a4dd6ab577715a114b1261d475aeddc676e7750489ffff1b83de8cef`  
Timing: 5 iterations × 3 samples × 2 rounds

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
| `metal/bf16/matmul/transposed/rhs_m1_k8_n8` | matrix | 1 × 8 × 8 | transposed | correct | [passed](operator_peer/metal__bf16__matmul__transposed__rhs_m1_k8_n8.json) | 0.2994 / 0.2321 | [passed](program_peer/metal__bf16__matmul__transposed__rhs_m1_k8_n8.json) | 0.2805 / 0.1995 |
| `metal/bf16/matmul/transposed/rhs_m1_k8_n32` | matrix | 1 × 8 × 32 | transposed | correct | [passed](operator_peer/metal__bf16__matmul__transposed__rhs_m1_k8_n32.json) | 0.2822 / 0.1967 | [passed](program_peer/metal__bf16__matmul__transposed__rhs_m1_k8_n32.json) | 0.2952 / 0.2135 |
| `metal/bf16/matmul/transposed/rhs_m1_k8_n64` | matrix | 1 × 8 × 64 | transposed | correct | [passed](operator_peer/metal__bf16__matmul__transposed__rhs_m1_k8_n64.json) | 0.2442 / 0.1964 | [passed](program_peer/metal__bf16__matmul__transposed__rhs_m1_k8_n64.json) | 0.269 / 0.2042 |
| `metal/bf16/matmul/transposed/rhs_m1_k8_n128` | matrix | 1 × 8 × 128 | transposed | correct | [passed](operator_peer/metal__bf16__matmul__transposed__rhs_m1_k8_n128.json) | 0.3126 / 0.2061 | [passed](program_peer/metal__bf16__matmul__transposed__rhs_m1_k8_n128.json) | 0.2667 / 0.2211 |
| `metal/bf16/matmul/transposed/rhs_m1_k32_n8` | matrix | 1 × 32 × 8 | transposed | correct | [passed](operator_peer/metal__bf16__matmul__transposed__rhs_m1_k32_n8.json) | 0.2625 / 0.2151 | [passed](program_peer/metal__bf16__matmul__transposed__rhs_m1_k32_n8.json) | 0.3399 / 0.2006 |
| `metal/bf16/matmul/transposed/rhs_m1_k32_n32` | matrix | 1 × 32 × 32 | transposed | correct | [passed](operator_peer/metal__bf16__matmul__transposed__rhs_m1_k32_n32.json) | 0.3267 / 0.1968 | [passed](program_peer/metal__bf16__matmul__transposed__rhs_m1_k32_n32.json) | 0.3088 / 0.203 |
| `metal/bf16/matmul/transposed/rhs_m1_k32_n64` | matrix | 1 × 32 × 64 | transposed | correct | [passed](operator_peer/metal__bf16__matmul__transposed__rhs_m1_k32_n64.json) | 0.3208 / 0.218 | [passed](program_peer/metal__bf16__matmul__transposed__rhs_m1_k32_n64.json) | 0.3169 / 0.2022 |
| `metal/bf16/matmul/transposed/rhs_m1_k32_n128` | matrix | 1 × 32 × 128 | transposed | correct | [passed](operator_peer/metal__bf16__matmul__transposed__rhs_m1_k32_n128.json) | 0.3137 / 0.2009 | [passed](program_peer/metal__bf16__matmul__transposed__rhs_m1_k32_n128.json) | 0.3312 / 0.2316 |
| `metal/bf16/matmul/transposed/rhs_m1_k63_n8` | matrix | 1 × 63 × 8 | transposed | correct | [passed](operator_peer/metal__bf16__matmul__transposed__rhs_m1_k63_n8.json) | 0.293 / 0.1909 | [passed](program_peer/metal__bf16__matmul__transposed__rhs_m1_k63_n8.json) | 0.3674 / 0.1897 |
| `metal/bf16/matmul/transposed/rhs_m1_k63_n32` | matrix | 1 × 63 × 32 | transposed | correct | [passed](operator_peer/metal__bf16__matmul__transposed__rhs_m1_k63_n32.json) | 0.3292 / 0.2066 | [passed](program_peer/metal__bf16__matmul__transposed__rhs_m1_k63_n32.json) | 0.3712 / 0.2431 |
| `metal/bf16/matmul/transposed/rhs_m1_k63_n64` | matrix | 1 × 63 × 64 | transposed | correct | [passed](operator_peer/metal__bf16__matmul__transposed__rhs_m1_k63_n64.json) | 0.3364 / 0.2 | [passed](program_peer/metal__bf16__matmul__transposed__rhs_m1_k63_n64.json) | 0.3523 / 0.2123 |
| `metal/bf16/matmul/transposed/rhs_m1_k63_n128` | matrix | 1 × 63 × 128 | transposed | correct | [passed](operator_peer/metal__bf16__matmul__transposed__rhs_m1_k63_n128.json) | 0.2357 / 0.2075 | [passed](program_peer/metal__bf16__matmul__transposed__rhs_m1_k63_n128.json) | 0.3664 / 0.215 |
| `metal/bf16/matmul/strided/rhs_m1_k8_n8` | matrix | 1 × 8 × 8 | strided | correct | [passed](operator_peer/metal__bf16__matmul__strided__rhs_m1_k8_n8.json) | 0.2635 / 0.2075 | [passed](program_peer/metal__bf16__matmul__strided__rhs_m1_k8_n8.json) | 0.3009 / 0.2056 |
| `metal/bf16/matmul/strided/rhs_m1_k8_n32` | matrix | 1 × 8 × 32 | strided | correct | [passed](operator_peer/metal__bf16__matmul__strided__rhs_m1_k8_n32.json) | 0.283 / 0.2084 | [passed](program_peer/metal__bf16__matmul__strided__rhs_m1_k8_n32.json) | 0.2231 / 0.213 |
| `metal/bf16/matmul/strided/rhs_m1_k8_n64` | matrix | 1 × 8 × 64 | strided | correct | [passed](operator_peer/metal__bf16__matmul__strided__rhs_m1_k8_n64.json) | 0.32 / 0.2031 | [passed](program_peer/metal__bf16__matmul__strided__rhs_m1_k8_n64.json) | 0.2484 / 0.2202 |
| `metal/bf16/matmul/strided/rhs_m1_k8_n128` | matrix | 1 × 8 × 128 | strided | correct | [passed](operator_peer/metal__bf16__matmul__strided__rhs_m1_k8_n128.json) | 0.2811 / 0.2372 | [passed](program_peer/metal__bf16__matmul__strided__rhs_m1_k8_n128.json) | 0.3275 / 0.2145 |
| `metal/bf16/matmul/strided/rhs_m1_k32_n8` | matrix | 1 × 32 × 8 | strided | correct | [passed](operator_peer/metal__bf16__matmul__strided__rhs_m1_k32_n8.json) | 0.2521 / 0.2069 | [passed](program_peer/metal__bf16__matmul__strided__rhs_m1_k32_n8.json) | 0.3477 / 0.2141 |
| `metal/bf16/matmul/strided/rhs_m1_k32_n32` | matrix | 1 × 32 × 32 | strided | correct | [passed](operator_peer/metal__bf16__matmul__strided__rhs_m1_k32_n32.json) | 0.238 / 0.2122 | [passed](program_peer/metal__bf16__matmul__strided__rhs_m1_k32_n32.json) | 0.324 / 0.2189 |
| `metal/bf16/matmul/strided/rhs_m1_k32_n64` | matrix | 1 × 32 × 64 | strided | correct | [passed](operator_peer/metal__bf16__matmul__strided__rhs_m1_k32_n64.json) | 0.3211 / 0.2187 | [passed](program_peer/metal__bf16__matmul__strided__rhs_m1_k32_n64.json) | 0.3029 / 0.2225 |
| `metal/bf16/matmul/strided/rhs_m1_k32_n128` | matrix | 1 × 32 × 128 | strided | correct | [passed](operator_peer/metal__bf16__matmul__strided__rhs_m1_k32_n128.json) | 0.2317 / 0.2171 | [passed](program_peer/metal__bf16__matmul__strided__rhs_m1_k32_n128.json) | 0.3202 / 0.2137 |
| `metal/bf16/matmul/strided/rhs_m1_k63_n8` | matrix | 1 × 63 × 8 | strided | correct | [passed](operator_peer/metal__bf16__matmul__strided__rhs_m1_k63_n8.json) | 0.3407 / 0.2055 | [passed](program_peer/metal__bf16__matmul__strided__rhs_m1_k63_n8.json) | 0.3458 / 0.2108 |
| `metal/bf16/matmul/strided/rhs_m1_k63_n32` | matrix | 1 × 63 × 32 | strided | correct | [passed](operator_peer/metal__bf16__matmul__strided__rhs_m1_k63_n32.json) | 0.3675 / 0.2106 | [passed](program_peer/metal__bf16__matmul__strided__rhs_m1_k63_n32.json) | 0.2504 / 0.2163 |
| `metal/bf16/matmul/strided/rhs_m1_k63_n64` | matrix | 1 × 63 × 64 | strided | correct | [passed](operator_peer/metal__bf16__matmul__strided__rhs_m1_k63_n64.json) | 0.3254 / 0.2138 | [passed](program_peer/metal__bf16__matmul__strided__rhs_m1_k63_n64.json) | 0.3471 / 0.2157 |
| `metal/bf16/matmul/strided/rhs_m1_k63_n128` | matrix | 1 × 63 × 128 | strided | correct | [passed](operator_peer/metal__bf16__matmul__strided__rhs_m1_k63_n128.json) | 0.3885 / 0.1979 | [passed](program_peer/metal__bf16__matmul__strided__rhs_m1_k63_n128.json) | 0.4164 / 0.2214 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
