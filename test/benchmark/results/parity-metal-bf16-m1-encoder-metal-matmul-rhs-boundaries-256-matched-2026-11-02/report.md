# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `8f32ffd41ff174a12c2785e42a4528e7b99dca3f9031315b8089ae1c48b98d70`  
Binary SHA-256: `8522698ce689fb48c552a41c4b152287fbec36f469250cc8d198a9b99cd03a8d`  
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
| `metal/bf16/matmul/transposed/rhs_m1_k8_n8` | matrix | 1 × 8 × 8 | transposed | correct | [passed](operator_peer/metal__bf16__matmul__transposed__rhs_m1_k8_n8.json) | 0.2817 / 0.2362 | [passed](program_peer/metal__bf16__matmul__transposed__rhs_m1_k8_n8.json) | 0.3297 / 0.2085 |
| `metal/bf16/matmul/transposed/rhs_m1_k8_n32` | matrix | 1 × 8 × 32 | transposed | correct | [passed](operator_peer/metal__bf16__matmul__transposed__rhs_m1_k8_n32.json) | 0.2216 / 0.1966 | [passed](program_peer/metal__bf16__matmul__transposed__rhs_m1_k8_n32.json) | 0.2667 / 0.2045 |
| `metal/bf16/matmul/transposed/rhs_m1_k8_n64` | matrix | 1 × 8 × 64 | transposed | correct | [passed](operator_peer/metal__bf16__matmul__transposed__rhs_m1_k8_n64.json) | 0.2625 / 0.2025 | [passed](program_peer/metal__bf16__matmul__transposed__rhs_m1_k8_n64.json) | 0.2588 / 0.2054 |
| `metal/bf16/matmul/transposed/rhs_m1_k8_n128` | matrix | 1 × 8 × 128 | transposed | correct | [passed](operator_peer/metal__bf16__matmul__transposed__rhs_m1_k8_n128.json) | 0.2954 / 0.2144 | [passed](program_peer/metal__bf16__matmul__transposed__rhs_m1_k8_n128.json) | 0.3053 / 0.2089 |
| `metal/bf16/matmul/transposed/rhs_m1_k32_n8` | matrix | 1 × 32 × 8 | transposed | correct | [passed](operator_peer/metal__bf16__matmul__transposed__rhs_m1_k32_n8.json) | 0.2972 / 0.2049 | [passed](program_peer/metal__bf16__matmul__transposed__rhs_m1_k32_n8.json) | 0.3421 / 0.2373 |
| `metal/bf16/matmul/transposed/rhs_m1_k32_n32` | matrix | 1 × 32 × 32 | transposed | correct | [passed](operator_peer/metal__bf16__matmul__transposed__rhs_m1_k32_n32.json) | 0.2674 / 0.1948 | [passed](program_peer/metal__bf16__matmul__transposed__rhs_m1_k32_n32.json) | 0.2899 / 0.2163 |
| `metal/bf16/matmul/transposed/rhs_m1_k32_n64` | matrix | 1 × 32 × 64 | transposed | correct | [passed](operator_peer/metal__bf16__matmul__transposed__rhs_m1_k32_n64.json) | 0.2547 / 0.2052 | [passed](program_peer/metal__bf16__matmul__transposed__rhs_m1_k32_n64.json) | 0.2558 / 0.2092 |
| `metal/bf16/matmul/transposed/rhs_m1_k32_n128` | matrix | 1 × 32 × 128 | transposed | correct | [passed](operator_peer/metal__bf16__matmul__transposed__rhs_m1_k32_n128.json) | 0.3028 / 0.2029 | [passed](program_peer/metal__bf16__matmul__transposed__rhs_m1_k32_n128.json) | 0.3264 / 0.2186 |
| `metal/bf16/matmul/transposed/rhs_m1_k63_n8` | matrix | 1 × 63 × 8 | transposed | correct | [passed](operator_peer/metal__bf16__matmul__transposed__rhs_m1_k63_n8.json) | 0.2704 / 0.2017 | [passed](program_peer/metal__bf16__matmul__transposed__rhs_m1_k63_n8.json) | 0.3183 / 0.2111 |
| `metal/bf16/matmul/transposed/rhs_m1_k63_n32` | matrix | 1 × 63 × 32 | transposed | correct | [passed](operator_peer/metal__bf16__matmul__transposed__rhs_m1_k63_n32.json) | 0.3113 / 0.1994 | [passed](program_peer/metal__bf16__matmul__transposed__rhs_m1_k63_n32.json) | 0.3259 / 0.2033 |
| `metal/bf16/matmul/transposed/rhs_m1_k63_n64` | matrix | 1 × 63 × 64 | transposed | correct | [passed](operator_peer/metal__bf16__matmul__transposed__rhs_m1_k63_n64.json) | 0.3689 / 0.2201 | [passed](program_peer/metal__bf16__matmul__transposed__rhs_m1_k63_n64.json) | 0.3437 / 0.2215 |
| `metal/bf16/matmul/transposed/rhs_m1_k63_n128` | matrix | 1 × 63 × 128 | transposed | correct | [passed](operator_peer/metal__bf16__matmul__transposed__rhs_m1_k63_n128.json) | 0.3256 / 0.2225 | [passed](program_peer/metal__bf16__matmul__transposed__rhs_m1_k63_n128.json) | 0.3363 / 0.2117 |
| `metal/bf16/matmul/strided/rhs_m1_k8_n8` | matrix | 1 × 8 × 8 | strided | correct | [passed](operator_peer/metal__bf16__matmul__strided__rhs_m1_k8_n8.json) | 0.2773 / 0.2194 | [passed](program_peer/metal__bf16__matmul__strided__rhs_m1_k8_n8.json) | 0.339 / 0.2114 |
| `metal/bf16/matmul/strided/rhs_m1_k8_n32` | matrix | 1 × 8 × 32 | strided | correct | [passed](operator_peer/metal__bf16__matmul__strided__rhs_m1_k8_n32.json) | 0.284 / 0.2195 | [passed](program_peer/metal__bf16__matmul__strided__rhs_m1_k8_n32.json) | 0.2636 / 0.2136 |
| `metal/bf16/matmul/strided/rhs_m1_k8_n64` | matrix | 1 × 8 × 64 | strided | correct | [passed](operator_peer/metal__bf16__matmul__strided__rhs_m1_k8_n64.json) | 0.2929 / 0.2146 | [passed](program_peer/metal__bf16__matmul__strided__rhs_m1_k8_n64.json) | 0.2524 / 0.2313 |
| `metal/bf16/matmul/strided/rhs_m1_k8_n128` | matrix | 1 × 8 × 128 | strided | correct | [passed](operator_peer/metal__bf16__matmul__strided__rhs_m1_k8_n128.json) | 0.2449 / 0.2146 | [passed](program_peer/metal__bf16__matmul__strided__rhs_m1_k8_n128.json) | 0.2147 / 0.2033 |
| `metal/bf16/matmul/strided/rhs_m1_k32_n8` | matrix | 1 × 32 × 8 | strided | correct | [passed](operator_peer/metal__bf16__matmul__strided__rhs_m1_k32_n8.json) | 0.2814 / 0.2065 | [passed](program_peer/metal__bf16__matmul__strided__rhs_m1_k32_n8.json) | 0.2549 / 0.2084 |
| `metal/bf16/matmul/strided/rhs_m1_k32_n32` | matrix | 1 × 32 × 32 | strided | correct | [passed](operator_peer/metal__bf16__matmul__strided__rhs_m1_k32_n32.json) | 0.3148 / 0.2482 | [passed](program_peer/metal__bf16__matmul__strided__rhs_m1_k32_n32.json) | 0.3477 / 0.2132 |
| `metal/bf16/matmul/strided/rhs_m1_k32_n64` | matrix | 1 × 32 × 64 | strided | correct | [passed](operator_peer/metal__bf16__matmul__strided__rhs_m1_k32_n64.json) | 0.3263 / 0.2123 | [passed](program_peer/metal__bf16__matmul__strided__rhs_m1_k32_n64.json) | 0.3379 / 0.2145 |
| `metal/bf16/matmul/strided/rhs_m1_k32_n128` | matrix | 1 × 32 × 128 | strided | correct | [passed](operator_peer/metal__bf16__matmul__strided__rhs_m1_k32_n128.json) | 0.2941 / 0.2071 | [passed](program_peer/metal__bf16__matmul__strided__rhs_m1_k32_n128.json) | 0.3601 / 0.2166 |
| `metal/bf16/matmul/strided/rhs_m1_k63_n8` | matrix | 1 × 63 × 8 | strided | correct | [passed](operator_peer/metal__bf16__matmul__strided__rhs_m1_k63_n8.json) | 0.283 / 0.2057 | [passed](program_peer/metal__bf16__matmul__strided__rhs_m1_k63_n8.json) | 0.3227 / 0.2022 |
| `metal/bf16/matmul/strided/rhs_m1_k63_n32` | matrix | 1 × 63 × 32 | strided | correct | [passed](operator_peer/metal__bf16__matmul__strided__rhs_m1_k63_n32.json) | 0.3129 / 0.224 | [passed](program_peer/metal__bf16__matmul__strided__rhs_m1_k63_n32.json) | 0.3252 / 0.2408 |
| `metal/bf16/matmul/strided/rhs_m1_k63_n64` | matrix | 1 × 63 × 64 | strided | correct | [passed](operator_peer/metal__bf16__matmul__strided__rhs_m1_k63_n64.json) | 0.3737 / 0.2147 | [passed](program_peer/metal__bf16__matmul__strided__rhs_m1_k63_n64.json) | 0.3247 / 0.2115 |
| `metal/bf16/matmul/strided/rhs_m1_k63_n128` | matrix | 1 × 63 × 128 | strided | correct | [passed](operator_peer/metal__bf16__matmul__strided__rhs_m1_k63_n128.json) | 0.4055 / 0.2148 | [passed](program_peer/metal__bf16__matmul__strided__rhs_m1_k63_n128.json) | 0.2482 / 0.2216 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
