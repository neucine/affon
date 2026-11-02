# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `45ff34dc1f1c8c897fe069a697b5880c2271992a2bc152209a5b75e6cfd88328`  
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
| `metal/bf16/matmul/transposed/lhs_m1_k7_n17` | matrix | 1 × 7 × 17 | transposed | correct | [passed](operator_peer/metal__bf16__matmul__transposed__lhs_m1_k7_n17.json) | 0.2664 / 0.2281 | [passed](program_peer/metal__bf16__matmul__transposed__lhs_m1_k7_n17.json) | 0.2562 / 0.2075 |
| `metal/bf16/matmul/transposed/lhs_m1_k7_n65` | matrix | 1 × 7 × 65 | transposed | correct | [passed](operator_peer/metal__bf16__matmul__transposed__lhs_m1_k7_n65.json) | 0.2402 / 0.193 | [passed](program_peer/metal__bf16__matmul__transposed__lhs_m1_k7_n65.json) | 0.2104 / 0.1948 |
| `metal/bf16/matmul/transposed/lhs_m1_k31_n17` | matrix | 1 × 31 × 17 | transposed | correct | [passed](operator_peer/metal__bf16__matmul__transposed__lhs_m1_k31_n17.json) | 0.2552 / 0.1832 | [passed](program_peer/metal__bf16__matmul__transposed__lhs_m1_k31_n17.json) | 0.26 / 0.2067 |
| `metal/bf16/matmul/transposed/lhs_m1_k31_n65` | matrix | 1 × 31 × 65 | transposed | correct | [passed](operator_peer/metal__bf16__matmul__transposed__lhs_m1_k31_n65.json) | 0.2434 / 0.1855 | [passed](program_peer/metal__bf16__matmul__transposed__lhs_m1_k31_n65.json) | 0.2828 / 0.2044 |
| `metal/bf16/matmul/transposed/lhs_m1_k64_n17` | matrix | 1 × 64 × 17 | transposed | correct | [passed](operator_peer/metal__bf16__matmul__transposed__lhs_m1_k64_n17.json) | 0.3058 / 0.201 | [passed](program_peer/metal__bf16__matmul__transposed__lhs_m1_k64_n17.json) | 0.3166 / 0.206 |
| `metal/bf16/matmul/transposed/lhs_m1_k64_n65` | matrix | 1 × 64 × 65 | transposed | correct | [passed](operator_peer/metal__bf16__matmul__transposed__lhs_m1_k64_n65.json) | 0.312 / 0.2074 | [passed](program_peer/metal__bf16__matmul__transposed__lhs_m1_k64_n65.json) | 0.3214 / 0.2078 |
| `metal/bf16/matmul/strided/lhs_m1_k7_n17` | matrix | 1 × 7 × 17 | strided | correct | [passed](operator_peer/metal__bf16__matmul__strided__lhs_m1_k7_n17.json) | 0.2402 / 0.2107 | [passed](program_peer/metal__bf16__matmul__strided__lhs_m1_k7_n17.json) | 0.2558 / 0.2103 |
| `metal/bf16/matmul/strided/lhs_m1_k7_n65` | matrix | 1 × 7 × 65 | strided | correct | [passed](operator_peer/metal__bf16__matmul__strided__lhs_m1_k7_n65.json) | 0.244 / 0.1984 | [passed](program_peer/metal__bf16__matmul__strided__lhs_m1_k7_n65.json) | 0.258 / 0.2086 |
| `metal/bf16/matmul/strided/lhs_m1_k31_n17` | matrix | 1 × 31 × 17 | strided | correct | [passed](operator_peer/metal__bf16__matmul__strided__lhs_m1_k31_n17.json) | 0.2632 / 0.2046 | [passed](program_peer/metal__bf16__matmul__strided__lhs_m1_k31_n17.json) | 0.2968 / 0.1911 |
| `metal/bf16/matmul/strided/lhs_m1_k31_n65` | matrix | 1 × 31 × 65 | strided | correct | [passed](operator_peer/metal__bf16__matmul__strided__lhs_m1_k31_n65.json) | 0.2674 / 0.2019 | [passed](program_peer/metal__bf16__matmul__strided__lhs_m1_k31_n65.json) | 0.295 / 0.1957 |
| `metal/bf16/matmul/strided/lhs_m1_k64_n17` | matrix | 1 × 64 × 17 | strided | correct | [passed](operator_peer/metal__bf16__matmul__strided__lhs_m1_k64_n17.json) | 0.2788 / 0.2043 | [passed](program_peer/metal__bf16__matmul__strided__lhs_m1_k64_n17.json) | 0.3178 / 0.2088 |
| `metal/bf16/matmul/strided/lhs_m1_k64_n65` | matrix | 1 × 64 × 65 | strided | correct | [passed](operator_peer/metal__bf16__matmul__strided__lhs_m1_k64_n65.json) | 0.3122 / 0.2077 | [passed](program_peer/metal__bf16__matmul__strided__lhs_m1_k64_n65.json) | 0.2922 / 0.1959 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
