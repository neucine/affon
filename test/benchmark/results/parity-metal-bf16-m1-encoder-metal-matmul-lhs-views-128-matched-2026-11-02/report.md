# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `45ff34dc1f1c8c897fe069a697b5880c2271992a2bc152209a5b75e6cfd88328`  
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
| `metal/bf16/matmul/transposed/lhs_m1_k7_n17` | matrix | 1 × 7 × 17 | transposed | correct | [passed](operator_peer/metal__bf16__matmul__transposed__lhs_m1_k7_n17.json) | 0.2496 / 0.2097 | [passed](program_peer/metal__bf16__matmul__transposed__lhs_m1_k7_n17.json) | 0.254 / 0.2001 |
| `metal/bf16/matmul/transposed/lhs_m1_k7_n65` | matrix | 1 × 7 × 65 | transposed | correct | [passed](operator_peer/metal__bf16__matmul__transposed__lhs_m1_k7_n65.json) | 0.2406 / 0.2077 | [passed](program_peer/metal__bf16__matmul__transposed__lhs_m1_k7_n65.json) | 0.2574 / 0.1977 |
| `metal/bf16/matmul/transposed/lhs_m1_k31_n17` | matrix | 1 × 31 × 17 | transposed | correct | [passed](operator_peer/metal__bf16__matmul__transposed__lhs_m1_k31_n17.json) | 0.2568 / 0.2076 | [passed](program_peer/metal__bf16__matmul__transposed__lhs_m1_k31_n17.json) | 0.2926 / 0.2123 |
| `metal/bf16/matmul/transposed/lhs_m1_k31_n65` | matrix | 1 × 31 × 65 | transposed | correct | [passed](operator_peer/metal__bf16__matmul__transposed__lhs_m1_k31_n65.json) | 0.2616 / 0.2054 | [passed](program_peer/metal__bf16__matmul__transposed__lhs_m1_k31_n65.json) | 0.2848 / 0.2082 |
| `metal/bf16/matmul/transposed/lhs_m1_k64_n17` | matrix | 1 × 64 × 17 | transposed | correct | [passed](operator_peer/metal__bf16__matmul__transposed__lhs_m1_k64_n17.json) | 0.302 / 0.1952 | [passed](program_peer/metal__bf16__matmul__transposed__lhs_m1_k64_n17.json) | 0.2478 / 0.2071 |
| `metal/bf16/matmul/transposed/lhs_m1_k64_n65` | matrix | 1 × 64 × 65 | transposed | correct | [passed](operator_peer/metal__bf16__matmul__transposed__lhs_m1_k64_n65.json) | 0.3096 / 0.2074 | [passed](program_peer/metal__bf16__matmul__transposed__lhs_m1_k64_n65.json) | 0.3212 / 0.2022 |
| `metal/bf16/matmul/strided/lhs_m1_k7_n17` | matrix | 1 × 7 × 17 | strided | correct | [passed](operator_peer/metal__bf16__matmul__strided__lhs_m1_k7_n17.json) | 0.241 / 0.2014 | [passed](program_peer/metal__bf16__matmul__strided__lhs_m1_k7_n17.json) | 0.246 / 0.2182 |
| `metal/bf16/matmul/strided/lhs_m1_k7_n65` | matrix | 1 × 7 × 65 | strided | correct | [passed](operator_peer/metal__bf16__matmul__strided__lhs_m1_k7_n65.json) | 0.2398 / 0.2081 | [passed](program_peer/metal__bf16__matmul__strided__lhs_m1_k7_n65.json) | 0.2388 / 0.2088 |
| `metal/bf16/matmul/strided/lhs_m1_k31_n17` | matrix | 1 × 31 × 17 | strided | correct | [passed](operator_peer/metal__bf16__matmul__strided__lhs_m1_k31_n17.json) | 0.2574 / 0.2029 | [passed](program_peer/metal__bf16__matmul__strided__lhs_m1_k31_n17.json) | 0.292 / 0.2032 |
| `metal/bf16/matmul/strided/lhs_m1_k31_n65` | matrix | 1 × 31 × 65 | strided | correct | [passed](operator_peer/metal__bf16__matmul__strided__lhs_m1_k31_n65.json) | 0.265 / 0.2106 | [passed](program_peer/metal__bf16__matmul__strided__lhs_m1_k31_n65.json) | 0.2938 / 0.2087 |
| `metal/bf16/matmul/strided/lhs_m1_k64_n17` | matrix | 1 × 64 × 17 | strided | correct | [passed](operator_peer/metal__bf16__matmul__strided__lhs_m1_k64_n17.json) | 0.3038 / 0.2018 | [passed](program_peer/metal__bf16__matmul__strided__lhs_m1_k64_n17.json) | 0.2446 / 0.2097 |
| `metal/bf16/matmul/strided/lhs_m1_k64_n65` | matrix | 1 × 64 × 65 | strided | correct | [passed](operator_peer/metal__bf16__matmul__strided__lhs_m1_k64_n65.json) | 0.312 / 0.2067 | [passed](program_peer/metal__bf16__matmul__strided__lhs_m1_k64_n65.json) | 0.325 / 0.21 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
