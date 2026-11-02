# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `3c1f6d8774d7dcbbce43b92f9c8a6c44300d325d1483add63449461c430db04b`  
Binary SHA-256: `3552c52bc08639089d28f433463f3fb1e1502ad44033b793a0d38ceebf3bb968`  
Timing: 30 iterations × 5 samples × 3 rounds

## Checks

- Cases run: 4
- Numerical check passed: 4
- Operator check and timing passed: 4
- Program check and timing passed: 4
- Numerical failures: 0
- Operator failures: 0
- Program failures: 0

## Cases

| Case | Family | Shape | Layout | Numerical | Operator | Affon / PyTorch ms | Program | Affon / PyTorch ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `metal/bf16/matmul/transposed/rhs_m1_k8_n8` | matrix | 1 × 8 × 8 | transposed | correct | [passed](operator_peer/metal__bf16__matmul__transposed__rhs_m1_k8_n8.json) | 0.1803 / 0.2079 | [passed](program_peer/metal__bf16__matmul__transposed__rhs_m1_k8_n8.json) | 0.2043 / 0.2039 |
| `metal/bf16/matmul/transposed/rhs_m1_k63_n64` | matrix | 1 × 63 × 64 | transposed | correct | [passed](operator_peer/metal__bf16__matmul__transposed__rhs_m1_k63_n64.json) | 0.2263 / 0.1998 | [passed](program_peer/metal__bf16__matmul__transposed__rhs_m1_k63_n64.json) | 0.238 / 0.2036 |
| `metal/bf16/matmul/strided/rhs_m1_k32_n32` | matrix | 1 × 32 × 32 | strided | correct | [passed](operator_peer/metal__bf16__matmul__strided__rhs_m1_k32_n32.json) | 0.2321 / 0.2065 | [passed](program_peer/metal__bf16__matmul__strided__rhs_m1_k32_n32.json) | 0.2324 / 0.211 |
| `metal/bf16/matmul/strided/rhs_m1_k128_n128` | matrix | 1 × 128 × 128 | strided | correct | [passed](operator_peer/metal__bf16__matmul__strided__rhs_m1_k128_n128.json) | 0.2011 / 0.2082 | [passed](program_peer/metal__bf16__matmul__strided__rhs_m1_k128_n128.json) | 0.2201 / 0.2155 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
