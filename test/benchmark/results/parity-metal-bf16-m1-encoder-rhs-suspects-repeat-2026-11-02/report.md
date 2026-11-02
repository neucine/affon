# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `a93a241131fb92012e06b9360cf8f88c4affb572ca17b669cc678a12ae22185c`  
Binary SHA-256: `8522698ce689fb48c552a41c4b152287fbec36f469250cc8d198a9b99cd03a8d`  
Timing: 5 iterations × 3 samples × 2 rounds

## Checks

- Cases run: 2
- Numerical check passed: 2
- Operator check and timing passed: 2
- Program check and timing passed: 2
- Numerical failures: 0
- Operator failures: 0
- Program failures: 0

## Cases

| Case | Family | Shape | Layout | Numerical | Operator | Affon / PyTorch ms | Program | Affon / PyTorch ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `metal/bf16/matmul/strided/rhs_m1_k8_n8` | matrix | 1 × 8 × 8 | strided | correct | [passed](operator_peer/metal__bf16__matmul__strided__rhs_m1_k8_n8.json) | 0.2567 / 0.2211 | [passed](program_peer/metal__bf16__matmul__strided__rhs_m1_k8_n8.json) | 0.2624 / 0.2085 |
| `metal/bf16/matmul/strided/rhs_m1_k63_n128` | matrix | 1 × 63 × 128 | strided | correct | [passed](operator_peer/metal__bf16__matmul__strided__rhs_m1_k63_n128.json) | 0.326 / 0.2265 | [passed](program_peer/metal__bf16__matmul__strided__rhs_m1_k63_n128.json) | 0.3778 / 0.2217 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
