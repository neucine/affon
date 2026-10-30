# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `8c1d65b62f33cfb2bbb4cb46dfc7b687518c20d43961a5a3f885065164bd6eb3`  
Binary SHA-256: `420e5315d554f3c9d08baf19a6f5517b0471f8b2a96377cacc0a6d30be0eda16`  
Timing: 10 iterations × 5 samples × 3 rounds

## Checks

- Cases run: 1
- Numerical check passed: 1
- Operator check and timing passed: 1
- Program check and timing passed: 1
- Numerical failures: 0
- Operator failures: 0
- Program failures: 0

## Cases

| Case | Family | Shape | Layout | Numerical | Operator | Affon / PyTorch ms | Program | Affon / PyTorch ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `metal/f32/reshape/strided/r64_c127` | structural | 64 × 127 | strided | correct | [passed](operator_peer/metal__f32__reshape__strided__r64_c127.json) | 0.3763 / 0.2952 | [passed](program_peer/metal__f32__reshape__strided__r64_c127.json) | 0.3672 / 0.2826 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
