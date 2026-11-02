# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `13f864e58a4e2c8ba91b92e4f3e40195bce616b7b3ca0ed0a7858fd20dc2c557`  
Binary SHA-256: `880455d53f20a11059886477b295503cf1e8928a20ce407ee5beb0548962e4ff`  
Timing: 30 iterations × 5 samples × 3 rounds

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
| `metal/bf16/matmul/strided/r256_k257_n256` | matrix | 256 × 257 × 256 | strided | correct | [passed](operator_peer/metal__bf16__matmul__strided__r256_k257_n256.json) | 1.425 / 1.108 | [passed](program_peer/metal__bf16__matmul__strided__r256_k257_n256.json) | 1.456 / 1.214 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
