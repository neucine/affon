# Parity batch report

Target: `NVIDIA GeForce RTX 3090/7.0.0-38-generic/x86_64`
Catalog SHA-256: `929c109f51f8ffb9f8175bc442a8472d69e48c93dc10b05aec997cd0e73fd933`
Binary SHA-256: `a7016a051fc45cdb985aa05b4c27060cb979a41c6f1087266ecdf1a49ed78bca`
Timing: 3 iterations × 3 samples × 2 rounds

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
| `cuda/bf16/matmul/decode_m1_k4096_n4096` | matrix | 1 × 4096 × 4096 | dense | correct | [passed](operator_peer/cuda__bf16__matmul__decode_m1_k4096_n4096.json) | 0.1277 / 0.09194 | [passed](program_peer/cuda__bf16__matmul__decode_m1_k4096_n4096.json) | 0.1304 / 0.0922 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
