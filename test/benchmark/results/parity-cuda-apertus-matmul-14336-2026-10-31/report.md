# Parity batch report

Target: `NVIDIA GeForce RTX 3090/7.0.0-38-generic/x86_64`
Catalog SHA-256: `60b161aa86a091694a27c849a5997aab05ebb3771a335eda3406f6f52c9ee7f6`
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
| `cuda/bf16/matmul/decode_m1_k4096_n14336` | matrix | 1 × 4096 × 14336 | dense | correct | [passed](operator_peer/cuda__bf16__matmul__decode_m1_k4096_n14336.json) | 0.3855 / 0.2711 | [passed](program_peer/cuda__bf16__matmul__decode_m1_k4096_n14336.json) | 0.3917 / 0.2753 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
