# Parity batch report

Target: `NVIDIA GeForce RTX 3090/7.0.0-42-generic/x86_64`  
Catalog SHA-256: `74493413093fa656cdbc0765ef2ca0296edb1657aa44f4243504325c9d06ba49`  
Binary SHA-256: `7e4c26cb930ec660505be0e0e0452a7b29207727b0582819b3a1ee8fc3537447`  
Timing: 10 iterations × 5 samples × 3 rounds

## Checks

- Cases run: 8
- Numerical check passed: 8
- Operator check and timing passed: 8
- Program check and timing passed: 8
- Numerical failures: 0
- Operator failures: 0
- Program failures: 0
- Operator checks pending: 0
- Program checks pending: 0

## Cases

| Case | Family | Shape | Layout | Numerical | Operator | Affon / PyTorch ms | Program | Affon / PyTorch ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `cuda/f32/matmul/decode_m1/k63_n64_default` | matrix | 1 × 63 × 64 | dense | correct | [passed](operator_peer/cuda__f32__matmul__decode_m1__k63_n64_default.json) | 0.01886 / 0.01402 | [passed](program_peer/cuda__f32__matmul__decode_m1__k63_n64_default.json) | 0.02448 / 0.02548 |
| `cuda/f32/matmul/decode_m1/k63_n64_dense` | matrix | 1 × 63 × 64 | dense | correct | [passed](operator_peer/cuda__f32__matmul__decode_m1__k63_n64_dense.json) | 0.01907 / 0.01394 | [passed](program_peer/cuda__f32__matmul__decode_m1__k63_n64_dense.json) | 0.02443 / 0.01723 |
| `cuda/f32/matmul/decode_m1/k65_n65_default` | matrix | 1 × 65 × 65 | dense | correct | [passed](operator_peer/cuda__f32__matmul__decode_m1__k65_n65_default.json) | 0.01871 / 0.01384 | [passed](program_peer/cuda__f32__matmul__decode_m1__k65_n65_default.json) | 0.02423 / 0.01755 |
| `cuda/f32/matmul/decode_m1/k65_n65_dense` | matrix | 1 × 65 × 65 | dense | correct | [passed](operator_peer/cuda__f32__matmul__decode_m1__k65_n65_dense.json) | 0.01874 / 0.01419 | [passed](program_peer/cuda__f32__matmul__decode_m1__k65_n65_dense.json) | 0.02419 / 0.01742 |
| `cuda/f32/matmul/decode_m1/m1_k128_n128_default` | matrix | 1 × 128 × 128 | dense | correct | [passed](operator_peer/cuda__f32__matmul__decode_m1__m1_k128_n128_default.json) | 0.01892 / 0.01455 | [passed](program_peer/cuda__f32__matmul__decode_m1__m1_k128_n128_default.json) | 0.02461 / 0.01771 |
| `cuda/f32/matmul/decode_m1/m1_k128_n128_dense` | matrix | 1 × 128 × 128 | dense | correct | [passed](operator_peer/cuda__f32__matmul__decode_m1__m1_k128_n128_dense.json) | 0.01897 / 0.01443 | [passed](program_peer/cuda__f32__matmul__decode_m1__m1_k128_n128_dense.json) | 0.0246 / 0.01794 |
| `cuda/f32/matmul/decode_m1/m1_k31_n32_default` | matrix | 1 × 31 × 32 | dense | correct | [passed](operator_peer/cuda__f32__matmul__decode_m1__m1_k31_n32_default.json) | 0.01866 / 0.0134 | [passed](program_peer/cuda__f32__matmul__decode_m1__m1_k31_n32_default.json) | 0.02436 / 0.01698 |
| `cuda/f32/matmul/decode_m1/m1_k31_n32_dense` | matrix | 1 × 31 × 32 | dense | correct | [passed](operator_peer/cuda__f32__matmul__decode_m1__m1_k31_n32_dense.json) | 0.01867 / 0.01401 | [passed](program_peer/cuda__f32__matmul__decode_m1__m1_k31_n32_dense.json) | 0.02418 / 0.01699 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
