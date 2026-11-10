# Parity batch report

Target: `NVIDIA GeForce RTX 3090/7.0.0-42-generic/x86_64`  
Catalog SHA-256: `74493413093fa656cdbc0765ef2ca0296edb1657aa44f4243504325c9d06ba49`  
Binary SHA-256: `dce57e5223002a1e77668a3869779c27ba584e739ae57afd35f16f0af3e25aa5`  
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
| `cuda/f32/matmul/decode_m1/k63_n64_default` | matrix | 1 × 63 × 64 | dense | correct | [passed](operator_peer/cuda__f32__matmul__decode_m1__k63_n64_default.json) | 0.02103 / 0.01393 | [passed](program_peer/cuda__f32__matmul__decode_m1__k63_n64_default.json) | 0.02654 / 0.0174 |
| `cuda/f32/matmul/decode_m1/k63_n64_dense` | matrix | 1 × 63 × 64 | dense | correct | [passed](operator_peer/cuda__f32__matmul__decode_m1__k63_n64_dense.json) | 0.02101 / 0.01429 | [passed](program_peer/cuda__f32__matmul__decode_m1__k63_n64_dense.json) | 0.02653 / 0.01788 |
| `cuda/f32/matmul/decode_m1/k65_n65_default` | matrix | 1 × 65 × 65 | dense | correct | [passed](operator_peer/cuda__f32__matmul__decode_m1__k65_n65_default.json) | 0.02121 / 0.01385 | [passed](program_peer/cuda__f32__matmul__decode_m1__k65_n65_default.json) | 0.02675 / 0.01737 |
| `cuda/f32/matmul/decode_m1/k65_n65_dense` | matrix | 1 × 65 × 65 | dense | correct | [passed](operator_peer/cuda__f32__matmul__decode_m1__k65_n65_dense.json) | 0.02124 / 0.01391 | [passed](program_peer/cuda__f32__matmul__decode_m1__k65_n65_dense.json) | 0.02686 / 0.01744 |
| `cuda/f32/matmul/decode_m1/m1_k128_n128_default` | matrix | 1 × 128 × 128 | dense | correct | [passed](operator_peer/cuda__f32__matmul__decode_m1__m1_k128_n128_default.json) | 0.02548 / 0.01467 | [passed](program_peer/cuda__f32__matmul__decode_m1__m1_k128_n128_default.json) | 0.03099 / 0.01801 |
| `cuda/f32/matmul/decode_m1/m1_k128_n128_dense` | matrix | 1 × 128 × 128 | dense | correct | [passed](operator_peer/cuda__f32__matmul__decode_m1__m1_k128_n128_dense.json) | 0.02558 / 0.01455 | [passed](program_peer/cuda__f32__matmul__decode_m1__m1_k128_n128_dense.json) | 0.03124 / 0.01773 |
| `cuda/f32/matmul/decode_m1/m1_k31_n32_default` | matrix | 1 × 31 × 32 | dense | correct | [passed](operator_peer/cuda__f32__matmul__decode_m1__m1_k31_n32_default.json) | 0.01896 / 0.01353 | [passed](program_peer/cuda__f32__matmul__decode_m1__m1_k31_n32_default.json) | 0.02427 / 0.01693 |
| `cuda/f32/matmul/decode_m1/m1_k31_n32_dense` | matrix | 1 × 31 × 32 | dense | correct | [passed](operator_peer/cuda__f32__matmul__decode_m1__m1_k31_n32_dense.json) | 0.01874 / 0.01409 | [passed](program_peer/cuda__f32__matmul__decode_m1__m1_k31_n32_dense.json) | 0.02434 / 0.01719 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
