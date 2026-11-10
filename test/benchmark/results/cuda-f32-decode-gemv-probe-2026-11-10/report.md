# Parity batch report

Target: `NVIDIA GeForce RTX 3090/7.0.0-42-generic/x86_64`  
Catalog SHA-256: `74493413093fa656cdbc0765ef2ca0296edb1657aa44f4243504325c9d06ba49`  
Binary SHA-256: `010341c39969b2fba715807120e6b5b4c230b42005de0b05a99f990a3bfa268e`  
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
| `cuda/f32/matmul/decode_m1/k63_n64_default` | matrix | 1 × 63 × 64 | dense | correct | [passed](operator_peer/cuda__f32__matmul__decode_m1__k63_n64_default.json) | 0.0172 / 0.01399 | [passed](program_peer/cuda__f32__matmul__decode_m1__k63_n64_default.json) | 0.02277 / 0.01776 |
| `cuda/f32/matmul/decode_m1/k63_n64_dense` | matrix | 1 × 63 × 64 | dense | correct | [passed](operator_peer/cuda__f32__matmul__decode_m1__k63_n64_dense.json) | 0.01721 / 0.01392 | [passed](program_peer/cuda__f32__matmul__decode_m1__k63_n64_dense.json) | 0.02267 / 0.01742 |
| `cuda/f32/matmul/decode_m1/k65_n65_default` | matrix | 1 × 65 × 65 | dense | correct | [passed](operator_peer/cuda__f32__matmul__decode_m1__k65_n65_default.json) | 0.01735 / 0.01387 | [passed](program_peer/cuda__f32__matmul__decode_m1__k65_n65_default.json) | 0.02277 / 0.01762 |
| `cuda/f32/matmul/decode_m1/k65_n65_dense` | matrix | 1 × 65 × 65 | dense | correct | [passed](operator_peer/cuda__f32__matmul__decode_m1__k65_n65_dense.json) | 0.01733 / 0.01736 | [passed](program_peer/cuda__f32__matmul__decode_m1__k65_n65_dense.json) | 0.02285 / 0.01751 |
| `cuda/f32/matmul/decode_m1/m1_k128_n128_default` | matrix | 1 × 128 × 128 | dense | correct | [passed](operator_peer/cuda__f32__matmul__decode_m1__m1_k128_n128_default.json) | 0.01786 / 0.01457 | [passed](program_peer/cuda__f32__matmul__decode_m1__m1_k128_n128_default.json) | 0.02333 / 0.01797 |
| `cuda/f32/matmul/decode_m1/m1_k128_n128_dense` | matrix | 1 × 128 × 128 | dense | correct | [passed](operator_peer/cuda__f32__matmul__decode_m1__m1_k128_n128_dense.json) | 0.01789 / 0.01465 | [passed](program_peer/cuda__f32__matmul__decode_m1__m1_k128_n128_dense.json) | 0.02339 / 0.01859 |
| `cuda/f32/matmul/decode_m1/m1_k31_n32_default` | matrix | 1 × 31 × 32 | dense | correct | [passed](operator_peer/cuda__f32__matmul__decode_m1__m1_k31_n32_default.json) | 0.01676 / 0.01406 | [passed](program_peer/cuda__f32__matmul__decode_m1__m1_k31_n32_default.json) | 0.02223 / 0.01706 |
| `cuda/f32/matmul/decode_m1/m1_k31_n32_dense` | matrix | 1 × 31 × 32 | dense | correct | [passed](operator_peer/cuda__f32__matmul__decode_m1__m1_k31_n32_dense.json) | 0.01671 / 0.01367 | [passed](program_peer/cuda__f32__matmul__decode_m1__m1_k31_n32_dense.json) | 0.02221 / 0.01753 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
