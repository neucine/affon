# Parity batch report

Target: `NVIDIA GeForce RTX 3090/7.0.0-38-generic/x86_64`  
Catalog SHA-256: `d18a1314cde6e75a380e878f1d5d251b1c26abcfe208910bfe64b9037b7502ba`  
Binary SHA-256: `12b96cf6f255799f8b3059588a660354d32d75c2215155388e13a4d6dd838593`  
Timing: 10 iterations × 5 samples × 3 rounds

## Checks

- Cases run: 2
- Numerical check passed: 2
- Operator check and timing passed: 2
- Program check and timing passed: 2
- Numerical failures: 0
- Operator failures: 0
- Program failures: 0
- Operator checks pending: 0
- Program checks pending: 0
- Checkpoint updated: 2026-11-04T20:30:10.631930+00:00

## Cases

| Case | Family | Shape | Layout | Numerical | Operator | Affon / PyTorch ms | Program | Affon / PyTorch ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `cuda/bf16/abs/large_dense` | elementwise_math | 32 × 64 | dense | correct | [passed](operator_peer/cuda__bf16__abs__large_dense.json) | 0.06346 / 0.02859 | [passed](program_peer/cuda__bf16__abs__large_dense.json) | 0.06562 / 0.0334 |
| `cuda/f32/where/broadcast` | comparison | 8 × 16 | dense | correct | [passed](operator_peer/cuda__f32__where__broadcast.json) | 0.03572 / 0.01705 | [passed](program_peer/cuda__f32__where__broadcast.json) | 0.04041 / 0.01941 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
