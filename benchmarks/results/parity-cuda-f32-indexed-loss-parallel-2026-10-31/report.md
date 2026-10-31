# Parity batch report

Target: `NVIDIA GeForce RTX 3090/7.0.0-38-generic/x86_64`  
Catalog SHA-256: `ce8d5ad8eb79f0e2d5738c0dc5a24bfa9b7983527df9cbad2fe36c39eaa317fe`  
Binary SHA-256: `761f64c6961de2b4083e56d05b7f1b38b9942d3855678ddae529eb6c0b244edf`  
Timing: 10 iterations × 5 samples × 3 rounds

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
| `cuda/f32/cross_entropy_indexed/small` | loss | 4 × 8 | dense | correct | [passed](operator_peer/cuda__f32__cross_entropy_indexed__small.json) | 0.01827 / 0.0168 | [passed](program_peer/cuda__f32__cross_entropy_indexed__small.json) | 0.02339 / 0.01981 |
| `cuda/f32/cross_entropy_indexed/large` | loss | 128 × 256 | dense | correct | [passed](operator_peer/cuda__f32__cross_entropy_indexed__large.json) | 0.01903 / 0.0169 | [passed](program_peer/cuda__f32__cross_entropy_indexed__large.json) | 0.02369 / 0.02054 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
