# Parity batch report

Target: `NVIDIA GeForce RTX 3090/7.0.0-38-generic/x86_64`
Catalog SHA-256: `70cc6544b3c45a8dc25a1a10379e42894d91653c884ed10890135bdfa6dc21a5`
Binary SHA-256: `dc9721ddc85560d6402c3e666e807aec17725c9261baf4a3b5778df00d2a9958`
Timing: 5 iterations × 3 samples × 2 rounds

## Checks

- Cases run: 4
- Numerical check passed: 3
- Operator check and timing passed: 3
- Program check and timing passed: 3
- Numerical failures: 1
- Operator failures: 0
- Program failures: 0

## Cases

| Case | Family | Shape | Layout | Numerical | Operator | Affon / PyTorch ms | Program | Affon / PyTorch ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `cuda/f32/mean_all/rank1/width_32` | reduction | 1 × 32 | dense | correct | [passed](operator_peer/cuda__f32__mean_all__rank1__width_32.json) | 0.02299 / 0.01412 | [passed](program_peer/cuda__f32__mean_all__rank1__width_32.json) | 0.02837 / 0.01739 |
| `cuda/f32/mean_all/rank1/width_128` | reduction | 1 × 128 | dense | correct | [passed](operator_peer/cuda__f32__mean_all__rank1__width_128.json) | 0.02936 / 0.0141 | [passed](program_peer/cuda__f32__mean_all__rank1__width_128.json) | 0.03429 / 0.01726 |
| `cuda/f32/mean_all/rank1/width_4096` | reduction | 1 × 4096 | dense | correct | [passed](operator_peer/cuda__f32__mean_all__rank1__width_4096.json) | 0.2666 / 0.01438 | [passed](program_peer/cuda__f32__mean_all__rank1__width_4096.json) | 0.274 / 0.01761 |
| `cuda/f32/mean_all/rank1/width_8192` | reduction | 1 × 8192 | dense | incorrect | probe_not_correct | — | probe_not_correct | — |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
