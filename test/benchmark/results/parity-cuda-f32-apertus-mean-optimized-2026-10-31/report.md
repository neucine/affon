# Parity batch report

Target: `NVIDIA GeForce RTX 3090/7.0.0-38-generic/x86_64`
Catalog SHA-256: `70cc6544b3c45a8dc25a1a10379e42894d91653c884ed10890135bdfa6dc21a5`
Binary SHA-256: `0d97a08a8fc20545876748d34e18370cc3f174df66bf6aa5567e64b87ced6a50`
Timing: 5 iterations × 3 samples × 2 rounds

## Checks

- Cases run: 4
- Numerical check passed: 4
- Operator check and timing passed: 4
- Program check and timing passed: 4
- Numerical failures: 0
- Operator failures: 0
- Program failures: 0

## Cases

| Case | Family | Shape | Layout | Numerical | Operator | Affon / PyTorch ms | Program | Affon / PyTorch ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `cuda/f32/mean_all/rank1/width_32` | reduction | 1 × 32 | dense | correct | [passed](operator_peer/cuda__f32__mean_all__rank1__width_32.json) | 0.02325 / 0.01389 | [passed](program_peer/cuda__f32__mean_all__rank1__width_32.json) | 0.02815 / 0.02553 |
| `cuda/f32/mean_all/rank1/width_128` | reduction | 1 × 128 | dense | correct | [passed](operator_peer/cuda__f32__mean_all__rank1__width_128.json) | 0.02345 / 0.01416 | [passed](program_peer/cuda__f32__mean_all__rank1__width_128.json) | 0.0285 / 0.01753 |
| `cuda/f32/mean_all/rank1/width_4096` | reduction | 1 × 4096 | dense | correct | [passed](operator_peer/cuda__f32__mean_all__rank1__width_4096.json) | 0.03106 / 0.01445 | [passed](program_peer/cuda__f32__mean_all__rank1__width_4096.json) | 0.03775 / 0.01752 |
| `cuda/f32/mean_all/rank1/width_8192` | reduction | 1 × 8192 | dense | correct | [passed](operator_peer/cuda__f32__mean_all__rank1__width_8192.json) | 0.03833 / 0.01436 | [passed](program_peer/cuda__f32__mean_all__rank1__width_8192.json) | 0.04728 / 0.0169 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
