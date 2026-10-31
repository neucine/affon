# Parity batch report

Target: `NVIDIA GeForce RTX 3090/7.0.0-38-generic/x86_64`  
Catalog SHA-256: `f44f6e0134beddb27c8db7c4904a1260d6075f6724e7565a716c451b88ccf1eb`  
Binary SHA-256: `d7a1000d84b86986a323c3190ac4b7c642320e3d7a6a3ee2c67dd6eaddc89f80`  
Timing: 10 iterations × 5 samples × 3 rounds

## Checks

- Cases run: 3
- Numerical check passed: 3
- Operator check and timing passed: 3
- Program check and timing passed: 3
- Numerical failures: 0
- Operator failures: 0
- Program failures: 0

## Cases

| Case | Family | Shape | Layout | Numerical | Operator | Affon / PyTorch ms | Program | Affon / PyTorch ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `cuda/bf16/cross_entropy_indexed/small` | loss | 4 × 8 | dense | correct | [passed](operator_peer/cuda__bf16__cross_entropy_indexed__small.json) | 0.02829 / 0.02396 | [passed](program_peer/cuda__bf16__cross_entropy_indexed__small.json) | 0.03351 / 0.02711 |
| `cuda/bf16/cross_entropy_indexed/large` | loss | 128 × 256 | dense | correct | [passed](operator_peer/cuda__bf16__cross_entropy_indexed__large.json) | 0.02822 / 0.02411 | [passed](program_peer/cuda__bf16__cross_entropy_indexed__large.json) | 0.0334 / 0.02693 |
| `cuda/bf16/cross_entropy_indexed/batched` | loss | 16 × 32 | dense | correct | [passed](operator_peer/cuda__bf16__cross_entropy_indexed__batched.json) | 0.02842 / 0.02385 | [passed](program_peer/cuda__bf16__cross_entropy_indexed__batched.json) | 0.03326 / 0.02695 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
