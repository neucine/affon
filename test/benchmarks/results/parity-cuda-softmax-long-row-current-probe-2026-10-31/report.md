# Parity batch report

Target: `NVIDIA GeForce RTX 3090/7.0.0-38-generic/x86_64`  
Catalog SHA-256: `8fdab76c6a0d0cd2f37eb3da4ce689da4c05affaf43ab6370e23df2e49370ad4`  
Binary SHA-256: `d7a1000d84b86986a323c3190ac4b7c642320e3d7a6a3ee2c67dd6eaddc89f80`  
Timing: 10 iterations × 5 samples × 3 rounds

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
| `cuda/bf16/log_softmax/long_row` | normalization | 128 × 512 | dense | correct | [passed](operator_peer/cuda__bf16__log_softmax__long_row.json) | 1.112 / 0.5655 | [passed](program_peer/cuda__bf16__log_softmax__long_row.json) | 1.114 / 0.5682 |
| `cuda/bf16/softmax/long_row` | normalization | 128 × 512 | dense | correct | [passed](operator_peer/cuda__bf16__softmax__long_row.json) | 1.109 / 0.5537 | [passed](program_peer/cuda__bf16__softmax__long_row.json) | 1.109 / 0.5791 |
| `cuda/f32/log_softmax/long_row` | normalization | 128 × 512 | dense | correct | [passed](operator_peer/cuda__f32__log_softmax__long_row.json) | 1.267 / 0.5553 | [passed](program_peer/cuda__f32__log_softmax__long_row.json) | 1.282 / 0.5559 |
| `cuda/f32/softmax/long_row` | normalization | 128 × 512 | dense | correct | [passed](operator_peer/cuda__f32__softmax__long_row.json) | 1.269 / 0.548 | [passed](program_peer/cuda__f32__softmax__long_row.json) | 1.288 / 0.554 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
