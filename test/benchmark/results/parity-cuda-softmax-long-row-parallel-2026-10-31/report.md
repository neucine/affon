# Parity batch report

Target: `NVIDIA GeForce RTX 3090/7.0.0-38-generic/x86_64`  
Catalog SHA-256: `8fdab76c6a0d0cd2f37eb3da4ce689da4c05affaf43ab6370e23df2e49370ad4`  
Binary SHA-256: `f098ec8d9a87196ba0855f490aa06c389b3554647a287cfd5ad855a9fe71e762`  
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
| `cuda/bf16/log_softmax/long_row` | normalization | 128 × 512 | dense | correct | [passed](operator_peer/cuda__bf16__log_softmax__long_row.json) | 1.089 / 0.5662 | [passed](program_peer/cuda__bf16__log_softmax__long_row.json) | 1.083 / 0.5708 |
| `cuda/bf16/softmax/long_row` | normalization | 128 × 512 | dense | correct | [passed](operator_peer/cuda__bf16__softmax__long_row.json) | 1.087 / 0.5799 | [passed](program_peer/cuda__bf16__softmax__long_row.json) | 1.082 / 0.5857 |
| `cuda/f32/log_softmax/long_row` | normalization | 128 × 512 | dense | correct | [passed](operator_peer/cuda__f32__log_softmax__long_row.json) | 1.075 / 0.5534 | [passed](program_peer/cuda__f32__log_softmax__long_row.json) | 1.102 / 0.5532 |
| `cuda/f32/softmax/long_row` | normalization | 128 × 512 | dense | correct | [passed](operator_peer/cuda__f32__softmax__long_row.json) | 1.089 / 0.5504 | [passed](program_peer/cuda__f32__softmax__long_row.json) | 1.102 / 0.577 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
