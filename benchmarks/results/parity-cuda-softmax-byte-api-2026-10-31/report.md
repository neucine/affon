# Parity batch report

Target: `NVIDIA GeForce RTX 3090/7.0.0-38-generic/x86_64`  
Catalog SHA-256: `8fdab76c6a0d0cd2f37eb3da4ce689da4c05affaf43ab6370e23df2e49370ad4`  
Binary SHA-256: `0c165f3b7327de6659b973611275ee22091ae5ad49b75aa24bcea4e9fc1fe67d`  
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
| `cuda/bf16/log_softmax/long_row` | normalization | 128 × 512 | dense | correct | [passed](operator_peer/cuda__bf16__log_softmax__long_row.json) | 1.094 / 0.557 | [passed](program_peer/cuda__bf16__log_softmax__long_row.json) | 1.094 / 0.564 |
| `cuda/bf16/softmax/long_row` | normalization | 128 × 512 | dense | correct | [passed](operator_peer/cuda__bf16__softmax__long_row.json) | 1.099 / 0.5802 | [passed](program_peer/cuda__bf16__softmax__long_row.json) | 1.094 / 0.5536 |
| `cuda/f32/log_softmax/long_row` | normalization | 128 × 512 | dense | correct | [passed](operator_peer/cuda__f32__log_softmax__long_row.json) | 1.106 / 0.5559 | [passed](program_peer/cuda__f32__log_softmax__long_row.json) | 1.101 / 0.5574 |
| `cuda/f32/softmax/long_row` | normalization | 128 × 512 | dense | correct | [passed](operator_peer/cuda__f32__softmax__long_row.json) | 1.089 / 0.5519 | [passed](program_peer/cuda__f32__softmax__long_row.json) | 1.117 / 0.5815 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
