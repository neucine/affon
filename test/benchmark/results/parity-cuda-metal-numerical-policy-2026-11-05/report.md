# Parity batch report

Target: `NVIDIA GeForce RTX 3090/7.0.0-38-generic/x86_64`  
Catalog SHA-256: `ca191b99ce19274b62c608dc2d161e9d9d87012922d903f748d8197f051f7930`  
Binary SHA-256: `9f92b876a43d0ec46a772a25cb9fb696750e171e16da75e9830a077464bad7f9`  
Timing: 10 iterations × 5 samples × 3 rounds

## Checks

- Cases run: 6
- Numerical check passed: 6
- Operator check and timing passed: 6
- Program check and timing passed: 6
- Numerical failures: 0
- Operator failures: 0
- Program failures: 0

## Cases

| Case | Family | Shape | Layout | Numerical | Operator | Affon / PyTorch ms | Program | Affon / PyTorch ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `cuda/bf16/layer_norm/long_row` | normalization | 128 × 512 | dense | correct | [passed](operator_peer/cuda__bf16__layer_norm__long_row.json) | 1.103 / 0.5527 | [passed](program_peer/cuda__bf16__layer_norm__long_row.json) | 1.093 / 0.5554 |
| `cuda/bf16/quantized_matmul/decode_m1` | quantized_matrix | 1 × 32 × 1 | dense | correct | [passed](operator_peer/cuda__bf16__quantized_matmul__decode_m1.json) | 0.02622 / 0.05162 | [passed](program_peer/cuda__bf16__quantized_matmul__decode_m1.json) | 0.03125 / 0.05636 |
| `cuda/bf16/rms_norm/long_row` | normalization | 128 × 512 | dense | correct | [passed](operator_peer/cuda__bf16__rms_norm__long_row.json) | 1.094 / 0.5664 | [passed](program_peer/cuda__bf16__rms_norm__long_row.json) | 1.114 / 0.5676 |
| `cuda/f32/layer_norm/long_row` | normalization | 128 × 512 | dense | correct | [passed](operator_peer/cuda__f32__layer_norm__long_row.json) | 1.11 / 0.547 | [passed](program_peer/cuda__f32__layer_norm__long_row.json) | 1.125 / 0.5547 |
| `cuda/f32/quantized_matmul/decode_m1` | quantized_matrix | 1 × 32 × 1 | dense | correct | [passed](operator_peer/cuda__f32__quantized_matmul__decode_m1.json) | 0.02894 / 0.04815 | [passed](program_peer/cuda__f32__quantized_matmul__decode_m1.json) | 0.03368 / 0.05113 |
| `cuda/f32/rms_norm/long_row` | normalization | 128 × 512 | dense | correct | [passed](operator_peer/cuda__f32__rms_norm__long_row.json) | 1.124 / 0.5669 | [passed](program_peer/cuda__f32__rms_norm__long_row.json) | 1.111 / 0.5889 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
