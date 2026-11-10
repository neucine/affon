# Parity batch report

Target: `NVIDIA GeForce RTX 3090/7.0.0-38-generic/x86_64`  
Catalog SHA-256: `7dc401d0f7aa38c995fabbb81975f003bcb10a4011464f7b03cea92ca422179f`  
Binary SHA-256: `c12eb8cc3336da1c7fa30a3b51e01e6c6a8b84c509b2fdb542fb73d69093e077`  
Timing: 10 iterations × 5 samples × 3 rounds

## Checks

- Cases run: 19
- Numerical check passed: 19
- Operator check and timing passed: 19
- Program check and timing passed: 19
- Numerical failures: 0
- Operator failures: 0
- Program failures: 0
- Operator checks pending: 0
- Program checks pending: 0

## Cases

| Case | Family | Shape | Layout | Numerical | Operator | Affon / PyTorch ms | Program | Affon / PyTorch ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `cuda/bf16/matmul/decode_m1/k63_n64_default` | matrix | 1 × 63 × 64 | dense | correct | [passed](operator_peer/cuda__bf16__matmul__decode_m1__k63_n64_default.json) | 0.01958 / 0.01427 | [passed](program_peer/cuda__bf16__matmul__decode_m1__k63_n64_default.json) | 0.02281 / 0.01761 |
| `cuda/bf16/matmul/large_matrix/grid_m128_k127_n129` | matrix | 128 × 127 × 129 | dense | correct | [passed](operator_peer/cuda__bf16__matmul__large_matrix__grid_m128_k127_n129.json) | 0.324 / 0.1586 | [passed](program_peer/cuda__bf16__matmul__large_matrix__grid_m128_k127_n129.json) | 0.3312 / 0.1577 |
| `cuda/bf16/matmul/prefill` | matrix | 32 × 128 × 128 | dense | correct | [passed](operator_peer/cuda__bf16__matmul__prefill.json) | 0.09722 / 0.05092 | [passed](program_peer/cuda__bf16__matmul__prefill.json) | 0.09938 / 0.05327 |
| `cuda/bf16/matmul/prefill/model_768` | matrix | 128 × 768 × 768 | dense | correct | [passed](operator_peer/cuda__bf16__matmul__prefill__model_768.json) | 1.588 / 0.9249 | [passed](program_peer/cuda__bf16__matmul__prefill__model_768.json) | 1.594 / 0.8866 |
| `cuda/bf16/matmul/small_batch/grid_m2_k33_n31` | matrix | 2 × 33 × 31 | dense | correct | [passed](operator_peer/cuda__bf16__matmul__small_batch__grid_m2_k33_n31.json) | 0.02015 / 0.0143 | [passed](program_peer/cuda__bf16__matmul__small_batch__grid_m2_k33_n31.json) | 0.02305 / 0.01782 |
| `cuda/bf16/matmul/small_batch/grid_m2_k5_n7` | matrix | 2 × 5 × 7 | dense | correct | [passed](operator_peer/cuda__bf16__matmul__small_batch__grid_m2_k5_n7.json) | 0.01892 / 0.0138 | [passed](program_peer/cuda__bf16__matmul__small_batch__grid_m2_k5_n7.json) | 0.02211 / 0.01697 |
| `cuda/bf16/matmul/small_batch/grid_m8_k127_n129` | matrix | 8 × 127 × 129 | dense | correct | [passed](operator_peer/cuda__bf16__matmul__small_batch__grid_m8_k127_n129.json) | 0.0398 / 0.025 | [passed](program_peer/cuda__bf16__matmul__small_batch__grid_m8_k127_n129.json) | 0.0429 / 0.02655 |
| `cuda/bf16/matmul/small_batch/grid_m8_k65_n63` | matrix | 8 × 65 × 63 | dense | correct | [passed](operator_peer/cuda__bf16__matmul__small_batch__grid_m8_k65_n63.json) | 0.03027 / 0.01884 | [passed](program_peer/cuda__bf16__matmul__small_batch__grid_m8_k65_n63.json) | 0.03323 / 0.02342 |
| `cuda/bf16/matmul/strided/lhs_m16_k129_n17` | matrix | 16 × 129 × 17 | strided | correct | [passed](operator_peer/cuda__bf16__matmul__strided__lhs_m16_k129_n17.json) | 0.0317 / 0.022 | [passed](program_peer/cuda__bf16__matmul__strided__lhs_m16_k129_n17.json) | 0.03509 / 0.02524 |
| `cuda/bf16/matmul/transposed/lhs_m16_k129_n17` | matrix | 16 × 129 × 17 | transposed | correct | [passed](operator_peer/cuda__bf16__matmul__transposed__lhs_m16_k129_n17.json) | 0.02956 / 0.01734 | [passed](program_peer/cuda__bf16__matmul__transposed__lhs_m16_k129_n17.json) | 0.03264 / 0.02134 |
| `cuda/f32/matmul/decode_m1/k63_n64_default` | matrix | 1 × 63 × 64 | dense | correct | [passed](operator_peer/cuda__f32__matmul__decode_m1__k63_n64_default.json) | 0.01986 / 0.01379 | [passed](program_peer/cuda__f32__matmul__decode_m1__k63_n64_default.json) | 0.02535 / 0.0172 |
| `cuda/f32/matmul/large_matrix/grid_m128_k127_n129` | matrix | 128 × 127 × 129 | dense | correct | [passed](operator_peer/cuda__f32__matmul__large_matrix__grid_m128_k127_n129.json) | 0.3295 / 0.1566 | [passed](program_peer/cuda__f32__matmul__large_matrix__grid_m128_k127_n129.json) | 0.341 / 0.16 |
| `cuda/f32/matmul/prefill` | matrix | 128 × 768 × 768 | dense | correct | [passed](operator_peer/cuda__f32__matmul__prefill.json) | 1.627 / 0.9312 | [passed](program_peer/cuda__f32__matmul__prefill.json) | 1.632 / 0.9016 |
| `cuda/f32/matmul/small_batch/grid_m2_k33_n31` | matrix | 2 × 33 × 31 | dense | correct | [passed](operator_peer/cuda__f32__matmul__small_batch__grid_m2_k33_n31.json) | 0.02012 / 0.01389 | [passed](program_peer/cuda__f32__matmul__small_batch__grid_m2_k33_n31.json) | 0.02577 / 0.01811 |
| `cuda/f32/matmul/small_batch/grid_m2_k5_n7` | matrix | 2 × 5 × 7 | dense | correct | [passed](operator_peer/cuda__f32__matmul__small_batch__grid_m2_k5_n7.json) | 0.01854 / 0.01341 | [passed](program_peer/cuda__f32__matmul__small_batch__grid_m2_k5_n7.json) | 0.02427 / 0.01742 |
| `cuda/f32/matmul/small_batch/grid_m8_k127_n129` | matrix | 8 × 127 × 129 | dense | correct | [passed](operator_peer/cuda__f32__matmul__small_batch__grid_m8_k127_n129.json) | 0.04026 / 0.0255 | [passed](program_peer/cuda__f32__matmul__small_batch__grid_m8_k127_n129.json) | 0.04619 / 0.02912 |
| `cuda/f32/matmul/small_batch/grid_m8_k65_n63` | matrix | 8 × 65 × 63 | dense | correct | [passed](operator_peer/cuda__f32__matmul__small_batch__grid_m8_k65_n63.json) | 0.02991 / 0.0197 | [passed](program_peer/cuda__f32__matmul__small_batch__grid_m8_k65_n63.json) | 0.03598 / 0.02315 |
| `cuda/f32/matmul/strided/lhs_m16_k129_n17` | matrix | 16 × 129 × 17 | strided | correct | [passed](operator_peer/cuda__f32__matmul__strided__lhs_m16_k129_n17.json) | 0.0356 / 0.02203 | [passed](program_peer/cuda__f32__matmul__strided__lhs_m16_k129_n17.json) | 0.04162 / 0.02615 |
| `cuda/f32/matmul/transposed/lhs_m16_k129_n17` | matrix | 16 × 129 × 17 | transposed | correct | [passed](operator_peer/cuda__f32__matmul__transposed__lhs_m16_k129_n17.json) | 0.03127 / 0.02709 | [passed](program_peer/cuda__f32__matmul__transposed__lhs_m16_k129_n17.json) | 0.03765 / 0.02288 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
