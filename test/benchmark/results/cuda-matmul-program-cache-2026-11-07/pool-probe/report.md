# Parity batch report

Target: `NVIDIA GeForce RTX 3090/7.0.0-38-generic/x86_64`  
Catalog SHA-256: `7dc401d0f7aa38c995fabbb81975f003bcb10a4011464f7b03cea92ca422179f`  
Binary SHA-256: `2bc8e0a719c4ea27f7f02dd39ded92fc32fa2c7847ecbb8819a2f81d5ab99e1e`  
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
| `cuda/bf16/matmul/decode_m1/k63_n64_default` | matrix | 1 × 63 × 64 | dense | correct | [passed](operator_peer/cuda__bf16__matmul__decode_m1__k63_n64_default.json) | 0.0207 / 0.01418 | [passed](program_peer/cuda__bf16__matmul__decode_m1__k63_n64_default.json) | 0.02402 / 0.0178 |
| `cuda/bf16/matmul/large_matrix/grid_m128_k127_n129` | matrix | 128 × 127 × 129 | dense | correct | [passed](operator_peer/cuda__bf16__matmul__large_matrix__grid_m128_k127_n129.json) | 0.3269 / 0.1606 | [passed](program_peer/cuda__bf16__matmul__large_matrix__grid_m128_k127_n129.json) | 0.3335 / 0.1576 |
| `cuda/bf16/matmul/prefill` | matrix | 32 × 128 × 128 | dense | correct | [passed](operator_peer/cuda__bf16__matmul__prefill.json) | 0.09817 / 0.04733 | [passed](program_peer/cuda__bf16__matmul__prefill.json) | 0.1011 / 0.05364 |
| `cuda/bf16/matmul/prefill/model_768` | matrix | 128 × 768 × 768 | dense | correct | [passed](operator_peer/cuda__bf16__matmul__prefill__model_768.json) | 1.611 / 0.907 | [passed](program_peer/cuda__bf16__matmul__prefill__model_768.json) | 1.608 / 0.9006 |
| `cuda/bf16/matmul/small_batch/grid_m2_k33_n31` | matrix | 2 × 33 × 31 | dense | correct | [passed](operator_peer/cuda__bf16__matmul__small_batch__grid_m2_k33_n31.json) | 0.02133 / 0.01442 | [passed](program_peer/cuda__bf16__matmul__small_batch__grid_m2_k33_n31.json) | 0.02447 / 0.01741 |
| `cuda/bf16/matmul/small_batch/grid_m2_k5_n7` | matrix | 2 × 5 × 7 | dense | correct | [passed](operator_peer/cuda__bf16__matmul__small_batch__grid_m2_k5_n7.json) | 0.01991 / 0.01364 | [passed](program_peer/cuda__bf16__matmul__small_batch__grid_m2_k5_n7.json) | 0.02315 / 0.01747 |
| `cuda/bf16/matmul/small_batch/grid_m8_k127_n129` | matrix | 8 × 127 × 129 | dense | correct | [passed](operator_peer/cuda__bf16__matmul__small_batch__grid_m8_k127_n129.json) | 0.04084 / 0.02575 | [passed](program_peer/cuda__bf16__matmul__small_batch__grid_m8_k127_n129.json) | 0.0446 / 0.02523 |
| `cuda/bf16/matmul/small_batch/grid_m8_k65_n63` | matrix | 8 × 65 × 63 | dense | correct | [passed](operator_peer/cuda__bf16__matmul__small_batch__grid_m8_k65_n63.json) | 0.03135 / 0.0189 | [passed](program_peer/cuda__bf16__matmul__small_batch__grid_m8_k65_n63.json) | 0.0348 / 0.02168 |
| `cuda/bf16/matmul/strided/lhs_m16_k129_n17` | matrix | 16 × 129 × 17 | strided | correct | [passed](operator_peer/cuda__bf16__matmul__strided__lhs_m16_k129_n17.json) | 0.03311 / 0.0219 | [passed](program_peer/cuda__bf16__matmul__strided__lhs_m16_k129_n17.json) | 0.03628 / 0.02493 |
| `cuda/bf16/matmul/transposed/lhs_m16_k129_n17` | matrix | 16 × 129 × 17 | transposed | correct | [passed](operator_peer/cuda__bf16__matmul__transposed__lhs_m16_k129_n17.json) | 0.0303 / 0.01792 | [passed](program_peer/cuda__bf16__matmul__transposed__lhs_m16_k129_n17.json) | 0.03413 / 0.02136 |
| `cuda/f32/matmul/decode_m1/k63_n64_default` | matrix | 1 × 63 × 64 | dense | correct | [passed](operator_peer/cuda__f32__matmul__decode_m1__k63_n64_default.json) | 0.02095 / 0.01391 | [passed](program_peer/cuda__f32__matmul__decode_m1__k63_n64_default.json) | 0.02685 / 0.01733 |
| `cuda/f32/matmul/large_matrix/grid_m128_k127_n129` | matrix | 128 × 127 × 129 | dense | correct | [passed](operator_peer/cuda__f32__matmul__large_matrix__grid_m128_k127_n129.json) | 0.3317 / 0.1606 | [passed](program_peer/cuda__f32__matmul__large_matrix__grid_m128_k127_n129.json) | 0.3386 / 0.163 |
| `cuda/f32/matmul/prefill` | matrix | 128 × 768 × 768 | dense | correct | [passed](operator_peer/cuda__f32__matmul__prefill.json) | 1.638 / 0.9376 | [passed](program_peer/cuda__f32__matmul__prefill.json) | 1.626 / 0.903 |
| `cuda/f32/matmul/small_batch/grid_m2_k33_n31` | matrix | 2 × 33 × 31 | dense | correct | [passed](operator_peer/cuda__f32__matmul__small_batch__grid_m2_k33_n31.json) | 0.02149 / 0.01405 | [passed](program_peer/cuda__f32__matmul__small_batch__grid_m2_k33_n31.json) | 0.02689 / 0.01798 |
| `cuda/f32/matmul/small_batch/grid_m2_k5_n7` | matrix | 2 × 5 × 7 | dense | correct | [passed](operator_peer/cuda__f32__matmul__small_batch__grid_m2_k5_n7.json) | 0.01977 / 0.01363 | [passed](program_peer/cuda__f32__matmul__small_batch__grid_m2_k5_n7.json) | 0.0256 / 0.01762 |
| `cuda/f32/matmul/small_batch/grid_m8_k127_n129` | matrix | 8 × 127 × 129 | dense | correct | [passed](operator_peer/cuda__f32__matmul__small_batch__grid_m8_k127_n129.json) | 0.04125 / 0.0235 | [passed](program_peer/cuda__f32__matmul__small_batch__grid_m8_k127_n129.json) | 0.04794 / 0.02907 |
| `cuda/f32/matmul/small_batch/grid_m8_k65_n63` | matrix | 8 × 65 × 63 | dense | correct | [passed](operator_peer/cuda__f32__matmul__small_batch__grid_m8_k65_n63.json) | 0.03136 / 0.01937 | [passed](program_peer/cuda__f32__matmul__small_batch__grid_m8_k65_n63.json) | 0.03799 / 0.02179 |
| `cuda/f32/matmul/strided/lhs_m16_k129_n17` | matrix | 16 × 129 × 17 | strided | correct | [passed](operator_peer/cuda__f32__matmul__strided__lhs_m16_k129_n17.json) | 0.03654 / 0.02167 | [passed](program_peer/cuda__f32__matmul__strided__lhs_m16_k129_n17.json) | 0.04305 / 0.02516 |
| `cuda/f32/matmul/transposed/lhs_m16_k129_n17` | matrix | 16 × 129 × 17 | transposed | correct | [passed](operator_peer/cuda__f32__matmul__transposed__lhs_m16_k129_n17.json) | 0.03266 / 0.02756 | [passed](program_peer/cuda__f32__matmul__transposed__lhs_m16_k129_n17.json) | 0.03916 / 0.02228 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
