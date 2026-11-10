# Parity batch report

Target: `NVIDIA GeForce RTX 3090/7.0.0-38-generic/x86_64`  
Catalog SHA-256: `7dc401d0f7aa38c995fabbb81975f003bcb10a4011464f7b03cea92ca422179f`  
Binary SHA-256: `f3b481e37eb9d5a16e7a2208f85cf952f644d5b35102d5e989ddacefb345641b`  
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
| `cuda/bf16/matmul/decode_m1/k63_n64_default` | matrix | 1 × 63 × 64 | dense | correct | [passed](operator_peer/cuda__bf16__matmul__decode_m1__k63_n64_default.json) | 0.01973 / 0.01397 | [passed](program_peer/cuda__bf16__matmul__decode_m1__k63_n64_default.json) | 0.02303 / 0.01723 |
| `cuda/bf16/matmul/large_matrix/grid_m128_k127_n129` | matrix | 128 × 127 × 129 | dense | correct | [passed](operator_peer/cuda__bf16__matmul__large_matrix__grid_m128_k127_n129.json) | 0.3294 / 0.161 | [passed](program_peer/cuda__bf16__matmul__large_matrix__grid_m128_k127_n129.json) | 0.3292 / 0.1585 |
| `cuda/bf16/matmul/prefill` | matrix | 32 × 128 × 128 | dense | correct | [passed](operator_peer/cuda__bf16__matmul__prefill.json) | 0.09744 / 0.05254 | [passed](program_peer/cuda__bf16__matmul__prefill.json) | 0.1006 / 0.05244 |
| `cuda/bf16/matmul/prefill/model_768` | matrix | 128 × 768 × 768 | dense | correct | [passed](operator_peer/cuda__bf16__matmul__prefill__model_768.json) | 1.608 / 0.9355 | [passed](program_peer/cuda__bf16__matmul__prefill__model_768.json) | 1.568 / 0.9068 |
| `cuda/bf16/matmul/small_batch/grid_m2_k33_n31` | matrix | 2 × 33 × 31 | dense | correct | [passed](operator_peer/cuda__bf16__matmul__small_batch__grid_m2_k33_n31.json) | 0.02035 / 0.01391 | [passed](program_peer/cuda__bf16__matmul__small_batch__grid_m2_k33_n31.json) | 0.02372 / 0.01768 |
| `cuda/bf16/matmul/small_batch/grid_m2_k5_n7` | matrix | 2 × 5 × 7 | dense | correct | [passed](operator_peer/cuda__bf16__matmul__small_batch__grid_m2_k5_n7.json) | 0.01904 / 0.01351 | [passed](program_peer/cuda__bf16__matmul__small_batch__grid_m2_k5_n7.json) | 0.02232 / 0.01758 |
| `cuda/bf16/matmul/small_batch/grid_m8_k127_n129` | matrix | 8 × 127 × 129 | dense | correct | [passed](operator_peer/cuda__bf16__matmul__small_batch__grid_m8_k127_n129.json) | 0.04008 / 0.02728 | [passed](program_peer/cuda__bf16__matmul__small_batch__grid_m8_k127_n129.json) | 0.04349 / 0.02604 |
| `cuda/bf16/matmul/small_batch/grid_m8_k65_n63` | matrix | 8 × 65 × 63 | dense | correct | [passed](operator_peer/cuda__bf16__matmul__small_batch__grid_m8_k65_n63.json) | 0.03017 / 0.01854 | [passed](program_peer/cuda__bf16__matmul__small_batch__grid_m8_k65_n63.json) | 0.03366 / 0.0217 |
| `cuda/bf16/matmul/strided/lhs_m16_k129_n17` | matrix | 16 × 129 × 17 | strided | correct | [passed](operator_peer/cuda__bf16__matmul__strided__lhs_m16_k129_n17.json) | 0.03178 / 0.02234 | [passed](program_peer/cuda__bf16__matmul__strided__lhs_m16_k129_n17.json) | 0.03513 / 0.02822 |
| `cuda/bf16/matmul/transposed/lhs_m16_k129_n17` | matrix | 16 × 129 × 17 | transposed | correct | [passed](operator_peer/cuda__bf16__matmul__transposed__lhs_m16_k129_n17.json) | 0.02929 / 0.01818 | [passed](program_peer/cuda__bf16__matmul__transposed__lhs_m16_k129_n17.json) | 0.0329 / 0.02183 |
| `cuda/f32/matmul/decode_m1/k63_n64_default` | matrix | 1 × 63 × 64 | dense | correct | [passed](operator_peer/cuda__f32__matmul__decode_m1__k63_n64_default.json) | 0.01989 / 0.01408 | [passed](program_peer/cuda__f32__matmul__decode_m1__k63_n64_default.json) | 0.02524 / 0.01724 |
| `cuda/f32/matmul/large_matrix/grid_m128_k127_n129` | matrix | 128 × 127 × 129 | dense | correct | [passed](operator_peer/cuda__f32__matmul__large_matrix__grid_m128_k127_n129.json) | 0.3312 / 0.1574 | [passed](program_peer/cuda__f32__matmul__large_matrix__grid_m128_k127_n129.json) | 0.3376 / 0.1612 |
| `cuda/f32/matmul/prefill` | matrix | 128 × 768 × 768 | dense | correct | [passed](operator_peer/cuda__f32__matmul__prefill.json) | 1.607 / 0.934 | [passed](program_peer/cuda__f32__matmul__prefill.json) | 1.615 / 0.9131 |
| `cuda/f32/matmul/small_batch/grid_m2_k33_n31` | matrix | 2 × 33 × 31 | dense | correct | [passed](operator_peer/cuda__f32__matmul__small_batch__grid_m2_k33_n31.json) | 0.02036 / 0.01427 | [passed](program_peer/cuda__f32__matmul__small_batch__grid_m2_k33_n31.json) | 0.02571 / 0.01792 |
| `cuda/f32/matmul/small_batch/grid_m2_k5_n7` | matrix | 2 × 5 × 7 | dense | correct | [passed](operator_peer/cuda__f32__matmul__small_batch__grid_m2_k5_n7.json) | 0.0189 / 0.01369 | [passed](program_peer/cuda__f32__matmul__small_batch__grid_m2_k5_n7.json) | 0.02456 / 0.01768 |
| `cuda/f32/matmul/small_batch/grid_m8_k127_n129` | matrix | 8 × 127 × 129 | dense | correct | [passed](operator_peer/cuda__f32__matmul__small_batch__grid_m8_k127_n129.json) | 0.04045 / 0.02294 | [passed](program_peer/cuda__f32__matmul__small_batch__grid_m8_k127_n129.json) | 0.0464 / 0.02851 |
| `cuda/f32/matmul/small_batch/grid_m8_k65_n63` | matrix | 8 × 65 × 63 | dense | correct | [passed](operator_peer/cuda__f32__matmul__small_batch__grid_m8_k65_n63.json) | 0.03037 / 0.01951 | [passed](program_peer/cuda__f32__matmul__small_batch__grid_m8_k65_n63.json) | 0.03668 / 0.02351 |
| `cuda/f32/matmul/strided/lhs_m16_k129_n17` | matrix | 16 × 129 × 17 | strided | correct | [passed](operator_peer/cuda__f32__matmul__strided__lhs_m16_k129_n17.json) | 0.03586 / 0.02178 | [passed](program_peer/cuda__f32__matmul__strided__lhs_m16_k129_n17.json) | 0.04211 / 0.02652 |
| `cuda/f32/matmul/transposed/lhs_m16_k129_n17` | matrix | 16 × 129 × 17 | transposed | correct | [passed](operator_peer/cuda__f32__matmul__transposed__lhs_m16_k129_n17.json) | 0.0316 / 0.02663 | [passed](program_peer/cuda__f32__matmul__transposed__lhs_m16_k129_n17.json) | 0.03824 / 0.02333 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
