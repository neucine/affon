# Parity batch report

Target: `NVIDIA GeForce RTX 3090/7.0.0-38-generic/x86_64`  
Catalog SHA-256: `7dc401d0f7aa38c995fabbb81975f003bcb10a4011464f7b03cea92ca422179f`  
Binary SHA-256: `0f3ef85adc62751b6ca858e37533bb8c15db5dcfc478840204a180ad8c9d7ade`  
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
| `cuda/bf16/matmul/decode_m1/k63_n64_default` | matrix | 1 × 63 × 64 | dense | correct | [passed](operator_peer/cuda__bf16__matmul__decode_m1__k63_n64_default.json) | 0.02061 / 0.01408 | [passed](program_peer/cuda__bf16__matmul__decode_m1__k63_n64_default.json) | 0.02405 / 0.01723 |
| `cuda/bf16/matmul/large_matrix/grid_m128_k127_n129` | matrix | 128 × 127 × 129 | dense | correct | [passed](operator_peer/cuda__bf16__matmul__large_matrix__grid_m128_k127_n129.json) | 0.3271 / 0.1621 | [passed](program_peer/cuda__bf16__matmul__large_matrix__grid_m128_k127_n129.json) | 0.3321 / 0.1563 |
| `cuda/bf16/matmul/prefill` | matrix | 32 × 128 × 128 | dense | correct | [passed](operator_peer/cuda__bf16__matmul__prefill.json) | 0.09818 / 0.04914 | [passed](program_peer/cuda__bf16__matmul__prefill.json) | 0.1012 / 0.05282 |
| `cuda/bf16/matmul/prefill/model_768` | matrix | 128 × 768 × 768 | dense | correct | [passed](operator_peer/cuda__bf16__matmul__prefill__model_768.json) | 1.586 / 0.9003 | [passed](program_peer/cuda__bf16__matmul__prefill__model_768.json) | 1.597 / 0.8939 |
| `cuda/bf16/matmul/small_batch/grid_m2_k33_n31` | matrix | 2 × 33 × 31 | dense | correct | [passed](operator_peer/cuda__bf16__matmul__small_batch__grid_m2_k33_n31.json) | 0.02121 / 0.01448 | [passed](program_peer/cuda__bf16__matmul__small_batch__grid_m2_k33_n31.json) | 0.02452 / 0.01805 |
| `cuda/bf16/matmul/small_batch/grid_m2_k5_n7` | matrix | 2 × 5 × 7 | dense | correct | [passed](operator_peer/cuda__bf16__matmul__small_batch__grid_m2_k5_n7.json) | 0.0199 / 0.01392 | [passed](program_peer/cuda__bf16__matmul__small_batch__grid_m2_k5_n7.json) | 0.02327 / 0.01716 |
| `cuda/bf16/matmul/small_batch/grid_m8_k127_n129` | matrix | 8 × 127 × 129 | dense | correct | [passed](operator_peer/cuda__bf16__matmul__small_batch__grid_m8_k127_n129.json) | 0.04117 / 0.02874 | [passed](program_peer/cuda__bf16__matmul__small_batch__grid_m8_k127_n129.json) | 0.04462 / 0.02666 |
| `cuda/bf16/matmul/small_batch/grid_m8_k65_n63` | matrix | 8 × 65 × 63 | dense | correct | [passed](operator_peer/cuda__bf16__matmul__small_batch__grid_m8_k65_n63.json) | 0.0315 / 0.0185 | [passed](program_peer/cuda__bf16__matmul__small_batch__grid_m8_k65_n63.json) | 0.03452 / 0.02195 |
| `cuda/bf16/matmul/strided/lhs_m16_k129_n17` | matrix | 16 × 129 × 17 | strided | correct | [passed](operator_peer/cuda__bf16__matmul__strided__lhs_m16_k129_n17.json) | 0.03273 / 0.02272 | [passed](program_peer/cuda__bf16__matmul__strided__lhs_m16_k129_n17.json) | 0.03641 / 0.02496 |
| `cuda/bf16/matmul/transposed/lhs_m16_k129_n17` | matrix | 16 × 129 × 17 | transposed | correct | [passed](operator_peer/cuda__bf16__matmul__transposed__lhs_m16_k129_n17.json) | 0.03066 / 0.01751 | [passed](program_peer/cuda__bf16__matmul__transposed__lhs_m16_k129_n17.json) | 0.03422 / 0.0209 |
| `cuda/f32/matmul/decode_m1/k63_n64_default` | matrix | 1 × 63 × 64 | dense | correct | [passed](operator_peer/cuda__f32__matmul__decode_m1__k63_n64_default.json) | 0.02077 / 0.01402 | [passed](program_peer/cuda__f32__matmul__decode_m1__k63_n64_default.json) | 0.0266 / 0.01736 |
| `cuda/f32/matmul/large_matrix/grid_m128_k127_n129` | matrix | 128 × 127 × 129 | dense | correct | [passed](operator_peer/cuda__f32__matmul__large_matrix__grid_m128_k127_n129.json) | 0.3341 / 0.1573 | [passed](program_peer/cuda__f32__matmul__large_matrix__grid_m128_k127_n129.json) | 0.341 / 0.1609 |
| `cuda/f32/matmul/prefill` | matrix | 128 × 768 × 768 | dense | correct | [passed](operator_peer/cuda__f32__matmul__prefill.json) | 1.623 / 0.9306 | [passed](program_peer/cuda__f32__matmul__prefill.json) | 1.641 / 0.9039 |
| `cuda/f32/matmul/small_batch/grid_m2_k33_n31` | matrix | 2 × 33 × 31 | dense | correct | [passed](operator_peer/cuda__f32__matmul__small_batch__grid_m2_k33_n31.json) | 0.0213 / 0.01425 | [passed](program_peer/cuda__f32__matmul__small_batch__grid_m2_k33_n31.json) | 0.02695 / 0.0181 |
| `cuda/f32/matmul/small_batch/grid_m2_k5_n7` | matrix | 2 × 5 × 7 | dense | correct | [passed](operator_peer/cuda__f32__matmul__small_batch__grid_m2_k5_n7.json) | 0.02001 / 0.01368 | [passed](program_peer/cuda__f32__matmul__small_batch__grid_m2_k5_n7.json) | 0.02551 / 0.01766 |
| `cuda/f32/matmul/small_batch/grid_m8_k127_n129` | matrix | 8 × 127 × 129 | dense | correct | [passed](operator_peer/cuda__f32__matmul__small_batch__grid_m8_k127_n129.json) | 0.04133 / 0.02462 | [passed](program_peer/cuda__f32__matmul__small_batch__grid_m8_k127_n129.json) | 0.04786 / 0.02956 |
| `cuda/f32/matmul/small_batch/grid_m8_k65_n63` | matrix | 8 × 65 × 63 | dense | correct | [passed](operator_peer/cuda__f32__matmul__small_batch__grid_m8_k65_n63.json) | 0.03128 / 0.01983 | [passed](program_peer/cuda__f32__matmul__small_batch__grid_m8_k65_n63.json) | 0.03761 / 0.02337 |
| `cuda/f32/matmul/strided/lhs_m16_k129_n17` | matrix | 16 × 129 × 17 | strided | correct | [passed](operator_peer/cuda__f32__matmul__strided__lhs_m16_k129_n17.json) | 0.03666 / 0.02188 | [passed](program_peer/cuda__f32__matmul__strided__lhs_m16_k129_n17.json) | 0.04329 / 0.02609 |
| `cuda/f32/matmul/transposed/lhs_m16_k129_n17` | matrix | 16 × 129 × 17 | transposed | correct | [passed](operator_peer/cuda__f32__matmul__transposed__lhs_m16_k129_n17.json) | 0.03277 / 0.02727 | [passed](program_peer/cuda__f32__matmul__transposed__lhs_m16_k129_n17.json) | 0.03905 / 0.02308 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
