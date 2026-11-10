# Parity batch report

Target: `NVIDIA GeForce RTX 3090/7.0.0-38-generic/x86_64`  
Catalog SHA-256: `7dc401d0f7aa38c995fabbb81975f003bcb10a4011464f7b03cea92ca422179f`  
Binary SHA-256: `a4075cc5af98f31db3ac53ba72a21d4fc5800648f2d5c9eb5bee1160c3756ec2`  
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
| `cuda/bf16/matmul/decode_m1/k63_n64_default` | matrix | 1 × 63 × 64 | dense | correct | [passed](operator_peer/cuda__bf16__matmul__decode_m1__k63_n64_default.json) | 0.02216 / 0.01397 | [passed](program_peer/cuda__bf16__matmul__decode_m1__k63_n64_default.json) | 0.02554 / 0.01745 |
| `cuda/bf16/matmul/large_matrix/grid_m128_k127_n129` | matrix | 128 × 127 × 129 | dense | correct | [passed](operator_peer/cuda__bf16__matmul__large_matrix__grid_m128_k127_n129.json) | 0.3262 / 0.1618 | [passed](program_peer/cuda__bf16__matmul__large_matrix__grid_m128_k127_n129.json) | 0.3348 / 0.1597 |
| `cuda/bf16/matmul/prefill` | matrix | 32 × 128 × 128 | dense | correct | [passed](operator_peer/cuda__bf16__matmul__prefill.json) | 0.09755 / 0.05075 | [passed](program_peer/cuda__bf16__matmul__prefill.json) | 0.1018 / 0.0518 |
| `cuda/bf16/matmul/prefill/model_768` | matrix | 128 × 768 × 768 | dense | correct | [passed](operator_peer/cuda__bf16__matmul__prefill__model_768.json) | 1.591 / 0.9312 | [passed](program_peer/cuda__bf16__matmul__prefill__model_768.json) | 1.584 / 0.8956 |
| `cuda/bf16/matmul/small_batch/grid_m2_k33_n31` | matrix | 2 × 33 × 31 | dense | correct | [passed](operator_peer/cuda__bf16__matmul__small_batch__grid_m2_k33_n31.json) | 0.02073 / 0.01428 | [passed](program_peer/cuda__bf16__matmul__small_batch__grid_m2_k33_n31.json) | 0.02402 / 0.01772 |
| `cuda/bf16/matmul/small_batch/grid_m2_k5_n7` | matrix | 2 × 5 × 7 | dense | correct | [passed](operator_peer/cuda__bf16__matmul__small_batch__grid_m2_k5_n7.json) | 0.01846 / 0.0137 | [passed](program_peer/cuda__bf16__matmul__small_batch__grid_m2_k5_n7.json) | 0.02171 / 0.01731 |
| `cuda/bf16/matmul/small_batch/grid_m8_k127_n129` | matrix | 8 × 127 × 129 | dense | correct | [passed](operator_peer/cuda__bf16__matmul__small_batch__grid_m8_k127_n129.json) | 0.04081 / 0.02654 | [passed](program_peer/cuda__bf16__matmul__small_batch__grid_m8_k127_n129.json) | 0.04421 / 0.02581 |
| `cuda/bf16/matmul/small_batch/grid_m8_k65_n63` | matrix | 8 × 65 × 63 | dense | correct | [passed](operator_peer/cuda__bf16__matmul__small_batch__grid_m8_k65_n63.json) | 0.03137 / 0.02227 | [passed](program_peer/cuda__bf16__matmul__small_batch__grid_m8_k65_n63.json) | 0.0348 / 0.02241 |
| `cuda/bf16/matmul/strided/lhs_m16_k129_n17` | matrix | 16 × 129 × 17 | strided | correct | [passed](operator_peer/cuda__bf16__matmul__strided__lhs_m16_k129_n17.json) | 0.03284 / 0.02195 | [passed](program_peer/cuda__bf16__matmul__strided__lhs_m16_k129_n17.json) | 0.03626 / 0.02604 |
| `cuda/bf16/matmul/transposed/lhs_m16_k129_n17` | matrix | 16 × 129 × 17 | transposed | correct | [passed](operator_peer/cuda__bf16__matmul__transposed__lhs_m16_k129_n17.json) | 0.03068 / 0.01737 | [passed](program_peer/cuda__bf16__matmul__transposed__lhs_m16_k129_n17.json) | 0.03425 / 0.02108 |
| `cuda/f32/matmul/decode_m1/k63_n64_default` | matrix | 1 × 63 × 64 | dense | correct | [passed](operator_peer/cuda__f32__matmul__decode_m1__k63_n64_default.json) | 0.02298 / 0.01428 | [passed](program_peer/cuda__f32__matmul__decode_m1__k63_n64_default.json) | 0.02889 / 0.0174 |
| `cuda/f32/matmul/large_matrix/grid_m128_k127_n129` | matrix | 128 × 127 × 129 | dense | correct | [passed](operator_peer/cuda__f32__matmul__large_matrix__grid_m128_k127_n129.json) | 0.3322 / 0.1591 | [passed](program_peer/cuda__f32__matmul__large_matrix__grid_m128_k127_n129.json) | 0.3406 / 0.1582 |
| `cuda/f32/matmul/prefill` | matrix | 128 × 768 × 768 | dense | correct | [passed](operator_peer/cuda__f32__matmul__prefill.json) | 1.64 / 0.9335 | [passed](program_peer/cuda__f32__matmul__prefill.json) | 1.658 / 0.9107 |
| `cuda/f32/matmul/small_batch/grid_m2_k33_n31` | matrix | 2 × 33 × 31 | dense | correct | [passed](operator_peer/cuda__f32__matmul__small_batch__grid_m2_k33_n31.json) | 0.02127 / 0.01412 | [passed](program_peer/cuda__f32__matmul__small_batch__grid_m2_k33_n31.json) | 0.02687 / 0.01805 |
| `cuda/f32/matmul/small_batch/grid_m2_k5_n7` | matrix | 2 × 5 × 7 | dense | correct | [passed](operator_peer/cuda__f32__matmul__small_batch__grid_m2_k5_n7.json) | 0.01866 / 0.01365 | [passed](program_peer/cuda__f32__matmul__small_batch__grid_m2_k5_n7.json) | 0.0241 / 0.01711 |
| `cuda/f32/matmul/small_batch/grid_m8_k127_n129` | matrix | 8 × 127 × 129 | dense | correct | [passed](operator_peer/cuda__f32__matmul__small_batch__grid_m8_k127_n129.json) | 0.04051 / 0.02481 | [passed](program_peer/cuda__f32__matmul__small_batch__grid_m8_k127_n129.json) | 0.04693 / 0.02839 |
| `cuda/f32/matmul/small_batch/grid_m8_k65_n63` | matrix | 8 × 65 × 63 | dense | correct | [passed](operator_peer/cuda__f32__matmul__small_batch__grid_m8_k65_n63.json) | 0.03077 / 0.01935 | [passed](program_peer/cuda__f32__matmul__small_batch__grid_m8_k65_n63.json) | 0.03733 / 0.02301 |
| `cuda/f32/matmul/strided/lhs_m16_k129_n17` | matrix | 16 × 129 × 17 | strided | correct | [passed](operator_peer/cuda__f32__matmul__strided__lhs_m16_k129_n17.json) | 0.03675 / 0.02152 | [passed](program_peer/cuda__f32__matmul__strided__lhs_m16_k129_n17.json) | 0.04327 / 0.02529 |
| `cuda/f32/matmul/transposed/lhs_m16_k129_n17` | matrix | 16 × 129 × 17 | transposed | correct | [passed](operator_peer/cuda__f32__matmul__transposed__lhs_m16_k129_n17.json) | 0.03245 / 0.02628 | [passed](program_peer/cuda__f32__matmul__transposed__lhs_m16_k129_n17.json) | 0.03912 / 0.02319 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
