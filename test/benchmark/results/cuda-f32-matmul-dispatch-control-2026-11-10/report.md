# Parity batch report

Target: `NVIDIA GeForce RTX 3090/7.0.0-42-generic/x86_64`  
Catalog SHA-256: `2040db3f61f58b607df877c381cb82f5e601c35c8718a3841a8d788a08d5b5bb`  
Binary SHA-256: `c33c7b3b4db3388ff9574262f1e0fb938beffc6dfd1921bb455b6a5f7ef892f2`  
Timing: 10 iterations × 5 samples × 3 rounds

## Checks

- Cases run: 32
- Numerical check passed: 32
- Operator check and timing passed: 32
- Program check and timing passed: 32
- Numerical failures: 0
- Operator failures: 0
- Program failures: 0
- Operator checks pending: 0
- Program checks pending: 0

## Cases

| Case | Family | Shape | Layout | Numerical | Operator | Affon / PyTorch ms | Program | Affon / PyTorch ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `cuda/f32/matmul/transposed/lhs_m1_k129_n17` | matrix | 1 × 129 × 17 | transposed | correct | [passed](operator_peer/cuda__f32__matmul__transposed__lhs_m1_k129_n17.json) | 0.01863 / 0.01423 | [passed](program_peer/cuda__f32__matmul__transposed__lhs_m1_k129_n17.json) | 0.02417 / 0.01741 |
| `cuda/f32/matmul/transposed/lhs_m1_k64_n65` | matrix | 1 × 64 × 65 | transposed | correct | [passed](operator_peer/cuda__f32__matmul__transposed__lhs_m1_k64_n65.json) | 0.01873 / 0.01472 | [passed](program_peer/cuda__f32__matmul__transposed__lhs_m1_k64_n65.json) | 0.02447 / 0.01771 |
| `cuda/f32/matmul/transposed/rhs_m1_k128_n64` | matrix | 1 × 128 × 64 | transposed | correct | [passed](operator_peer/cuda__f32__matmul__transposed__rhs_m1_k128_n64.json) | 0.0207 / 0.01556 | [passed](program_peer/cuda__f32__matmul__transposed__rhs_m1_k128_n64.json) | 0.02624 / 0.01912 |
| `cuda/f32/matmul/transposed/rhs_m1_k31_n17` | matrix | 1 × 31 × 17 | transposed | correct | [passed](operator_peer/cuda__f32__matmul__transposed__rhs_m1_k31_n17.json) | 0.01877 / 0.01398 | [passed](program_peer/cuda__f32__matmul__transposed__rhs_m1_k31_n17.json) | 0.02419 / 0.01781 |
| `cuda/f32/matmul/transposed/rhs_m1_k32_n8` | matrix | 1 × 32 × 8 | transposed | correct | [passed](operator_peer/cuda__f32__matmul__transposed__rhs_m1_k32_n8.json) | 0.01844 / 0.01646 | [passed](program_peer/cuda__f32__matmul__transposed__rhs_m1_k32_n8.json) | 0.02387 / 0.01727 |
| `cuda/f32/matmul/transposed/rhs_m1_k64_n17` | matrix | 1 × 64 × 17 | transposed | correct | [passed](operator_peer/cuda__f32__matmul__transposed__rhs_m1_k64_n17.json) | 0.01876 / 0.01385 | [passed](program_peer/cuda__f32__matmul__transposed__rhs_m1_k64_n17.json) | 0.02437 / 0.01738 |
| `cuda/f32/matmul/transposed/rhs_m1_k8_n32` | matrix | 1 × 8 × 32 | transposed | correct | [passed](operator_peer/cuda__f32__matmul__transposed__rhs_m1_k8_n32.json) | 0.01844 / 0.01448 | [passed](program_peer/cuda__f32__matmul__transposed__rhs_m1_k8_n32.json) | 0.02397 / 0.02234 |
| `cuda/f32/matmul/transposed/rhs_m33_k8_n8` | matrix | 33 × 8 × 8 | transposed | correct | [passed](operator_peer/cuda__f32__matmul__transposed__rhs_m33_k8_n8.json) | 0.02404 / 0.01825 | [passed](program_peer/cuda__f32__matmul__transposed__rhs_m33_k8_n8.json) | 0.03031 / 0.02157 |
| `cuda/f32/matmul/small_batch/grid_m2_k127_n129` | matrix | 2 × 127 × 129 | dense | correct | [passed](operator_peer/cuda__f32__matmul__small_batch__grid_m2_k127_n129.json) | 0.02156 / 0.01595 | [passed](program_peer/cuda__f32__matmul__small_batch__grid_m2_k127_n129.json) | 0.02792 / 0.01933 |
| `cuda/f32/matmul/small_batch/grid_m2_k33_n31` | matrix | 2 × 33 × 31 | dense | correct | [passed](operator_peer/cuda__f32__matmul__small_batch__grid_m2_k33_n31.json) | 0.01922 / 0.01381 | [passed](program_peer/cuda__f32__matmul__small_batch__grid_m2_k33_n31.json) | 0.02478 / 0.01775 |
| `cuda/f32/matmul/small_batch/grid_m2_k5_n31` | matrix | 2 × 5 × 31 | dense | correct | [passed](operator_peer/cuda__f32__matmul__small_batch__grid_m2_k5_n31.json) | 0.01928 / 0.01387 | [passed](program_peer/cuda__f32__matmul__small_batch__grid_m2_k5_n31.json) | 0.02466 / 0.01765 |
| `cuda/f32/matmul/small_batch/grid_m2_k65_n7` | matrix | 2 × 65 × 7 | dense | correct | [passed](operator_peer/cuda__f32__matmul__small_batch__grid_m2_k65_n7.json) | 0.01929 / 0.01335 | [passed](program_peer/cuda__f32__matmul__small_batch__grid_m2_k65_n7.json) | 0.02476 / 0.01684 |
| `cuda/f32/matmul/small_batch/grid_m8_k33_n7` | matrix | 8 × 33 × 7 | dense | correct | [passed](operator_peer/cuda__f32__matmul__small_batch__grid_m8_k33_n7.json) | 0.01939 / 0.01393 | [passed](program_peer/cuda__f32__matmul__small_batch__grid_m8_k33_n7.json) | 0.02515 / 0.02355 |
| `cuda/f32/matmul/small_batch/grid_m8_k65_n7` | matrix | 8 × 65 × 7 | dense | correct | [passed](operator_peer/cuda__f32__matmul__small_batch__grid_m8_k65_n7.json) | 0.0197 / 0.01381 | [passed](program_peer/cuda__f32__matmul__small_batch__grid_m8_k65_n7.json) | 0.02534 / 0.01755 |
| `cuda/f32/matmul/small_batch/m2_k31_n31_default` | matrix | 2 × 31 × 31 | dense | correct | [passed](operator_peer/cuda__f32__matmul__small_batch__m2_k31_n31_default.json) | 0.01951 / 0.01375 | [passed](program_peer/cuda__f32__matmul__small_batch__m2_k31_n31_default.json) | 0.02492 / 0.01758 |
| `cuda/f32/matmul/small_batch/m2_k64_n128_dense` | matrix | 2 × 64 × 128 | dense | correct | [passed](operator_peer/cuda__f32__matmul__small_batch__m2_k64_n128_dense.json) | 0.02063 / 0.01635 | [passed](program_peer/cuda__f32__matmul__small_batch__m2_k64_n128_dense.json) | 0.02704 / 0.01935 |
| `cuda/f32/matmul/decode_m1/k63_n64_dense` | matrix | 1 × 63 × 64 | dense | correct | [passed](operator_peer/cuda__f32__matmul__decode_m1__k63_n64_dense.json) | 0.01917 / 0.01389 | [passed](program_peer/cuda__f32__matmul__decode_m1__k63_n64_dense.json) | 0.02441 / 0.01762 |
| `cuda/f32/matmul/decode_m1/k64_n65_dense` | matrix | 1 × 64 × 65 | dense | correct | [passed](operator_peer/cuda__f32__matmul__decode_m1__k64_n65_dense.json) | 0.01874 / 0.01391 | [passed](program_peer/cuda__f32__matmul__decode_m1__k64_n65_dense.json) | 0.02442 / 0.01724 |
| `cuda/f32/matmul/decode_m1/m1_k127_n127_default` | matrix | 1 × 127 × 127 | dense | correct | [passed](operator_peer/cuda__f32__matmul__decode_m1__m1_k127_n127_default.json) | 0.01936 / 0.01499 | [passed](program_peer/cuda__f32__matmul__decode_m1__m1_k127_n127_default.json) | 0.02508 / 0.01758 |
| `cuda/f32/matmul/decode_m1/m1_k128_n128_dense` | matrix | 1 × 128 × 128 | dense | correct | [passed](operator_peer/cuda__f32__matmul__decode_m1__m1_k128_n128_dense.json) | 0.01884 / 0.01476 | [passed](program_peer/cuda__f32__matmul__decode_m1__m1_k128_n128_dense.json) | 0.02465 / 0.01819 |
| `cuda/f32/matmul/decode_m1/m1_k31_n31` | matrix | 1 × 31 × 31 | dense | correct | [passed](operator_peer/cuda__f32__matmul__decode_m1__m1_k31_n31.json) | 0.01872 / 0.01341 | [passed](program_peer/cuda__f32__matmul__decode_m1__m1_k31_n31.json) | 0.02425 / 0.01694 |
| `cuda/f32/matmul/decode_m1/m1_k32_n127_dense` | matrix | 1 × 32 × 127 | dense | correct | [passed](operator_peer/cuda__f32__matmul__decode_m1__m1_k32_n127_dense.json) | 0.01905 / 0.0145 | [passed](program_peer/cuda__f32__matmul__decode_m1__m1_k32_n127_dense.json) | 0.02458 / 0.0179 |
| `cuda/f32/matmul/decode_m1/m1_k32_n33` | matrix | 1 × 32 × 33 | dense | correct | [passed](operator_peer/cuda__f32__matmul__decode_m1__m1_k32_n33.json) | 0.0183 / 0.01359 | [passed](program_peer/cuda__f32__matmul__decode_m1__m1_k32_n33.json) | 0.02384 / 0.01734 |
| `cuda/f32/matmul/decode_m1/m1_k33_n32_default` | matrix | 1 × 33 × 32 | dense | correct | [passed](operator_peer/cuda__f32__matmul__decode_m1__m1_k33_n32_default.json) | 0.01855 / 0.01351 | [passed](program_peer/cuda__f32__matmul__decode_m1__m1_k33_n32_default.json) | 0.02406 / 0.0169 |
| `cuda/f32/matmul/strided/lhs_m16_k129_n17` | matrix | 16 × 129 × 17 | strided | correct | [passed](operator_peer/cuda__f32__matmul__strided__lhs_m16_k129_n17.json) | 0.02949 / 0.02193 | [passed](program_peer/cuda__f32__matmul__strided__lhs_m16_k129_n17.json) | 0.0358 / 0.02505 |
| `cuda/f32/matmul/strided/lhs_m1_k129_n17` | matrix | 1 × 129 × 17 | strided | correct | [passed](operator_peer/cuda__f32__matmul__strided__lhs_m1_k129_n17.json) | 0.02494 / 0.01864 | [passed](program_peer/cuda__f32__matmul__strided__lhs_m1_k129_n17.json) | 0.03039 / 0.02231 |
| `cuda/f32/matmul/strided/lhs_m4_k129_n17` | matrix | 4 × 129 × 17 | strided | correct | [passed](operator_peer/cuda__f32__matmul__strided__lhs_m4_k129_n17.json) | 0.02546 / 0.01897 | [passed](program_peer/cuda__f32__matmul__strided__lhs_m4_k129_n17.json) | 0.03109 / 0.02273 |
| `cuda/f32/matmul/strided/lhs_m4_k129_n65` | matrix | 4 × 129 × 65 | strided | correct | [passed](operator_peer/cuda__f32__matmul__strided__lhs_m4_k129_n65.json) | 0.02883 / 0.0215 | [passed](program_peer/cuda__f32__matmul__strided__lhs_m4_k129_n65.json) | 0.03506 / 0.0248 |
| `cuda/f32/matmul/strided/rhs_m16_k129_n17` | matrix | 16 × 129 × 17 | strided | correct | [passed](operator_peer/cuda__f32__matmul__strided__rhs_m16_k129_n17.json) | 0.02974 / 0.02225 | [passed](program_peer/cuda__f32__matmul__strided__rhs_m16_k129_n17.json) | 0.03583 / 0.02557 |
| `cuda/f32/matmul/strided/rhs_m1_k128_n128` | matrix | 1 × 128 × 128 | strided | correct | [passed](operator_peer/cuda__f32__matmul__strided__rhs_m1_k128_n128.json) | 0.02583 / 0.02069 | [passed](program_peer/cuda__f32__matmul__strided__rhs_m1_k128_n128.json) | 0.03126 / 0.02356 |
| `cuda/f32/matmul/strided/rhs_m1_k128_n32` | matrix | 1 × 128 × 32 | strided | correct | [passed](operator_peer/cuda__f32__matmul__strided__rhs_m1_k128_n32.json) | 0.02495 / 0.01936 | [passed](program_peer/cuda__f32__matmul__strided__rhs_m1_k128_n32.json) | 0.03036 / 0.02265 |
| `cuda/f32/matmul/strided/rhs_m1_k128_n64` | matrix | 1 × 128 × 64 | strided | correct | [passed](operator_peer/cuda__f32__matmul__strided__rhs_m1_k128_n64.json) | 0.02513 / 0.0192 | [passed](program_peer/cuda__f32__matmul__strided__rhs_m1_k128_n64.json) | 0.03041 / 0.02271 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
