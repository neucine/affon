# Parity batch report

Target: `NVIDIA GeForce RTX 3090/7.0.0-42-generic/x86_64`  
Catalog SHA-256: `2040db3f61f58b607df877c381cb82f5e601c35c8718a3841a8d788a08d5b5bb`  
Binary SHA-256: `a27e00d6407d867d4a8150a81875c7f723bff29a6f2f55d22e276cd329e1c144`  
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
| `cuda/f32/matmul/transposed/lhs_m1_k129_n17` | matrix | 1 × 129 × 17 | transposed | correct | [passed](operator_peer/cuda__f32__matmul__transposed__lhs_m1_k129_n17.json) | 0.01953 / 0.01396 | [passed](program_peer/cuda__f32__matmul__transposed__lhs_m1_k129_n17.json) | 0.02514 / 0.01777 |
| `cuda/f32/matmul/transposed/lhs_m1_k64_n65` | matrix | 1 × 64 × 65 | transposed | correct | [passed](operator_peer/cuda__f32__matmul__transposed__lhs_m1_k64_n65.json) | 0.01841 / 0.01449 | [passed](program_peer/cuda__f32__matmul__transposed__lhs_m1_k64_n65.json) | 0.024 / 0.01745 |
| `cuda/f32/matmul/transposed/rhs_m1_k128_n64` | matrix | 1 × 128 × 64 | transposed | correct | [passed](operator_peer/cuda__f32__matmul__transposed__rhs_m1_k128_n64.json) | 0.02331 / 0.01633 | [passed](program_peer/cuda__f32__matmul__transposed__rhs_m1_k128_n64.json) | 0.02874 / 0.01913 |
| `cuda/f32/matmul/transposed/rhs_m1_k31_n17` | matrix | 1 × 31 × 17 | transposed | correct | [passed](operator_peer/cuda__f32__matmul__transposed__rhs_m1_k31_n17.json) | 0.01723 / 0.01419 | [passed](program_peer/cuda__f32__matmul__transposed__rhs_m1_k31_n17.json) | 0.02272 / 0.01774 |
| `cuda/f32/matmul/transposed/rhs_m1_k32_n8` | matrix | 1 × 32 × 8 | transposed | correct | [passed](operator_peer/cuda__f32__matmul__transposed__rhs_m1_k32_n8.json) | 0.017 / 0.01387 | [passed](program_peer/cuda__f32__matmul__transposed__rhs_m1_k32_n8.json) | 0.02266 / 0.01723 |
| `cuda/f32/matmul/transposed/rhs_m1_k64_n17` | matrix | 1 × 64 × 17 | transposed | correct | [passed](operator_peer/cuda__f32__matmul__transposed__rhs_m1_k64_n17.json) | 0.01817 / 0.01448 | [passed](program_peer/cuda__f32__matmul__transposed__rhs_m1_k64_n17.json) | 0.0237 / 0.01754 |
| `cuda/f32/matmul/transposed/rhs_m1_k8_n32` | matrix | 1 × 8 × 32 | transposed | correct | [passed](operator_peer/cuda__f32__matmul__transposed__rhs_m1_k8_n32.json) | 0.01662 / 0.01442 | [passed](program_peer/cuda__f32__matmul__transposed__rhs_m1_k8_n32.json) | 0.02209 / 0.01824 |
| `cuda/f32/matmul/transposed/rhs_m33_k8_n8` | matrix | 33 × 8 × 8 | transposed | correct | [passed](operator_peer/cuda__f32__matmul__transposed__rhs_m33_k8_n8.json) | 0.0242 / 0.01774 | [passed](program_peer/cuda__f32__matmul__transposed__rhs_m33_k8_n8.json) | 0.03053 / 0.02085 |
| `cuda/f32/matmul/small_batch/grid_m2_k127_n129` | matrix | 2 × 127 × 129 | dense | correct | [passed](operator_peer/cuda__f32__matmul__small_batch__grid_m2_k127_n129.json) | 0.02172 / 0.01674 | [passed](program_peer/cuda__f32__matmul__small_batch__grid_m2_k127_n129.json) | 0.02801 / 0.01907 |
| `cuda/f32/matmul/small_batch/grid_m2_k33_n31` | matrix | 2 × 33 × 31 | dense | correct | [passed](operator_peer/cuda__f32__matmul__small_batch__grid_m2_k33_n31.json) | 0.01754 / 0.01379 | [passed](program_peer/cuda__f32__matmul__small_batch__grid_m2_k33_n31.json) | 0.0233 / 0.01786 |
| `cuda/f32/matmul/small_batch/grid_m2_k5_n31` | matrix | 2 × 5 × 31 | dense | correct | [passed](operator_peer/cuda__f32__matmul__small_batch__grid_m2_k5_n31.json) | 0.0171 / 0.01433 | [passed](program_peer/cuda__f32__matmul__small_batch__grid_m2_k5_n31.json) | 0.02267 / 0.01756 |
| `cuda/f32/matmul/small_batch/grid_m2_k65_n7` | matrix | 2 × 65 × 7 | dense | correct | [passed](operator_peer/cuda__f32__matmul__small_batch__grid_m2_k65_n7.json) | 0.01804 / 0.01378 | [passed](program_peer/cuda__f32__matmul__small_batch__grid_m2_k65_n7.json) | 0.02353 / 0.01689 |
| `cuda/f32/matmul/small_batch/grid_m8_k33_n7` | matrix | 8 × 33 × 7 | dense | correct | [passed](operator_peer/cuda__f32__matmul__small_batch__grid_m8_k33_n7.json) | 0.01773 / 0.01413 | [passed](program_peer/cuda__f32__matmul__small_batch__grid_m8_k33_n7.json) | 0.02348 / 0.01798 |
| `cuda/f32/matmul/small_batch/grid_m8_k65_n7` | matrix | 8 × 65 × 7 | dense | correct | [passed](operator_peer/cuda__f32__matmul__small_batch__grid_m8_k65_n7.json) | 0.01863 / 0.01432 | [passed](program_peer/cuda__f32__matmul__small_batch__grid_m8_k65_n7.json) | 0.02436 / 0.01807 |
| `cuda/f32/matmul/small_batch/m2_k31_n31_default` | matrix | 2 × 31 × 31 | dense | correct | [passed](operator_peer/cuda__f32__matmul__small_batch__m2_k31_n31_default.json) | 0.01782 / 0.01419 | [passed](program_peer/cuda__f32__matmul__small_batch__m2_k31_n31_default.json) | 0.02344 / 0.01754 |
| `cuda/f32/matmul/small_batch/m2_k64_n128_dense` | matrix | 2 × 64 × 128 | dense | correct | [passed](operator_peer/cuda__f32__matmul__small_batch__m2_k64_n128_dense.json) | 0.01944 / 0.01644 | [passed](program_peer/cuda__f32__matmul__small_batch__m2_k64_n128_dense.json) | 0.02593 / 0.01984 |
| `cuda/f32/matmul/decode_m1/k63_n64_dense` | matrix | 1 × 63 × 64 | dense | correct | [passed](operator_peer/cuda__f32__matmul__decode_m1__k63_n64_dense.json) | 0.0186 / 0.01386 | [passed](program_peer/cuda__f32__matmul__decode_m1__k63_n64_dense.json) | 0.02413 / 0.01725 |
| `cuda/f32/matmul/decode_m1/k64_n65_dense` | matrix | 1 × 64 × 65 | dense | correct | [passed](operator_peer/cuda__f32__matmul__decode_m1__k64_n65_dense.json) | 0.01833 / 0.01437 | [passed](program_peer/cuda__f32__matmul__decode_m1__k64_n65_dense.json) | 0.02391 / 0.01721 |
| `cuda/f32/matmul/decode_m1/m1_k127_n127_default` | matrix | 1 × 127 × 127 | dense | correct | [passed](operator_peer/cuda__f32__matmul__decode_m1__m1_k127_n127_default.json) | 0.02079 / 0.01459 | [passed](program_peer/cuda__f32__matmul__decode_m1__m1_k127_n127_default.json) | 0.02652 / 0.01821 |
| `cuda/f32/matmul/decode_m1/m1_k128_n128_dense` | matrix | 1 × 128 × 128 | dense | correct | [passed](operator_peer/cuda__f32__matmul__decode_m1__m1_k128_n128_dense.json) | 0.02052 / 0.01477 | [passed](program_peer/cuda__f32__matmul__decode_m1__m1_k128_n128_dense.json) | 0.02598 / 0.01779 |
| `cuda/f32/matmul/decode_m1/m1_k31_n31` | matrix | 1 × 31 × 31 | dense | correct | [passed](operator_peer/cuda__f32__matmul__decode_m1__m1_k31_n31.json) | 0.01758 / 0.01348 | [passed](program_peer/cuda__f32__matmul__decode_m1__m1_k31_n31.json) | 0.02316 / 0.01712 |
| `cuda/f32/matmul/decode_m1/m1_k32_n127_dense` | matrix | 1 × 32 × 127 | dense | correct | [passed](operator_peer/cuda__f32__matmul__decode_m1__m1_k32_n127_dense.json) | 0.01798 / 0.0146 | [passed](program_peer/cuda__f32__matmul__decode_m1__m1_k32_n127_dense.json) | 0.02351 / 0.01806 |
| `cuda/f32/matmul/decode_m1/m1_k32_n33` | matrix | 1 × 32 × 33 | dense | correct | [passed](operator_peer/cuda__f32__matmul__decode_m1__m1_k32_n33.json) | 0.01713 / 0.01359 | [passed](program_peer/cuda__f32__matmul__decode_m1__m1_k32_n33.json) | 0.02283 / 0.01734 |
| `cuda/f32/matmul/decode_m1/m1_k33_n32_default` | matrix | 1 × 33 × 32 | dense | correct | [passed](operator_peer/cuda__f32__matmul__decode_m1__m1_k33_n32_default.json) | 0.01726 / 0.0136 | [passed](program_peer/cuda__f32__matmul__decode_m1__m1_k33_n32_default.json) | 0.02277 / 0.01756 |
| `cuda/f32/matmul/strided/lhs_m16_k129_n17` | matrix | 16 × 129 × 17 | strided | correct | [passed](operator_peer/cuda__f32__matmul__strided__lhs_m16_k129_n17.json) | 0.02961 / 0.02163 | [passed](program_peer/cuda__f32__matmul__strided__lhs_m16_k129_n17.json) | 0.03605 / 0.02603 |
| `cuda/f32/matmul/strided/lhs_m1_k129_n17` | matrix | 1 × 129 × 17 | strided | correct | [passed](operator_peer/cuda__f32__matmul__strided__lhs_m1_k129_n17.json) | 0.0196 / 0.01899 | [passed](program_peer/cuda__f32__matmul__strided__lhs_m1_k129_n17.json) | 0.02505 / 0.02245 |
| `cuda/f32/matmul/strided/lhs_m4_k129_n17` | matrix | 4 × 129 × 17 | strided | correct | [passed](operator_peer/cuda__f32__matmul__strided__lhs_m4_k129_n17.json) | 0.02038 / 0.01917 | [passed](program_peer/cuda__f32__matmul__strided__lhs_m4_k129_n17.json) | 0.02578 / 0.02286 |
| `cuda/f32/matmul/strided/lhs_m4_k129_n65` | matrix | 4 × 129 × 65 | strided | correct | [passed](operator_peer/cuda__f32__matmul__strided__lhs_m4_k129_n65.json) | 0.02166 / 0.02106 | [passed](program_peer/cuda__f32__matmul__strided__lhs_m4_k129_n65.json) | 0.02795 / 0.02491 |
| `cuda/f32/matmul/strided/rhs_m16_k129_n17` | matrix | 16 × 129 × 17 | strided | correct | [passed](operator_peer/cuda__f32__matmul__strided__rhs_m16_k129_n17.json) | 0.02965 / 0.02167 | [passed](program_peer/cuda__f32__matmul__strided__rhs_m16_k129_n17.json) | 0.03599 / 0.02607 |
| `cuda/f32/matmul/strided/rhs_m1_k128_n128` | matrix | 1 × 128 × 128 | strided | correct | [passed](operator_peer/cuda__f32__matmul__strided__rhs_m1_k128_n128.json) | 0.02118 / 0.01966 | [passed](program_peer/cuda__f32__matmul__strided__rhs_m1_k128_n128.json) | 0.02684 / 0.02335 |
| `cuda/f32/matmul/strided/rhs_m1_k128_n32` | matrix | 1 × 128 × 32 | strided | correct | [passed](operator_peer/cuda__f32__matmul__strided__rhs_m1_k128_n32.json) | 0.01966 / 0.01892 | [passed](program_peer/cuda__f32__matmul__strided__rhs_m1_k128_n32.json) | 0.02515 / 0.02282 |
| `cuda/f32/matmul/strided/rhs_m1_k128_n64` | matrix | 1 × 128 × 64 | strided | correct | [passed](operator_peer/cuda__f32__matmul__strided__rhs_m1_k128_n64.json) | 0.01993 / 0.01937 | [passed](program_peer/cuda__f32__matmul__strided__rhs_m1_k128_n64.json) | 0.02546 / 0.02303 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
