# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `a54092d200face11a74b1ff8ce34e1762216b3278e9b8c14e1defdb629f38a59`  
Binary SHA-256: `6e09730aae404f355f91c462cfd343d46c9352868f310ca2d844c85b9e02b8a3`  
Timing: 10 iterations × 5 samples × 3 rounds

## Checks

- Cases run: 12
- Numerical check passed: 12
- Operator check and timing passed: 12
- Program check and timing passed: 12
- Numerical failures: 0
- Operator failures: 0
- Program failures: 0

## Cases

| Case | Family | Shape | Layout | Numerical | Operator | Affon / PyTorch ms | Program | Affon / PyTorch ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `metal/f32/add/small_dense` | arithmetic | 2 × 4 | dense | correct | [passed](operator_peer/metal__f32__add__small_dense.json) | 0.3075 / 0.1973 | [passed](program_peer/metal__f32__add__small_dense.json) | 0.1915 / 0.5434 |
| `metal/bf16/mul/small_dense` | arithmetic | 2 × 4 | dense | correct | [passed](operator_peer/metal__bf16__mul__small_dense.json) | 0.3851 / 0.356 | [passed](program_peer/metal__bf16__mul__small_dense.json) | 0.2822 / 0.1919 |
| `metal/f32/matmul/decode_m1` | matrix | 1 × 768 × 768 | dense | correct | [passed](operator_peer/metal__f32__matmul__decode_m1.json) | 0.6615 / 0.5434 | [passed](program_peer/metal__f32__matmul__decode_m1.json) | 0.5946 / 0.2899 |
| `metal/bf16/matmul/decode_m1` | matrix | 1 × 128 × 128 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1.json) | 0.4977 / 0.4929 | [passed](program_peer/metal__bf16__matmul__decode_m1.json) | 0.2379 / 0.2661 |
| `metal/f32/sum_axis/short_axis` | reduction | 4 × 32 | dense | correct | [passed](operator_peer/metal__f32__sum_axis__short_axis.json) | 0.2448 / 0.2878 | [passed](program_peer/metal__f32__sum_axis__short_axis.json) | 0.477 / 0.3157 |
| `metal/bf16/softmax/short_row` | normalization | 4 × 32 | dense | correct | [passed](operator_peer/metal__bf16__softmax__short_row.json) | 0.3145 / 0.4985 | [passed](program_peer/metal__bf16__softmax__short_row.json) | 0.4662 / 0.3607 |
| `metal/f32/variance_axis/long_axis` | reduction | 128 × 768 | dense | correct | [passed](operator_peer/metal__f32__variance_axis__long_axis.json) | 0.6729 / 0.2681 | [passed](program_peer/metal__f32__variance_axis__long_axis.json) | 0.7228 / 0.4861 |
| `metal/bf16/index_select/small` | indexing | 8 × 16 | dense | correct | [passed](operator_peer/metal__bf16__index_select__small.json) | 0.6896 / 0.3168 | [passed](program_peer/metal__bf16__index_select__small.json) | 0.7574 / 0.5726 |
| `metal/f32/scatter_add/small/axis0_r3_c3` | indexing | 3 × 3 | dense | correct | [passed](operator_peer/metal__f32__scatter_add__small__axis0_r3_c3.json) | 0.2957 / 0.6281 | [passed](program_peer/metal__f32__scatter_add__small__axis0_r3_c3.json) | 0.3211 / 0.2812 |
| `metal/bf16/topk/small/dense_axis0_r3_c3_k2` | indexing | 3 × 3 | dense | correct | [passed](operator_peer/metal__bf16__topk__small__dense_axis0_r3_c3_k2.json) | 0.3216 / 1.098 | [passed](program_peer/metal__bf16__topk__small__dense_axis0_r3_c3_k2.json) | 0.5609 / 0.9656 |
| `metal/f32/cross_entropy_indexed/small` | loss | 4 × 8 | dense | correct | [passed](operator_peer/metal__f32__cross_entropy_indexed__small.json) | 0.5331 / 0.7948 | [passed](program_peer/metal__f32__cross_entropy_indexed__small.json) | 0.236 / 0.364 |
| `metal/bf16/quantized_matmul/decode_m1/grid_m1_k32_n1_b2_g16` | quantized_matrix | 1 × 32 × 1 | dense | correct | [passed](operator_peer/metal__bf16__quantized_matmul__decode_m1__grid_m1_k32_n1_b2_g16.json) | 0.2678 / 0.7008 | [passed](program_peer/metal__bf16__quantized_matmul__decode_m1__grid_m1_k32_n1_b2_g16.json) | 0.1967 / 0.3188 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
