# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `a54092d200face11a74b1ff8ce34e1762216b3278e9b8c14e1defdb629f38a59`  
Binary SHA-256: `164de0a2a31d0861b68c9ed0b40373fa5509390338cc05ee342db6d317df4f82`  
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
| `metal/f32/add/small_dense` | arithmetic | 2 × 4 | dense | correct | [passed](operator_peer/metal__f32__add__small_dense.json) | 0.2067 / 0.1928 | [passed](program_peer/metal__f32__add__small_dense.json) | 0.261 / 0.2064 |
| `metal/bf16/mul/small_dense` | arithmetic | 2 × 4 | dense | correct | [passed](operator_peer/metal__bf16__mul__small_dense.json) | 0.2392 / 0.273 | [passed](program_peer/metal__bf16__mul__small_dense.json) | 0.2396 / 0.2009 |
| `metal/f32/matmul/decode_m1` | matrix | 1 × 768 × 768 | dense | correct | [passed](operator_peer/metal__f32__matmul__decode_m1.json) | 0.4977 / 0.3002 | [passed](program_peer/metal__f32__matmul__decode_m1.json) | 0.8987 / 0.2895 |
| `metal/bf16/matmul/decode_m1` | matrix | 1 × 128 × 128 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1.json) | 0.2433 / 0.2275 | [passed](program_peer/metal__bf16__matmul__decode_m1.json) | 0.2891 / 0.2366 |
| `metal/f32/sum_axis/short_axis` | reduction | 4 × 32 | dense | correct | [passed](operator_peer/metal__f32__sum_axis__short_axis.json) | 0.268 / 0.2874 | [passed](program_peer/metal__f32__sum_axis__short_axis.json) | 0.2677 / 0.2519 |
| `metal/bf16/softmax/short_row` | normalization | 4 × 32 | dense | correct | [passed](operator_peer/metal__bf16__softmax__short_row.json) | 0.3096 / 0.4712 | [passed](program_peer/metal__bf16__softmax__short_row.json) | 0.2895 / 0.309 |
| `metal/f32/variance_axis/long_axis` | reduction | 128 × 768 | dense | correct | [passed](operator_peer/metal__f32__variance_axis__long_axis.json) | 0.98 / 0.5151 | [passed](program_peer/metal__f32__variance_axis__long_axis.json) | 0.9395 / 0.3391 |
| `metal/bf16/index_select/small` | indexing | 8 × 16 | dense | correct | [passed](operator_peer/metal__bf16__index_select__small.json) | 0.4624 / 0.2271 | [passed](program_peer/metal__bf16__index_select__small.json) | 0.4791 / 0.2509 |
| `metal/f32/scatter_add/small/axis0_r3_c3` | indexing | 3 × 3 | dense | correct | [passed](operator_peer/metal__f32__scatter_add__small__axis0_r3_c3.json) | 0.2644 / 0.3297 | [passed](program_peer/metal__f32__scatter_add__small__axis0_r3_c3.json) | 0.2968 / 0.3646 |
| `metal/bf16/topk/small/dense_axis0_r3_c3_k2` | indexing | 3 × 3 | dense | correct | [passed](operator_peer/metal__bf16__topk__small__dense_axis0_r3_c3_k2.json) | 0.2527 / 0.6149 | [passed](program_peer/metal__bf16__topk__small__dense_axis0_r3_c3_k2.json) | 0.2977 / 0.8459 |
| `metal/f32/cross_entropy_indexed/small` | loss | 4 × 8 | dense | correct | [passed](operator_peer/metal__f32__cross_entropy_indexed__small.json) | 0.2486 / 0.3546 | [passed](program_peer/metal__f32__cross_entropy_indexed__small.json) | 0.2992 / 0.4073 |
| `metal/bf16/quantized_matmul/decode_m1/grid_m1_k32_n1_b2_g16` | quantized_matrix | 1 × 32 × 1 | dense | correct | [passed](operator_peer/metal__bf16__quantized_matmul__decode_m1__grid_m1_k32_n1_b2_g16.json) | 0.2402 / 0.3164 | [passed](program_peer/metal__bf16__quantized_matmul__decode_m1__grid_m1_k32_n1_b2_g16.json) | 0.2818 / 0.3482 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
