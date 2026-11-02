# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `a54092d200face11a74b1ff8ce34e1762216b3278e9b8c14e1defdb629f38a59`  
Binary SHA-256: `a737d13d173da16aca0fd3caf22d49f9ebb1d93c0c48f5ffb16351071ef20fab`  
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
| `metal/f32/add/small_dense` | arithmetic | 2 × 4 | dense | correct | [passed](operator_peer/metal__f32__add__small_dense.json) | 0.2153 / 0.1821 | [passed](program_peer/metal__f32__add__small_dense.json) | 0.2053 / 0.1963 |
| `metal/bf16/mul/small_dense` | arithmetic | 2 × 4 | dense | correct | [passed](operator_peer/metal__bf16__mul__small_dense.json) | 0.2194 / 0.1869 | [passed](program_peer/metal__bf16__mul__small_dense.json) | 0.2537 / 0.1916 |
| `metal/f32/matmul/decode_m1` | matrix | 1 × 768 × 768 | dense | correct | [passed](operator_peer/metal__f32__matmul__decode_m1.json) | 0.4778 / 0.2905 | [passed](program_peer/metal__f32__matmul__decode_m1.json) | 0.7808 / 0.295 |
| `metal/bf16/matmul/decode_m1` | matrix | 1 × 128 × 128 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1.json) | 0.2652 / 0.1989 | [passed](program_peer/metal__bf16__matmul__decode_m1.json) | 0.2645 / 0.2219 |
| `metal/f32/sum_axis/short_axis` | reduction | 4 × 32 | dense | correct | [passed](operator_peer/metal__f32__sum_axis__short_axis.json) | 0.2459 / 0.2161 | [passed](program_peer/metal__f32__sum_axis__short_axis.json) | 0.25 / 0.1952 |
| `metal/bf16/softmax/short_row` | normalization | 4 × 32 | dense | correct | [passed](operator_peer/metal__bf16__softmax__short_row.json) | 0.2455 / 0.2076 | [passed](program_peer/metal__bf16__softmax__short_row.json) | 0.2473 / 0.257 |
| `metal/f32/variance_axis/long_axis` | reduction | 128 × 768 | dense | correct | [passed](operator_peer/metal__f32__variance_axis__long_axis.json) | 0.4309 / 0.2314 | [passed](program_peer/metal__f32__variance_axis__long_axis.json) | 0.4029 / 0.2206 |
| `metal/bf16/index_select/small` | indexing | 8 × 16 | dense | correct | [passed](operator_peer/metal__bf16__index_select__small.json) | 0.4169 / 0.1997 | [passed](program_peer/metal__bf16__index_select__small.json) | 0.4328 / 0.2347 |
| `metal/f32/scatter_add/small/axis0_r3_c3` | indexing | 3 × 3 | dense | correct | [passed](operator_peer/metal__f32__scatter_add__small__axis0_r3_c3.json) | 0.2338 / 0.2441 | [passed](program_peer/metal__f32__scatter_add__small__axis0_r3_c3.json) | 0.2569 / 0.2438 |
| `metal/bf16/topk/small/dense_axis0_r3_c3_k2` | indexing | 3 × 3 | dense | correct | [passed](operator_peer/metal__bf16__topk__small__dense_axis0_r3_c3_k2.json) | 0.2533 / 0.4429 | [passed](program_peer/metal__bf16__topk__small__dense_axis0_r3_c3_k2.json) | 0.257 / 0.5799 |
| `metal/f32/cross_entropy_indexed/small` | loss | 4 × 8 | dense | correct | [passed](operator_peer/metal__f32__cross_entropy_indexed__small.json) | 0.2489 / 0.3443 | [passed](program_peer/metal__f32__cross_entropy_indexed__small.json) | 0.2798 / 0.2985 |
| `metal/bf16/quantized_matmul/decode_m1/grid_m1_k32_n1_b2_g16` | quantized_matrix | 1 × 32 × 1 | dense | correct | [passed](operator_peer/metal__bf16__quantized_matmul__decode_m1__grid_m1_k32_n1_b2_g16.json) | 0.1871 / 0.2696 | [passed](program_peer/metal__bf16__quantized_matmul__decode_m1__grid_m1_k32_n1_b2_g16.json) | 0.2347 / 0.2565 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
