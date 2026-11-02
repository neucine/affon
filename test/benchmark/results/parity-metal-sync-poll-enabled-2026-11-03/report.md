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
| `metal/f32/add/small_dense` | arithmetic | 2 × 4 | dense | correct | [passed](operator_peer/metal__f32__add__small_dense.json) | 0.1921 / 0.1903 | [passed](program_peer/metal__f32__add__small_dense.json) | 0.214 / 0.2049 |
| `metal/bf16/mul/small_dense` | arithmetic | 2 × 4 | dense | correct | [passed](operator_peer/metal__bf16__mul__small_dense.json) | 0.2218 / 0.192 | [passed](program_peer/metal__bf16__mul__small_dense.json) | 0.2409 / 0.206 |
| `metal/f32/matmul/decode_m1` | matrix | 1 × 768 × 768 | dense | correct | [passed](operator_peer/metal__f32__matmul__decode_m1.json) | 0.4892 / 0.2872 | [passed](program_peer/metal__f32__matmul__decode_m1.json) | 0.7911 / 0.2907 |
| `metal/bf16/matmul/decode_m1` | matrix | 1 × 128 × 128 | dense | correct | [passed](operator_peer/metal__bf16__matmul__decode_m1.json) | 0.2687 / 0.2208 | [passed](program_peer/metal__bf16__matmul__decode_m1.json) | 0.2798 / 0.2292 |
| `metal/f32/sum_axis/short_axis` | reduction | 4 × 32 | dense | correct | [passed](operator_peer/metal__f32__sum_axis__short_axis.json) | 0.2437 / 0.2276 | [passed](program_peer/metal__f32__sum_axis__short_axis.json) | 0.2488 / 0.2362 |
| `metal/bf16/softmax/short_row` | normalization | 4 × 32 | dense | correct | [passed](operator_peer/metal__bf16__softmax__short_row.json) | 0.2501 / 0.2595 | [passed](program_peer/metal__bf16__softmax__short_row.json) | 0.2673 / 0.2717 |
| `metal/f32/variance_axis/long_axis` | reduction | 128 × 768 | dense | correct | [passed](operator_peer/metal__f32__variance_axis__long_axis.json) | 0.5317 / 0.2858 | [passed](program_peer/metal__f32__variance_axis__long_axis.json) | 0.8701 / 0.2841 |
| `metal/bf16/index_select/small` | indexing | 8 × 16 | dense | correct | [passed](operator_peer/metal__bf16__index_select__small.json) | 0.4332 / 0.2258 | [passed](program_peer/metal__bf16__index_select__small.json) | 0.4394 / 0.2329 |
| `metal/f32/scatter_add/small/axis0_r3_c3` | indexing | 3 × 3 | dense | correct | [passed](operator_peer/metal__f32__scatter_add__small__axis0_r3_c3.json) | 0.2466 / 0.2683 | [passed](program_peer/metal__f32__scatter_add__small__axis0_r3_c3.json) | 0.2517 / 0.2705 |
| `metal/bf16/topk/small/dense_axis0_r3_c3_k2` | indexing | 3 × 3 | dense | correct | [passed](operator_peer/metal__bf16__topk__small__dense_axis0_r3_c3_k2.json) | 0.2523 / 0.6111 | [passed](program_peer/metal__bf16__topk__small__dense_axis0_r3_c3_k2.json) | 0.2624 / 0.6071 |
| `metal/f32/cross_entropy_indexed/small` | loss | 4 × 8 | dense | correct | [passed](operator_peer/metal__f32__cross_entropy_indexed__small.json) | 0.2598 / 0.3523 | [passed](program_peer/metal__f32__cross_entropy_indexed__small.json) | 0.2555 / 0.3504 |
| `metal/bf16/quantized_matmul/decode_m1/grid_m1_k32_n1_b2_g16` | quantized_matrix | 1 × 32 × 1 | dense | correct | [passed](operator_peer/metal__bf16__quantized_matmul__decode_m1__grid_m1_k32_n1_b2_g16.json) | 0.2462 / 0.3402 | [passed](program_peer/metal__bf16__quantized_matmul__decode_m1__grid_m1_k32_n1_b2_g16.json) | 0.2404 / 0.3386 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
