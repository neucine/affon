# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `c5b016c5a418d1e74a6a5643e9f4e4762debe3c1888cf177863052655ccc4fc7`  
Binary SHA-256: `af1b758d59cd4f42a812cf3dcaae67d3b0f01abca19878e39bc7300e1ae67e35`  
Timing: 5 iterations × 5 samples × 3 rounds

## Coverage

- Candidate cells: 36
- Numerical capability passed: 32
- Operator parity and timing passed: 32
- Program parity and timing passed: 32
- Probe failures or incorrect: 4
- Operator incorrect or error: 0
- Program incorrect or error: 0

## Cells

| Cell | Probe | Operator parity | Affon / peer ms | Program parity | Affon / peer ms |
| --- | --- | --- | --- | --- | --- |
| `metal/bf16/log_softmax_nll/small` | correct | [passed](operator_peer/metal__bf16__log_softmax_nll__small.json) | 0.2644 / 0.3008 | [passed](program_peer/metal__bf16__log_softmax_nll__small.json) | 0.2504 / 0.2867 |
| `metal/bf16/log_softmax_nll/large` | correct | [passed](operator_peer/metal__bf16__log_softmax_nll__large.json) | 0.2248 / 0.2766 | [passed](program_peer/metal__bf16__log_softmax_nll__large.json) | 0.2518 / 0.3712 |
| `metal/bf16/log_softmax_nll/strided` | correct | [passed](operator_peer/metal__bf16__log_softmax_nll__strided.json) | 0.261 / 0.301 | [passed](program_peer/metal__bf16__log_softmax_nll__strided.json) | 0.256 / 0.2939 |
| `metal/bf16/cross_entropy/small` | correct | [passed](operator_peer/metal__bf16__cross_entropy__small.json) | 0.2638 / 0.2994 | [passed](program_peer/metal__bf16__cross_entropy__small.json) | 0.2692 / 0.2935 |
| `metal/bf16/cross_entropy/large` | correct | [passed](operator_peer/metal__bf16__cross_entropy__large.json) | 0.264 / 0.2934 | [passed](program_peer/metal__bf16__cross_entropy__large.json) | 0.272 / 0.3042 |
| `metal/bf16/cross_entropy/strided` | correct | [passed](operator_peer/metal__bf16__cross_entropy__strided.json) | 0.2556 / 0.3025 | [passed](program_peer/metal__bf16__cross_entropy__strided.json) | 0.246 / 0.2932 |
| `metal/bf16/scatter_add/small` | correct | [passed](operator_peer/metal__bf16__scatter_add__small.json) | 0.3724 / 0.2228 | [passed](program_peer/metal__bf16__scatter_add__small.json) | 0.4604 / 0.2182 |
| `metal/bf16/scatter_add/large` | correct | [passed](operator_peer/metal__bf16__scatter_add__large.json) | 0.4906 / 0.2246 | [passed](program_peer/metal__bf16__scatter_add__large.json) | 0.4516 / 0.2117 |
| `metal/bf16/scatter_add/strided` | correct | [passed](operator_peer/metal__bf16__scatter_add__strided.json) | 0.4922 / 0.2208 | [passed](program_peer/metal__bf16__scatter_add__strided.json) | 0.512 / 0.2311 |
| `metal/bf16/topk/small` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/bf16/topk/large` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/bf16/quantized_matmul/decode_m1` | correct | [passed](operator_peer/metal__bf16__quantized_matmul__decode_m1.json) | 0.2482 / 0.2816 | [passed](program_peer/metal__bf16__quantized_matmul__decode_m1.json) | 0.2508 / 0.2694 |
| `metal/bf16/quantized_matmul/small_batch` | correct | [passed](operator_peer/metal__bf16__quantized_matmul__small_batch.json) | 0.2002 / 0.2677 | [passed](program_peer/metal__bf16__quantized_matmul__small_batch.json) | 0.2478 / 0.2674 |
| `metal/bf16/quantized_matmul/matrix` | correct | [passed](operator_peer/metal__bf16__quantized_matmul__matrix.json) | 0.2452 / 0.2757 | [passed](program_peer/metal__bf16__quantized_matmul__matrix.json) | 0.2524 / 0.3612 |
| `metal/bf16/quantized_matmul/transposed` | correct | [passed](operator_peer/metal__bf16__quantized_matmul__transposed.json) | 0.2342 / 0.3147 | [passed](program_peer/metal__bf16__quantized_matmul__transposed.json) | 0.2366 / 0.2784 |
| `metal/bf16/quantized_matmul/ragged` | correct | [passed](operator_peer/metal__bf16__quantized_matmul__ragged.json) | 0.2342 / 0.2715 | [passed](program_peer/metal__bf16__quantized_matmul__ragged.json) | 0.25 / 0.2723 |
| `metal/f32/log_softmax_nll/small` | correct | [passed](operator_peer/metal__f32__log_softmax_nll__small.json) | 0.2598 / 0.25 | [passed](program_peer/metal__f32__log_softmax_nll__small.json) | 0.2278 / 0.2767 |
| `metal/f32/log_softmax_nll/large` | correct | [passed](operator_peer/metal__f32__log_softmax_nll__large.json) | 0.2616 / 0.251 | [passed](program_peer/metal__f32__log_softmax_nll__large.json) | 0.245 / 0.2493 |
| `metal/f32/log_softmax_nll/strided` | correct | [passed](operator_peer/metal__f32__log_softmax_nll__strided.json) | 0.2638 / 0.2463 | [passed](program_peer/metal__f32__log_softmax_nll__strided.json) | 0.2542 / 0.2587 |
| `metal/f32/cross_entropy/small` | correct | [passed](operator_peer/metal__f32__cross_entropy__small.json) | 0.2252 / 0.2643 | [passed](program_peer/metal__f32__cross_entropy__small.json) | 0.2678 / 0.2536 |
| `metal/f32/cross_entropy/large` | correct | [passed](operator_peer/metal__f32__cross_entropy__large.json) | 0.2504 / 0.2488 | [passed](program_peer/metal__f32__cross_entropy__large.json) | 0.2448 / 0.2351 |
| `metal/f32/cross_entropy/strided` | correct | [passed](operator_peer/metal__f32__cross_entropy__strided.json) | 0.275 / 0.4204 | [passed](program_peer/metal__f32__cross_entropy__strided.json) | 0.265 / 0.2536 |
| `metal/f32/scatter_add/small` | correct | [passed](operator_peer/metal__f32__scatter_add__small.json) | 0.2558 / 0.2402 | [passed](program_peer/metal__f32__scatter_add__small.json) | 0.2656 / 0.2301 |
| `metal/f32/scatter_add/large` | correct | [passed](operator_peer/metal__f32__scatter_add__large.json) | 0.2588 / 0.2113 | [passed](program_peer/metal__f32__scatter_add__large.json) | 0.2614 / 0.2226 |
| `metal/f32/scatter_add/strided` | correct | [passed](operator_peer/metal__f32__scatter_add__strided.json) | 0.2564 / 0.2104 | [passed](program_peer/metal__f32__scatter_add__strided.json) | 0.2624 / 0.2205 |
| `metal/f32/topk/small` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/topk/large` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/reduce_to_shape/short_axis` | correct | [passed](operator_peer/metal__f32__reduce_to_shape__short_axis.json) | 0.316 / 0.2088 | [passed](program_peer/metal__f32__reduce_to_shape__short_axis.json) | 0.3166 / 0.2132 |
| `metal/f32/reduce_to_shape/long_axis` | correct | [passed](operator_peer/metal__f32__reduce_to_shape__long_axis.json) | 0.3152 / 0.207 | [passed](program_peer/metal__f32__reduce_to_shape__long_axis.json) | 0.2496 / 0.2258 |
| `metal/f32/reduce_to_shape/outer_axis` | correct | [passed](operator_peer/metal__f32__reduce_to_shape__outer_axis.json) | 0.306 / 0.2079 | [passed](program_peer/metal__f32__reduce_to_shape__outer_axis.json) | 0.2828 / 0.2104 |
| `metal/f32/reduce_to_shape/ragged` | correct | [passed](operator_peer/metal__f32__reduce_to_shape__ragged.json) | 0.3108 / 0.2009 | [passed](program_peer/metal__f32__reduce_to_shape__ragged.json) | 0.2924 / 0.2043 |
| `metal/f32/quantized_matmul/decode_m1` | correct | [passed](operator_peer/metal__f32__quantized_matmul__decode_m1.json) | 0.247 / 0.2453 | [passed](program_peer/metal__f32__quantized_matmul__decode_m1.json) | 0.2456 / 0.2587 |
| `metal/f32/quantized_matmul/small_batch` | correct | [passed](operator_peer/metal__f32__quantized_matmul__small_batch.json) | 0.2494 / 0.2576 | [passed](program_peer/metal__f32__quantized_matmul__small_batch.json) | 0.21 / 0.271 |
| `metal/f32/quantized_matmul/matrix` | correct | [passed](operator_peer/metal__f32__quantized_matmul__matrix.json) | 0.2198 / 0.2429 | [passed](program_peer/metal__f32__quantized_matmul__matrix.json) | 0.2518 / 0.2566 |
| `metal/f32/quantized_matmul/transposed` | correct | [passed](operator_peer/metal__f32__quantized_matmul__transposed.json) | 0.2106 / 0.2532 | [passed](program_peer/metal__f32__quantized_matmul__transposed.json) | 0.2448 / 0.2534 |
| `metal/f32/quantized_matmul/ragged` | correct | [passed](operator_peer/metal__f32__quantized_matmul__ragged.json) | 0.2462 / 0.2454 | [passed](program_peer/metal__f32__quantized_matmul__ragged.json) | 0.2734 / 0.2594 |

Each probe result is retained in [capability.json](capability.json).
Timing ratios are diagnostic; only reports with passing numerical parity are eligible for coverage links.
