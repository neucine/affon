# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `a3778bf419b4b102a99091cc19c3cb739eee9dc73400d32ab2e58f16e4c105d3`  
Binary SHA-256: `35a6066d3a3f4a07fec5cc7dbeb8caaac66a660948cc409b260487dd9a344024`  
Timing: 5 iterations × 5 samples × 3 rounds

## Coverage

- Candidate cells: 25
- Numerical capability passed: 21
- Operator parity and timing passed: 21
- Program parity and timing passed: 21
- Probe failures or incorrect: 4
- Operator incorrect or error: 0
- Program incorrect or error: 0

## Cells

| Cell | Probe | Operator parity | Affon / peer ms | Program parity | Affon / peer ms |
| --- | --- | --- | --- | --- | --- |
| `metal/bf16/contiguous/view_only` | correct | [passed](operator_peer/metal__bf16__contiguous__view_only.json) | 0.2066 / 0.1791 | [passed](program_peer/metal__bf16__contiguous__view_only.json) | 0.1798 / 0.1774 |
| `metal/f32/contiguous/view_only` | correct | [passed](operator_peer/metal__f32__contiguous__view_only.json) | 0.2362 / 0.1764 | [passed](program_peer/metal__f32__contiguous__view_only.json) | 0.1962 / 0.1781 |
| `metal/f32/add/offset` | correct | [passed](operator_peer/metal__f32__add__offset.json) | 0.211 / 0.2075 | [passed](program_peer/metal__f32__add__offset.json) | 0.2276 / 0.1914 |
| `metal/f32/add/offset/r32_c127` | correct | [passed](operator_peer/metal__f32__add__offset__r32_c127.json) | 0.253 / 0.2366 | [passed](program_peer/metal__f32__add__offset__r32_c127.json) | 0.2836 / 0.2264 |
| `metal/f32/sub/offset` | correct | [passed](operator_peer/metal__f32__sub__offset.json) | 0.247 / 0.1941 | [passed](program_peer/metal__f32__sub__offset.json) | 0.22 / 0.1801 |
| `metal/f32/sub/offset/r32_c127` | correct | [passed](operator_peer/metal__f32__sub__offset__r32_c127.json) | 0.298 / 0.239 | [passed](program_peer/metal__f32__sub__offset__r32_c127.json) | 0.2734 / 0.2321 |
| `metal/f32/mul/offset` | correct | [passed](operator_peer/metal__f32__mul__offset.json) | 0.2324 / 0.1772 | [passed](program_peer/metal__f32__mul__offset.json) | 0.1922 / 0.1902 |
| `metal/f32/mul/offset/r32_c127` | correct | [passed](operator_peer/metal__f32__mul__offset__r32_c127.json) | 0.2966 / 0.2375 | [passed](program_peer/metal__f32__mul__offset__r32_c127.json) | 0.284 / 0.2238 |
| `metal/f32/div/offset` | correct | [passed](operator_peer/metal__f32__div__offset.json) | 0.237 / 0.1953 | [passed](program_peer/metal__f32__div__offset.json) | 0.2278 / 0.187 |
| `metal/f32/div/offset/r32_c127` | correct | [passed](operator_peer/metal__f32__div__offset__r32_c127.json) | 0.2728 / 0.2366 | [passed](program_peer/metal__f32__div__offset__r32_c127.json) | 0.2808 / 0.2245 |
| `metal/f32/eq/offset` | correct | [passed](operator_peer/metal__f32__eq__offset.json) | 0.2418 / 0.2306 | [passed](program_peer/metal__f32__eq__offset.json) | 0.2088 / 0.2032 |
| `metal/f32/eq/offset/r32_c127` | correct | [passed](operator_peer/metal__f32__eq__offset__r32_c127.json) | 0.2868 / 0.2413 | [passed](program_peer/metal__f32__eq__offset__r32_c127.json) | 0.2946 / 0.2262 |
| `metal/f32/lt/offset` | correct | [passed](operator_peer/metal__f32__lt__offset.json) | 0.2082 / 0.2231 | [passed](program_peer/metal__f32__lt__offset.json) | 0.2536 / 0.2258 |
| `metal/f32/lt/offset/r32_c127` | correct | [passed](operator_peer/metal__f32__lt__offset__r32_c127.json) | 0.2686 / 0.2409 | [passed](program_peer/metal__f32__lt__offset__r32_c127.json) | 0.3112 / 0.2396 |
| `metal/f32/gt/offset` | correct | [passed](operator_peer/metal__f32__gt__offset.json) | 0.2314 / 0.2211 | [passed](program_peer/metal__f32__gt__offset.json) | 0.2326 / 0.2116 |
| `metal/f32/gt/offset/r32_c127` | correct | [passed](operator_peer/metal__f32__gt__offset__r32_c127.json) | 0.2984 / 0.2395 | [passed](program_peer/metal__f32__gt__offset__r32_c127.json) | 0.302 / 0.2263 |
| `metal/f32/variance_all/long_axis` | incorrect | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/variance_all/strided/r64_c127` | incorrect | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/std_all/long_axis` | incorrect | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/std_all/strided/r64_c127` | incorrect | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/topk/strided` | correct | [passed](operator_peer/metal__f32__topk__strided.json) | 2.16 / 0.5231 | [passed](program_peer/metal__f32__topk__strided.json) | 2.074 / 0.5055 |
| `metal/bf16/topk/strided` | correct | [passed](operator_peer/metal__bf16__topk__strided.json) | 0.3054 / 0.5999 | [passed](program_peer/metal__bf16__topk__strided.json) | 0.3062 / 0.5864 |
| `metal/f32/quantized_matmul/groups_and_bits` | correct | [passed](operator_peer/metal__f32__quantized_matmul__groups_and_bits.json) | 0.248 / 0.2965 | [passed](program_peer/metal__f32__quantized_matmul__groups_and_bits.json) | 0.2354 / 0.2768 |
| `metal/bf16/quantized_matmul/groups_and_bits` | correct | [passed](operator_peer/metal__bf16__quantized_matmul__groups_and_bits.json) | 0.2272 / 0.2999 | [passed](program_peer/metal__bf16__quantized_matmul__groups_and_bits.json) | 0.2376 / 0.298 |
| `metal/f32/reduce_to_shape/strided` | correct | [passed](operator_peer/metal__f32__reduce_to_shape__strided.json) | 2.256 / 0.2027 | [passed](program_peer/metal__f32__reduce_to_shape__strided.json) | 2.273 / 0.2113 |

Each probe result is retained in [capability.json](capability.json).
Timing ratios are diagnostic; only reports with passing numerical parity are eligible for coverage links.
