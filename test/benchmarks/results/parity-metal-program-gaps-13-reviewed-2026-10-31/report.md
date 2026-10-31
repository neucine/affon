# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `f9525ff79503c4bc7424bc04b80bac20a7a33a7e5d635d7f3888b5951afe1a70`  
Binary SHA-256: `af1b758d59cd4f42a812cf3dcaae67d3b0f01abca19878e39bc7300e1ae67e35`  
Timing: 5 iterations × 5 samples × 3 rounds

## Coverage

- Candidate cells: 13
- Numerical capability passed: 13
- Operator parity and timing passed: 13
- Program parity and timing passed: 13
- Probe failures or incorrect: 0
- Operator incorrect or error: 0
- Program incorrect or error: 0

## Cells

| Cell | Probe | Operator parity | Affon / peer ms | Program parity | Affon / peer ms |
| --- | --- | --- | --- | --- | --- |
| `metal/f32/add/small_dense` | correct | [passed](operator_peer/metal__f32__add__small_dense.json) | 0.278 / 0.2144 | [passed](program_peer/metal__f32__add__small_dense.json) | 0.223 / 0.1933 |
| `metal/f32/softmax/short_row` | correct | [passed](operator_peer/metal__f32__softmax__short_row.json) | 0.2518 / 0.2661 | [passed](program_peer/metal__f32__softmax__short_row.json) | 0.241 / 0.2084 |
| `metal/f32/softmax/long_row` | correct | [passed](operator_peer/metal__f32__softmax__long_row.json) | 0.763 / 0.954 | [passed](program_peer/metal__f32__softmax__long_row.json) | 0.7842 / 0.9465 |
| `metal/f32/sum_axis/short_axis` | correct | [passed](operator_peer/metal__f32__sum_axis__short_axis.json) | 0.252 / 0.2125 | [passed](program_peer/metal__f32__sum_axis__short_axis.json) | 0.25 / 0.2366 |
| `metal/f32/slice/offset` | correct | [passed](operator_peer/metal__f32__slice__offset.json) | 0.0114 / 0.2339 | [passed](program_peer/metal__f32__slice__offset.json) | 0.2458 / 0.2051 |
| `metal/f32/permute/offset` | correct | [passed](operator_peer/metal__f32__permute__offset.json) | 0.0224 / 0.2383 | [passed](program_peer/metal__f32__permute__offset.json) | 0.2492 / 0.2118 |
| `metal/f32/transpose/offset` | correct | [passed](operator_peer/metal__f32__transpose__offset.json) | 0.0226 / 0.2357 | [passed](program_peer/metal__f32__transpose__offset.json) | 0.2622 / 0.2053 |
| `metal/f32/squeeze/offset` | correct | [passed](operator_peer/metal__f32__squeeze__offset.json) | 0.0086 / 0.2444 | [passed](program_peer/metal__f32__squeeze__offset.json) | 0.2148 / 0.1956 |
| `metal/f32/unsqueeze/offset` | correct | [passed](operator_peer/metal__f32__unsqueeze__offset.json) | 0.0118 / 0.2867 | [passed](program_peer/metal__f32__unsqueeze__offset.json) | 0.2428 / 0.1966 |
| `metal/f32/matmul/decode_m1/dense_gemv` | correct | [passed](operator_peer/metal__f32__matmul__decode_m1__dense_gemv.json) | 0.3786 / 0.3204 | [passed](program_peer/metal__f32__matmul__decode_m1__dense_gemv.json) | 0.3812 / 0.3313 |
| `metal/f32/matmul/batched` | correct | [passed](operator_peer/metal__f32__matmul__batched.json) | 0.2616 / 0.2104 | [passed](program_peer/metal__f32__matmul__batched.json) | 0.2754 / 0.2173 |
| `metal/f32/matmul/batched/b3_small` | correct | [passed](operator_peer/metal__f32__matmul__batched__b3_small.json) | 0.2538 / 0.2178 | [passed](program_peer/metal__f32__matmul__batched__b3_small.json) | 0.2694 / 0.2245 |
| `metal/f32/matmul/batched/b4_small` | correct | [passed](operator_peer/metal__f32__matmul__batched__b4_small.json) | 0.2576 / 0.2181 | [passed](program_peer/metal__f32__matmul__batched__b4_small.json) | 0.275 / 0.2226 |

Each probe result is retained in [capability.json](capability.json).
Timing ratios are diagnostic; only reports with passing numerical parity are eligible for coverage links.
