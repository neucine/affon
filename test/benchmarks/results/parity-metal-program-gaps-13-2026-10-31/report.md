# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `f9525ff79503c4bc7424bc04b80bac20a7a33a7e5d635d7f3888b5951afe1a70`  
Binary SHA-256: `057e9fb9bc293dc9eb30a5874e868990448947a4d4863271901d4fd681711bf0`  
Timing: 5 iterations × 5 samples × 3 rounds

## Coverage

- Candidate cells: 13
- Numerical capability passed: 13
- Operator parity and timing passed: 13
- Program parity and timing passed: 10
- Probe failures or incorrect: 0
- Operator incorrect or error: 0
- Program incorrect or error: 3

## Cells

| Cell | Probe | Operator parity | Affon / peer ms | Program parity | Affon / peer ms |
| --- | --- | --- | --- | --- | --- |
| `metal/f32/add/small_dense` | correct | [passed](operator_peer/metal__f32__add__small_dense.json) | 0.2844 / 0.2198 | [passed](program_peer/metal__f32__add__small_dense.json) | 0.2116 / 0.2006 |
| `metal/f32/softmax/short_row` | correct | [passed](operator_peer/metal__f32__softmax__short_row.json) | 0.254 / 0.2567 | [passed](program_peer/metal__f32__softmax__short_row.json) | 0.2582 / 0.2055 |
| `metal/f32/softmax/long_row` | correct | [passed](operator_peer/metal__f32__softmax__long_row.json) | 0.7582 / 0.8899 | [passed](program_peer/metal__f32__softmax__long_row.json) | 0.7988 / 0.9283 |
| `metal/f32/sum_axis/short_axis` | correct | [passed](operator_peer/metal__f32__sum_axis__short_axis.json) | 0.2518 / 0.2143 | [passed](program_peer/metal__f32__sum_axis__short_axis.json) | 0.2914 / 0.2346 |
| `metal/f32/slice/offset` | correct | [passed](operator_peer/metal__f32__slice__offset.json) | 0.0108 / 0.2305 | [passed](program_peer/metal__f32__slice__offset.json) | 0.2408 / 0.202 |
| `metal/f32/permute/offset` | correct | [passed](operator_peer/metal__f32__permute__offset.json) | 0.0224 / 0.237 | [passed](program_peer/metal__f32__permute__offset.json) | 0.2604 / 0.1937 |
| `metal/f32/transpose/offset` | correct | [passed](operator_peer/metal__f32__transpose__offset.json) | 0.0214 / 0.2517 | [passed](program_peer/metal__f32__transpose__offset.json) | 0.2674 / 0.2016 |
| `metal/f32/squeeze/offset` | correct | [passed](operator_peer/metal__f32__squeeze__offset.json) | 0.0078 / 0.2497 | [passed](program_peer/metal__f32__squeeze__offset.json) | 0.219 / 0.1968 |
| `metal/f32/unsqueeze/offset` | correct | [passed](operator_peer/metal__f32__unsqueeze__offset.json) | 0.011 / 0.2628 | [passed](program_peer/metal__f32__unsqueeze__offset.json) | 0.2464 / 0.2034 |
| `metal/f32/matmul/decode_m1/dense_gemv` | correct | [passed](operator_peer/metal__f32__matmul__decode_m1__dense_gemv.json) | 0.387 / 0.3141 | [passed](program_peer/metal__f32__matmul__decode_m1__dense_gemv.json) | 0.3806 / 0.3164 |
| `metal/f32/matmul/batched` | correct | [passed](operator_peer/metal__f32__matmul__batched.json) | 0.2444 / 0.2288 | [incorrect](program_peer_failures/metal__f32__matmul__batched.json) | 0.2714 / 0.2251 |
| `metal/f32/matmul/batched/b3_small` | correct | [passed](operator_peer/metal__f32__matmul__batched__b3_small.json) | 0.2126 / 0.2158 | [incorrect](program_peer_failures/metal__f32__matmul__batched__b3_small.json) | 0.2596 / 0.2713 |
| `metal/f32/matmul/batched/b4_small` | correct | [passed](operator_peer/metal__f32__matmul__batched__b4_small.json) | 0.2524 / 0.2146 | [incorrect](program_peer_failures/metal__f32__matmul__batched__b4_small.json) | 0.26 / 0.2307 |

Each probe result is retained in [capability.json](capability.json).
Timing ratios are diagnostic; only reports with passing numerical parity are eligible for coverage links.
