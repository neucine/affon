# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `06db80396252c00b8b5b98391d0af4f2d4f9681a11f0a394180b123a368200dc`  
Binary SHA-256: `af1b758d59cd4f42a812cf3dcaae67d3b0f01abca19878e39bc7300e1ae67e35`  
Timing: 5 iterations × 5 samples × 3 rounds

## Coverage

- Candidate cells: 132
- Numerical capability passed: 0
- Operator parity and timing passed: 0
- Program parity and timing passed: 0
- Probe failures or incorrect: 132
- Operator incorrect or error: 0
- Program incorrect or error: 0

## Cells

| Cell | Probe | Operator parity | Affon / peer ms | Program parity | Affon / peer ms |
| --- | --- | --- | --- | --- | --- |
| `metal/f32/cross_entropy/batched` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/cross_entropy/batched/dense_b2_c3_s2` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/cross_entropy/batched/dense_b2_c3_s5` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/cross_entropy/batched/dense_b2_c3_s17` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/cross_entropy/batched/dense_b2_c3_s31` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/cross_entropy/batched/dense_b2_c7_s2` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/cross_entropy/batched/dense_b2_c7_s5` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/cross_entropy/batched/dense_b2_c7_s17` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/cross_entropy/batched/dense_b2_c7_s31` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/cross_entropy/batched/dense_b2_c33_s2` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/cross_entropy/batched/dense_b2_c33_s5` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/cross_entropy/batched/dense_b2_c33_s17` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/cross_entropy/batched/dense_b2_c33_s31` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/cross_entropy/batched/dense_b2_c65_s2` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/cross_entropy/batched/dense_b2_c65_s5` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/cross_entropy/batched/dense_b2_c65_s17` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/cross_entropy/batched/dense_b2_c65_s31` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/cross_entropy/batched/dense_b4_c3_s2` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/cross_entropy/batched/dense_b4_c3_s5` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/cross_entropy/batched/dense_b4_c3_s17` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/cross_entropy/batched/dense_b4_c3_s31` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/cross_entropy/batched/dense_b4_c7_s2` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/cross_entropy/batched/dense_b4_c7_s5` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/cross_entropy/batched/dense_b4_c7_s17` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/cross_entropy/batched/dense_b4_c7_s31` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/cross_entropy/batched/dense_b4_c33_s2` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/cross_entropy/batched/dense_b4_c33_s5` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/cross_entropy/batched/dense_b4_c33_s17` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/cross_entropy/batched/dense_b4_c33_s31` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/cross_entropy/batched/dense_b4_c65_s2` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/cross_entropy/batched/dense_b4_c65_s5` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/cross_entropy/batched/dense_b4_c65_s17` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/cross_entropy/batched/dense_b4_c65_s31` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/log_softmax_nll/batched` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/log_softmax_nll/batched/dense_b2_c3_s2` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/log_softmax_nll/batched/dense_b2_c3_s5` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/log_softmax_nll/batched/dense_b2_c3_s17` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/log_softmax_nll/batched/dense_b2_c3_s31` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/log_softmax_nll/batched/dense_b2_c7_s2` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/log_softmax_nll/batched/dense_b2_c7_s5` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/log_softmax_nll/batched/dense_b2_c7_s17` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/log_softmax_nll/batched/dense_b2_c7_s31` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/log_softmax_nll/batched/dense_b2_c33_s2` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/log_softmax_nll/batched/dense_b2_c33_s5` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/log_softmax_nll/batched/dense_b2_c33_s17` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/log_softmax_nll/batched/dense_b2_c33_s31` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/log_softmax_nll/batched/dense_b2_c65_s2` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/log_softmax_nll/batched/dense_b2_c65_s5` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/log_softmax_nll/batched/dense_b2_c65_s17` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/log_softmax_nll/batched/dense_b2_c65_s31` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/log_softmax_nll/batched/dense_b4_c3_s2` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/log_softmax_nll/batched/dense_b4_c3_s5` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/log_softmax_nll/batched/dense_b4_c3_s17` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/log_softmax_nll/batched/dense_b4_c3_s31` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/log_softmax_nll/batched/dense_b4_c7_s2` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/log_softmax_nll/batched/dense_b4_c7_s5` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/log_softmax_nll/batched/dense_b4_c7_s17` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/log_softmax_nll/batched/dense_b4_c7_s31` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/log_softmax_nll/batched/dense_b4_c33_s2` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/log_softmax_nll/batched/dense_b4_c33_s5` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/log_softmax_nll/batched/dense_b4_c33_s17` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/log_softmax_nll/batched/dense_b4_c33_s31` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/log_softmax_nll/batched/dense_b4_c65_s2` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/log_softmax_nll/batched/dense_b4_c65_s5` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/log_softmax_nll/batched/dense_b4_c65_s17` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/log_softmax_nll/batched/dense_b4_c65_s31` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/bf16/cross_entropy/batched` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/bf16/cross_entropy/batched/dense_b2_c3_s2` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/bf16/cross_entropy/batched/dense_b2_c3_s5` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/bf16/cross_entropy/batched/dense_b2_c3_s17` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/bf16/cross_entropy/batched/dense_b2_c3_s31` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/bf16/cross_entropy/batched/dense_b2_c7_s2` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/bf16/cross_entropy/batched/dense_b2_c7_s5` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/bf16/cross_entropy/batched/dense_b2_c7_s17` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/bf16/cross_entropy/batched/dense_b2_c7_s31` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/bf16/cross_entropy/batched/dense_b2_c33_s2` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/bf16/cross_entropy/batched/dense_b2_c33_s5` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/bf16/cross_entropy/batched/dense_b2_c33_s17` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/bf16/cross_entropy/batched/dense_b2_c33_s31` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/bf16/cross_entropy/batched/dense_b2_c65_s2` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/bf16/cross_entropy/batched/dense_b2_c65_s5` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/bf16/cross_entropy/batched/dense_b2_c65_s17` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/bf16/cross_entropy/batched/dense_b2_c65_s31` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/bf16/cross_entropy/batched/dense_b4_c3_s2` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/bf16/cross_entropy/batched/dense_b4_c3_s5` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/bf16/cross_entropy/batched/dense_b4_c3_s17` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/bf16/cross_entropy/batched/dense_b4_c3_s31` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/bf16/cross_entropy/batched/dense_b4_c7_s2` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/bf16/cross_entropy/batched/dense_b4_c7_s5` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/bf16/cross_entropy/batched/dense_b4_c7_s17` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/bf16/cross_entropy/batched/dense_b4_c7_s31` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/bf16/cross_entropy/batched/dense_b4_c33_s2` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/bf16/cross_entropy/batched/dense_b4_c33_s5` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/bf16/cross_entropy/batched/dense_b4_c33_s17` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/bf16/cross_entropy/batched/dense_b4_c33_s31` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/bf16/cross_entropy/batched/dense_b4_c65_s2` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/bf16/cross_entropy/batched/dense_b4_c65_s5` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/bf16/cross_entropy/batched/dense_b4_c65_s17` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/bf16/cross_entropy/batched/dense_b4_c65_s31` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/bf16/log_softmax_nll/batched` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/bf16/log_softmax_nll/batched/dense_b2_c3_s2` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/bf16/log_softmax_nll/batched/dense_b2_c3_s5` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/bf16/log_softmax_nll/batched/dense_b2_c3_s17` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/bf16/log_softmax_nll/batched/dense_b2_c3_s31` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/bf16/log_softmax_nll/batched/dense_b2_c7_s2` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/bf16/log_softmax_nll/batched/dense_b2_c7_s5` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/bf16/log_softmax_nll/batched/dense_b2_c7_s17` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/bf16/log_softmax_nll/batched/dense_b2_c7_s31` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/bf16/log_softmax_nll/batched/dense_b2_c33_s2` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/bf16/log_softmax_nll/batched/dense_b2_c33_s5` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/bf16/log_softmax_nll/batched/dense_b2_c33_s17` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/bf16/log_softmax_nll/batched/dense_b2_c33_s31` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/bf16/log_softmax_nll/batched/dense_b2_c65_s2` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/bf16/log_softmax_nll/batched/dense_b2_c65_s5` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/bf16/log_softmax_nll/batched/dense_b2_c65_s17` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/bf16/log_softmax_nll/batched/dense_b2_c65_s31` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/bf16/log_softmax_nll/batched/dense_b4_c3_s2` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/bf16/log_softmax_nll/batched/dense_b4_c3_s5` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/bf16/log_softmax_nll/batched/dense_b4_c3_s17` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/bf16/log_softmax_nll/batched/dense_b4_c3_s31` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/bf16/log_softmax_nll/batched/dense_b4_c7_s2` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/bf16/log_softmax_nll/batched/dense_b4_c7_s5` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/bf16/log_softmax_nll/batched/dense_b4_c7_s17` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/bf16/log_softmax_nll/batched/dense_b4_c7_s31` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/bf16/log_softmax_nll/batched/dense_b4_c33_s2` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/bf16/log_softmax_nll/batched/dense_b4_c33_s5` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/bf16/log_softmax_nll/batched/dense_b4_c33_s17` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/bf16/log_softmax_nll/batched/dense_b4_c33_s31` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/bf16/log_softmax_nll/batched/dense_b4_c65_s2` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/bf16/log_softmax_nll/batched/dense_b4_c65_s5` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/bf16/log_softmax_nll/batched/dense_b4_c65_s17` | probe_failed | probe_not_correct | — | probe_not_correct | — |
| `metal/bf16/log_softmax_nll/batched/dense_b4_c65_s31` | probe_failed | probe_not_correct | — | probe_not_correct | — |

Each probe result is retained in [capability.json](capability.json).
Timing ratios are diagnostic; only reports with passing numerical parity are eligible for coverage links.
