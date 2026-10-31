# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `3b830de2a8213daeeb32c5ee6cf1e60402462f269e8ad815bc197e675912996a`  
Binary SHA-256: `af1b758d59cd4f42a812cf3dcaae67d3b0f01abca19878e39bc7300e1ae67e35`  
Timing: 5 iterations × 5 samples × 3 rounds

## Coverage

- Candidate cells: 4
- Numerical capability passed: 4
- Operator parity and timing passed: 4
- Program parity and timing passed: 4
- Probe failures or incorrect: 0
- Operator incorrect or error: 0
- Program incorrect or error: 0

## Cells

| Cell | Probe | Operator parity | Affon / peer ms | Program parity | Affon / peer ms |
| --- | --- | --- | --- | --- | --- |
| `metal/bf16/topk/small` | correct | [passed](operator_peer/metal__bf16__topk__small.json) | 0.2674 / 0.4716 | [passed](program_peer/metal__bf16__topk__small.json) | 0.2946 / 0.4585 |
| `metal/bf16/topk/large` | correct | [passed](operator_peer/metal__bf16__topk__large.json) | 0.2374 / 0.4489 | [passed](program_peer/metal__bf16__topk__large.json) | 0.2654 / 0.4613 |
| `metal/f32/topk/small` | correct | [passed](operator_peer/metal__f32__topk__small.json) | 0.246 / 0.4492 | [passed](program_peer/metal__f32__topk__small.json) | 0.2868 / 0.4651 |
| `metal/f32/topk/large` | correct | [passed](operator_peer/metal__f32__topk__large.json) | 0.2594 / 0.4706 | [passed](program_peer/metal__f32__topk__large.json) | 0.2694 / 0.4696 |

Each probe result is retained in [capability.json](capability.json).
Timing ratios are diagnostic; only reports with passing numerical parity are eligible for coverage links.
