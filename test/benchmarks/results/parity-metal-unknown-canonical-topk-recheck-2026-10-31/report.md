# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `3b830de2a8213daeeb32c5ee6cf1e60402462f269e8ad815bc197e675912996a`  
Binary SHA-256: `af1b758d59cd4f42a812cf3dcaae67d3b0f01abca19878e39bc7300e1ae67e35`  
Timing: 5 iterations × 5 samples × 3 rounds

## Coverage

- Candidate cells: 4
- Numerical capability passed: 4
- Operator parity and timing passed: 0
- Program parity and timing passed: 0
- Probe failures or incorrect: 0
- Operator incorrect or error: 4
- Program incorrect or error: 4

## Cells

| Cell | Probe | Operator parity | Affon / peer ms | Program parity | Affon / peer ms |
| --- | --- | --- | --- | --- | --- |
| `metal/bf16/topk/small` | correct | [error](operator_peer_failures/metal__bf16__topk__small.json) | — | [error](program_peer_failures/metal__bf16__topk__small.json) | — |
| `metal/bf16/topk/large` | correct | [error](operator_peer_failures/metal__bf16__topk__large.json) | — | [error](program_peer_failures/metal__bf16__topk__large.json) | — |
| `metal/f32/topk/small` | correct | [error](operator_peer_failures/metal__f32__topk__small.json) | — | [error](program_peer_failures/metal__f32__topk__small.json) | — |
| `metal/f32/topk/large` | correct | [error](operator_peer_failures/metal__f32__topk__large.json) | — | [error](program_peer_failures/metal__f32__topk__large.json) | — |

Each probe result is retained in [capability.json](capability.json).
Timing ratios are diagnostic; only reports with passing numerical parity are eligible for coverage links.
