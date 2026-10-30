# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `cf187603e5077897206fdb7c22bceec5651692a6cced37a81c381d17686635fc`  
Binary SHA-256: `35a6066d3a3f4a07fec5cc7dbeb8caaac66a660948cc409b260487dd9a344024`  
Timing: 5 iterations × 5 samples × 3 rounds

## Coverage

- Candidate cells: 2
- Numerical capability passed: 2
- Operator parity and timing passed: 2
- Program parity and timing passed: 2
- Probe failures or incorrect: 0
- Operator incorrect or error: 0
- Program incorrect or error: 0

## Cells

| Cell | Probe | Operator parity | Affon / peer ms | Program parity | Affon / peer ms |
| --- | --- | --- | --- | --- | --- |
| `metal/f32/topk/batched` | correct | [passed](operator_peer/metal__f32__topk__batched.json) | 0.3574 / 0.512 | [passed](program_peer/metal__f32__topk__batched.json) | 0.533 / 0.5044 |
| `metal/bf16/topk/batched` | correct | [passed](operator_peer/metal__bf16__topk__batched.json) | 0.3826 / 0.5057 | [passed](program_peer/metal__bf16__topk__batched.json) | 0.3432 / 0.5096 |

Each probe result is retained in [capability.json](capability.json).
Timing ratios are diagnostic; only reports with passing numerical parity are eligible for coverage links.
