# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `63c9906193991766d04f38dbea0c09d3962ea8f0fd9a61a38ad1361803721dd0`  
Binary SHA-256: `35a6066d3a3f4a07fec5cc7dbeb8caaac66a660948cc409b260487dd9a344024`  
Timing: 5 iterations × 5 samples × 3 rounds

## Coverage

- Candidate cells: 2
- Numerical capability passed: 1
- Operator parity and timing passed: 1
- Program parity and timing passed: 1
- Probe failures or incorrect: 1
- Operator incorrect or error: 0
- Program incorrect or error: 0

## Cells

| Cell | Probe | Operator parity | Affon / peer ms | Program parity | Affon / peer ms |
| --- | --- | --- | --- | --- | --- |
| `metal/f32/scatter_add/invalid_index` | correct | [passed](operator_peer/metal__f32__scatter_add__invalid_index.json) | 0.222 / 0.224 | [passed](program_peer/metal__f32__scatter_add__invalid_index.json) | 0.2592 / 0.2345 |
| `metal/bf16/scatter_add/invalid_index` | probe_failed | probe_not_correct | — | probe_not_correct | — |

Each probe result is retained in [capability.json](capability.json).
Timing ratios are diagnostic; only reports with passing numerical parity are eligible for coverage links.
