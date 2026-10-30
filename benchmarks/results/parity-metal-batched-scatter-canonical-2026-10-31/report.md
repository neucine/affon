# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `a0b9e5fe37a0c15acccf16efff8a4e91717761fe4606e95e5313fddb3052d121`  
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
| `metal/f32/scatter_add/batched` | correct | [passed](operator_peer/metal__f32__scatter_add__batched.json) | 0.2764 / 0.2827 | [passed](program_peer/metal__f32__scatter_add__batched.json) | 0.288 / 0.2226 |
| `metal/bf16/scatter_add/batched` | correct | [passed](operator_peer/metal__bf16__scatter_add__batched.json) | 0.378 / 0.237 | [passed](program_peer/metal__bf16__scatter_add__batched.json) | 0.4152 / 0.2368 |

Each probe result is retained in [capability.json](capability.json).
Timing ratios are diagnostic; only reports with passing numerical parity are eligible for coverage links.
