# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `ed7cd8e442707dc50f82e08aab1335ce7d060c01c9e12d63f884462838bba926`  
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
| `metal/f32/index_select/batched` | correct | [passed](operator_peer/metal__f32__index_select__batched.json) | 0.253 / 0.2457 | [passed](program_peer/metal__f32__index_select__batched.json) | 0.2596 / 0.2187 |
| `metal/bf16/index_select/batched` | correct | [passed](operator_peer/metal__bf16__index_select__batched.json) | 0.251 / 0.2484 | [passed](program_peer/metal__bf16__index_select__batched.json) | 0.2666 / 0.2124 |

Each probe result is retained in [capability.json](capability.json).
Timing ratios are diagnostic; only reports with passing numerical parity are eligible for coverage links.
