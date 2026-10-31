# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `62eae18127ec27edc5c950099c91d36e5dd8717482a8d249326cb5acc94b0a85`  
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
| `metal/f32/dot/ragged` | correct | [passed](operator_peer/metal__f32__dot__ragged.json) | 0.3532 / 0.2331 | [passed](program_peer/metal__f32__dot__ragged.json) | 0.3692 / 0.2698 |
| `metal/bf16/dot/ragged` | correct | [passed](operator_peer/metal__bf16__dot__ragged.json) | 0.2212 / 0.2089 | [passed](program_peer/metal__bf16__dot__ragged.json) | 0.2582 / 0.2558 |

Each probe result is retained in [capability.json](capability.json).
Timing ratios are diagnostic; only reports with passing numerical parity are eligible for coverage links.
