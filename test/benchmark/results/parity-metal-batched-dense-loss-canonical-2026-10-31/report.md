# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `9bd86ba88b34e0bbe512b9f5c9bc9fbfa59a081de2ee2efc002e721e3e235ce6`  
Binary SHA-256: `9a75f24e5bd8f46bd7df96404554d41c4707684e608e81232b99890c4a6e9306`  
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
| `metal/f32/cross_entropy/batched` | correct | [passed](operator_peer/metal__f32__cross_entropy__batched.json) | 0.3352 / 0.2872 | [passed](program_peer/metal__f32__cross_entropy__batched.json) | 0.2992 / 0.2777 |
| `metal/f32/log_softmax_nll/batched` | correct | [passed](operator_peer/metal__f32__log_softmax_nll__batched.json) | 0.3138 / 0.2716 | [passed](program_peer/metal__f32__log_softmax_nll__batched.json) | 0.3126 / 0.2783 |
| `metal/bf16/cross_entropy/batched` | correct | [passed](operator_peer/metal__bf16__cross_entropy__batched.json) | 0.304 / 0.3263 | [passed](program_peer/metal__bf16__cross_entropy__batched.json) | 0.2484 / 0.3213 |
| `metal/bf16/log_softmax_nll/batched` | correct | [passed](operator_peer/metal__bf16__log_softmax_nll__batched.json) | 0.3124 / 0.3189 | [passed](program_peer/metal__bf16__log_softmax_nll__batched.json) | 0.3154 / 0.3259 |

Each probe result is retained in [capability.json](capability.json).
Timing ratios are diagnostic; only reports with passing numerical parity are eligible for coverage links.
