# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `fab97ac08e81a6c86db874c0d3d5b808e7a4b248f05454462801079114e801dd`  
Binary SHA-256: `35a6066d3a3f4a07fec5cc7dbeb8caaac66a660948cc409b260487dd9a344024`  
Timing: 5 iterations × 5 samples × 3 rounds

## Coverage

- Candidate cells: 1
- Numerical capability passed: 1
- Operator parity and timing passed: 1
- Program parity and timing passed: 1
- Probe failures or incorrect: 0
- Operator incorrect or error: 0
- Program incorrect or error: 0

## Cells

| Cell | Probe | Operator parity | Affon / peer ms | Program parity | Affon / peer ms |
| --- | --- | --- | --- | --- | --- |
| `metal/f32/reduce_to_shape/scalar` | correct | [passed](operator_peer/metal__f32__reduce_to_shape__scalar.json) | 0.287 / 0.2145 | [passed](program_peer/metal__f32__reduce_to_shape__scalar.json) | 0.3044 / 0.2304 |

Each probe result is retained in [capability.json](capability.json).
Timing ratios are diagnostic; only reports with passing numerical parity are eligible for coverage links.
