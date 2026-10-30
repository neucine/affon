# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `fa4d459bf961b0b7587fe3504ae0008c0f68e5b4693bee3f3b5768e96f1312bd`  
Binary SHA-256: `88145a079e7c3c1c0b44ee4851c10f40b7a808257d6071c545b19e203914087f`  
Timing: 5 iterations × 5 samples × 3 rounds

## Coverage

- Candidate cells: 4
- Numerical capability passed: 0
- Operator parity and timing passed: 0
- Program parity and timing passed: 0
- Probe failures or incorrect: 4
- Operator incorrect or error: 0
- Program incorrect or error: 0

## Cells

| Cell | Probe | Operator parity | Affon / peer ms | Program parity | Affon / peer ms |
| --- | --- | --- | --- | --- | --- |
| `metal/f32/variance_all/long_axis` | incorrect | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/variance_all/strided/r64_c127` | incorrect | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/std_all/long_axis` | incorrect | probe_not_correct | — | probe_not_correct | — |
| `metal/f32/std_all/strided/r64_c127` | incorrect | probe_not_correct | — | probe_not_correct | — |

Each probe result is retained in [capability.json](capability.json).
Timing ratios are diagnostic; only reports with passing numerical parity are eligible for coverage links.
