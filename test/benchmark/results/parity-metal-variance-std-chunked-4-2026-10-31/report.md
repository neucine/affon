# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `fa4d459bf961b0b7587fe3504ae0008c0f68e5b4693bee3f3b5768e96f1312bd`  
Binary SHA-256: `ad90e56f47b7fb149ae7fd8cb3c96e37edb551854a4a2404f625a99a28304ff3`  
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
| `metal/f32/variance_all/long_axis` | correct | [passed](operator_peer/metal__f32__variance_all__long_axis.json) | 5.134 / 0.2206 | [passed](program_peer/metal__f32__variance_all__long_axis.json) | 5.149 / 0.2149 |
| `metal/f32/variance_all/strided/r64_c127` | correct | [passed](operator_peer/metal__f32__variance_all__strided__r64_c127.json) | 13.46 / 0.2469 | [passed](program_peer/metal__f32__variance_all__strided__r64_c127.json) | 13.54 / 0.2625 |
| `metal/f32/std_all/long_axis` | correct | [passed](operator_peer/metal__f32__std_all__long_axis.json) | 5.121 / 0.2143 | [passed](program_peer/metal__f32__std_all__long_axis.json) | 5.366 / 0.2239 |
| `metal/f32/std_all/strided/r64_c127` | correct | [passed](operator_peer/metal__f32__std_all__strided__r64_c127.json) | 13.56 / 0.253 | [passed](program_peer/metal__f32__std_all__strided__r64_c127.json) | 13.38 / 0.2497 |

Each probe result is retained in [capability.json](capability.json).
Timing ratios are diagnostic; only reports with passing numerical parity are eligible for coverage links.
