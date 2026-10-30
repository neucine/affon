# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `91a45f676bf08f4f9c382df9e6cae85c494f8872334586899c4705bb6ae881ca`  
Binary SHA-256: `ad90e56f47b7fb149ae7fd8cb3c96e37edb551854a4a2404f625a99a28304ff3`  
Timing: 5 iterations × 3 samples × 2 rounds

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
| `metal/bf16/variance_axis/outer_axis/strided_r7_c127` | correct | [passed](operator_peer/metal__bf16__variance_axis__outer_axis__strided_r7_c127.json) | 0.3212 / 0.2939 | [passed](program_peer/metal__bf16__variance_axis__outer_axis__strided_r7_c127.json) | 0.3127 / 0.2784 |
| `metal/bf16/variance_axis/outer_axis/strided_r7_c257` | correct | [passed](operator_peer/metal__bf16__variance_axis__outer_axis__strided_r7_c257.json) | 0.3681 / 0.3044 | [passed](program_peer/metal__bf16__variance_axis__outer_axis__strided_r7_c257.json) | 0.3092 / 0.2322 |
| `metal/bf16/variance_axis/outer_axis/offset_r2_c257` | correct | [passed](operator_peer/metal__bf16__variance_axis__outer_axis__offset_r2_c257.json) | 0.2503 / 0.252 | [passed](program_peer/metal__bf16__variance_axis__outer_axis__offset_r2_c257.json) | 0.2865 / 0.2173 |
| `metal/bf16/variance_axis/outer_axis/offset_r7_c257` | correct | [passed](operator_peer/metal__bf16__variance_axis__outer_axis__offset_r7_c257.json) | 0.2641 / 0.2806 | [passed](program_peer/metal__bf16__variance_axis__outer_axis__offset_r7_c257.json) | 0.3165 / 0.2846 |

Each probe result is retained in [capability.json](capability.json).
Timing ratios are diagnostic; only reports with passing numerical parity are eligible for coverage links.
