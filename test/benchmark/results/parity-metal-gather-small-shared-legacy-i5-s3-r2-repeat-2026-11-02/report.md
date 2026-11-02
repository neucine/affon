# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `0c75341bf20e0856234ce94c709a5d2b1c84759bb4a558e9195d52143c1b0fbf`  
Binary SHA-256: `5be7468dbd2419582c4ea8118a0c9de0adb843bdf1d69efd17968bf64c02152c`  
Timing: 5 iterations × 3 samples × 2 rounds

## Checks

- Cases run: 8
- Numerical check passed: 8
- Operator check and timing passed: 8
- Program check and timing passed: 8
- Numerical failures: 0
- Operator failures: 0
- Program failures: 0

## Cases

| Case | Family | Shape | Layout | Numerical | Operator | Affon / PyTorch ms | Program | Affon / PyTorch ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `metal/f32/gather/strided/axis0_r65_c127` | indexing | 65 × 127 | strided | correct | [passed](operator_peer/metal__f32__gather__strided__axis0_r65_c127.json) | 0.5289 / 0.2929 | [passed](program_peer/metal__f32__gather__strided__axis0_r65_c127.json) | 0.4952 / 0.2664 |
| `metal/f32/gather/strided/axis1_r65_c127` | indexing | 65 × 127 | strided | correct | [passed](operator_peer/metal__f32__gather__strided__axis1_r65_c127.json) | 0.4525 / 0.2718 | [passed](program_peer/metal__f32__gather__strided__axis1_r65_c127.json) | 0.4906 / 0.2668 |
| `metal/f32/gather/large/offset_axis0_r65_c127` | indexing | 65 × 127 | offset | correct | [passed](operator_peer/metal__f32__gather__large__offset_axis0_r65_c127.json) | 0.4068 / 0.2686 | [passed](program_peer/metal__f32__gather__large__offset_axis0_r65_c127.json) | 0.4066 / 0.2495 |
| `metal/f32/gather/large/offset_axis1_r65_c127` | indexing | 65 × 127 | offset | correct | [passed](operator_peer/metal__f32__gather__large__offset_axis1_r65_c127.json) | 0.5553 / 0.2759 | [passed](program_peer/metal__f32__gather__large__offset_axis1_r65_c127.json) | 0.5186 / 0.2603 |
| `metal/bf16/gather/strided/axis0_r65_c127` | indexing | 65 × 127 | strided | correct | [passed](operator_peer/metal__bf16__gather__strided__axis0_r65_c127.json) | 0.5474 / 0.2912 | [passed](program_peer/metal__bf16__gather__strided__axis0_r65_c127.json) | 0.4705 / 0.2523 |
| `metal/bf16/gather/strided/axis1_r65_c127` | indexing | 65 × 127 | strided | correct | [passed](operator_peer/metal__bf16__gather__strided__axis1_r65_c127.json) | 0.5591 / 0.2638 | [passed](program_peer/metal__bf16__gather__strided__axis1_r65_c127.json) | 0.4848 / 0.2735 |
| `metal/bf16/gather/large/offset_axis0_r65_c127` | indexing | 65 × 127 | offset | correct | [passed](operator_peer/metal__bf16__gather__large__offset_axis0_r65_c127.json) | 0.5504 / 0.2627 | [passed](program_peer/metal__bf16__gather__large__offset_axis0_r65_c127.json) | 0.4525 / 0.2705 |
| `metal/bf16/gather/large/offset_axis1_r65_c127` | indexing | 65 × 127 | offset | correct | [passed](operator_peer/metal__bf16__gather__large__offset_axis1_r65_c127.json) | 0.4316 / 0.2575 | [passed](program_peer/metal__bf16__gather__large__offset_axis1_r65_c127.json) | 0.5658 / 0.2665 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
