# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `126307bb3c643032fedfd78b0056679e79c92a6639003b60b95c749a7c3e2d28`  
Binary SHA-256: `41294c400027391903b8974101572e1b8d1ba4b6502136ab2f497b74524c9e1a`  
Timing: 10 iterations × 5 samples × 3 rounds

## Checks

- Cases run: 4
- Numerical check passed: 4
- Operator check and timing passed: 4
- Program check and timing passed: 4
- Numerical failures: 0
- Operator failures: 0
- Program failures: 0

## Cases

| Case | Family | Shape | Layout | Numerical | Operator | Affon / PyTorch ms | Program | Affon / PyTorch ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `metal/bf16/mean_axis/outer_axis/strided_r33_c5` | reduction | 33 × 5 | strided | correct | [passed](operator_peer/metal__bf16__mean_axis__outer_axis__strided_r33_c5.json) | 0.2367 / 0.2485 | [passed](program_peer/metal__bf16__mean_axis__outer_axis__strided_r33_c5.json) | 0.2467 / 0.2473 |
| `metal/bf16/mean_axis/strided/inner_strided_r16_c129` | reduction | 16 × 129 | strided | correct | [passed](operator_peer/metal__bf16__mean_axis__strided__inner_strided_r16_c129.json) | 0.2579 / 0.2333 | [passed](program_peer/metal__bf16__mean_axis__strided__inner_strided_r16_c129.json) | 0.2549 / 0.251 |
| `metal/bf16/mean_axis/strided/inner_strided_r129_c63` | reduction | 129 × 63 | strided | correct | [passed](operator_peer/metal__bf16__mean_axis__strided__inner_strided_r129_c63.json) | 0.2784 / 0.2344 | [passed](program_peer/metal__bf16__mean_axis__strided__inner_strided_r129_c63.json) | 0.2857 / 0.2688 |
| `metal/bf16/sum_axis/outer_axis/offset_r129_c257` | reduction | 129 × 257 | offset | correct | [passed](operator_peer/metal__bf16__sum_axis__outer_axis__offset_r129_c257.json) | 0.295 / 0.2674 | [passed](program_peer/metal__bf16__sum_axis__outer_axis__offset_r129_c257.json) | 0.3022 / 0.2806 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
