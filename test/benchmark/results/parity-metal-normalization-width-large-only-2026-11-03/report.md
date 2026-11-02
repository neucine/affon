# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `600b31b594a22dbbab0e2d78c6b08081302e7b6210cb58d3bd1deab7528a59de`  
Binary SHA-256: `53ce95289e56527778b27224cf8b6b576b79b2b8ecd054014a5fb4b246cfcb9a`  
Timing: 10 iterations × 5 samples × 3 rounds

## Checks

- Cases run: 6
- Numerical check passed: 6
- Operator check and timing passed: 6
- Program check and timing passed: 6
- Numerical failures: 0
- Operator failures: 0
- Program failures: 0

## Cases

| Case | Family | Shape | Layout | Numerical | Operator | Affon / PyTorch ms | Program | Affon / PyTorch ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `metal/f32/layer_norm/strided/inner_strided_r129_c257` | normalization | 129 × 257 | strided | correct | [passed](operator_peer/metal__f32__layer_norm__strided__inner_strided_r129_c257.json) | 0.5881 / 0.578 | [passed](program_peer/metal__f32__layer_norm__strided__inner_strided_r129_c257.json) | 0.5945 / 0.5755 |
| `metal/f32/layer_norm/outer_axis/outer_strided_r129_c257` | normalization | 129 × 257 | strided | correct | [passed](operator_peer/metal__f32__layer_norm__outer_axis__outer_strided_r129_c257.json) | 2.181 / 0.6257 | [passed](program_peer/metal__f32__layer_norm__outer_axis__outer_strided_r129_c257.json) | 2.169 / 0.6131 |
| `metal/f32/rms_norm/outer_axis/strided_r129_c257` | normalization | 129 × 257 | strided | correct | [passed](operator_peer/metal__f32__rms_norm__outer_axis__strided_r129_c257.json) | 2.165 / 0.6686 | [passed](program_peer/metal__f32__rms_norm__outer_axis__strided_r129_c257.json) | 2.172 / 0.7167 |
| `metal/bf16/layer_norm/strided/inner_strided_r129_c257` | normalization | 129 × 257 | strided | correct | [passed](operator_peer/metal__bf16__layer_norm__strided__inner_strided_r129_c257.json) | 0.5631 / 0.602 | [passed](program_peer/metal__bf16__layer_norm__strided__inner_strided_r129_c257.json) | 0.6394 / 0.5707 |
| `metal/bf16/layer_norm/outer_axis/outer_strided_r129_c257` | normalization | 129 × 257 | strided | correct | [passed](operator_peer/metal__bf16__layer_norm__outer_axis__outer_strided_r129_c257.json) | 1.621 / 0.5917 | [passed](program_peer/metal__bf16__layer_norm__outer_axis__outer_strided_r129_c257.json) | 1.718 / 0.6004 |
| `metal/bf16/rms_norm/outer_axis/strided_r129_c257` | normalization | 129 × 257 | strided | correct | [passed](operator_peer/metal__bf16__rms_norm__outer_axis__strided_r129_c257.json) | 1.648 / 0.7046 | [passed](program_peer/metal__bf16__rms_norm__outer_axis__strided_r129_c257.json) | 1.697 / 0.7023 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
