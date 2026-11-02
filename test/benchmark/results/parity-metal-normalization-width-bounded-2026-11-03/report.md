# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `600b31b594a22dbbab0e2d78c6b08081302e7b6210cb58d3bd1deab7528a59de`  
Binary SHA-256: `ac5fc1bac6bdfe224813e26e098d9892f8c881006b6d95326903a1621ac42ad3`  
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
| `metal/f32/layer_norm/strided/inner_strided_r129_c257` | normalization | 129 × 257 | strided | correct | [passed](operator_peer/metal__f32__layer_norm__strided__inner_strided_r129_c257.json) | 0.9569 / 1.091 | [passed](program_peer/metal__f32__layer_norm__strided__inner_strided_r129_c257.json) | 0.8788 / 1.233 |
| `metal/f32/layer_norm/outer_axis/outer_strided_r129_c257` | normalization | 129 × 257 | strided | correct | [passed](operator_peer/metal__f32__layer_norm__outer_axis__outer_strided_r129_c257.json) | 1.677 / 1.356 | [passed](program_peer/metal__f32__layer_norm__outer_axis__outer_strided_r129_c257.json) | 1.613 / 1.018 |
| `metal/f32/rms_norm/outer_axis/strided_r129_c257` | normalization | 129 × 257 | strided | correct | [passed](operator_peer/metal__f32__rms_norm__outer_axis__strided_r129_c257.json) | 1.593 / 1.577 | [passed](program_peer/metal__f32__rms_norm__outer_axis__strided_r129_c257.json) | 1.53 / 1.646 |
| `metal/bf16/layer_norm/strided/inner_strided_r129_c257` | normalization | 129 × 257 | strided | correct | [passed](operator_peer/metal__bf16__layer_norm__strided__inner_strided_r129_c257.json) | 1.133 / 0.9258 | [passed](program_peer/metal__bf16__layer_norm__strided__inner_strided_r129_c257.json) | 1.024 / 0.8335 |
| `metal/bf16/layer_norm/outer_axis/outer_strided_r129_c257` | normalization | 129 × 257 | strided | correct | [passed](operator_peer/metal__bf16__layer_norm__outer_axis__outer_strided_r129_c257.json) | 1.474 / 1.007 | [passed](program_peer/metal__bf16__layer_norm__outer_axis__outer_strided_r129_c257.json) | 1.452 / 0.8853 |
| `metal/bf16/rms_norm/outer_axis/strided_r129_c257` | normalization | 129 × 257 | strided | correct | [passed](operator_peer/metal__bf16__rms_norm__outer_axis__strided_r129_c257.json) | 1.48 / 1.486 | [passed](program_peer/metal__bf16__rms_norm__outer_axis__strided_r129_c257.json) | 1.386 / 1.154 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
