# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `600b31b594a22dbbab0e2d78c6b08081302e7b6210cb58d3bd1deab7528a59de`  
Binary SHA-256: `dd3c679a2f419fcc10d363188483951fc5e133a9d832863091c2f94da9919fe6`  
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
| `metal/f32/layer_norm/strided/inner_strided_r129_c257` | normalization | 129 × 257 | strided | correct | [passed](operator_peer/metal__f32__layer_norm__strided__inner_strided_r129_c257.json) | 0.7535 / 0.7259 | [passed](program_peer/metal__f32__layer_norm__strided__inner_strided_r129_c257.json) | 0.8641 / 0.7896 |
| `metal/f32/layer_norm/outer_axis/outer_strided_r129_c257` | normalization | 129 × 257 | strided | correct | [passed](operator_peer/metal__f32__layer_norm__outer_axis__outer_strided_r129_c257.json) | 1.089 / 0.7971 | [passed](program_peer/metal__f32__layer_norm__outer_axis__outer_strided_r129_c257.json) | 1.156 / 0.8057 |
| `metal/f32/rms_norm/outer_axis/strided_r129_c257` | normalization | 129 × 257 | strided | correct | [passed](operator_peer/metal__f32__rms_norm__outer_axis__strided_r129_c257.json) | 1.357 / 0.9013 | [passed](program_peer/metal__f32__rms_norm__outer_axis__strided_r129_c257.json) | 1.56 / 0.9441 |
| `metal/bf16/layer_norm/strided/inner_strided_r129_c257` | normalization | 129 × 257 | strided | correct | [passed](operator_peer/metal__bf16__layer_norm__strided__inner_strided_r129_c257.json) | 0.9325 / 0.81 | [passed](program_peer/metal__bf16__layer_norm__strided__inner_strided_r129_c257.json) | 0.7866 / 0.7663 |
| `metal/bf16/layer_norm/outer_axis/outer_strided_r129_c257` | normalization | 129 × 257 | strided | correct | [passed](operator_peer/metal__bf16__layer_norm__outer_axis__outer_strided_r129_c257.json) | 0.9752 / 0.7949 | [passed](program_peer/metal__bf16__layer_norm__outer_axis__outer_strided_r129_c257.json) | 1.06 / 0.7837 |
| `metal/bf16/rms_norm/outer_axis/strided_r129_c257` | normalization | 129 × 257 | strided | correct | [passed](operator_peer/metal__bf16__rms_norm__outer_axis__strided_r129_c257.json) | 0.9823 / 0.9596 | [passed](program_peer/metal__bf16__rms_norm__outer_axis__strided_r129_c257.json) | 1.052 / 0.9056 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
