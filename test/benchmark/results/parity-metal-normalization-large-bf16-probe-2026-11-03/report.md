# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `2e34902275c8c7bfdf854ed23fdaa1ab873af3b871d4713427b111d09f732b64`  
Binary SHA-256: `61a93a44517ea985151e700ce24b6e1ede3225e7215cb29def73418f82190de7`  
Timing: 10 iterations × 5 samples × 3 rounds

## Checks

- Cases run: 2
- Numerical check passed: 2
- Operator check and timing passed: 2
- Program check and timing passed: 2
- Numerical failures: 0
- Operator failures: 0
- Program failures: 0

## Cases

| Case | Family | Shape | Layout | Numerical | Operator | Affon / PyTorch ms | Program | Affon / PyTorch ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `metal/bf16/layer_norm/outer_axis/outer_strided_r129_c257` | normalization | 129 × 257 | strided | correct | [passed](operator_peer/metal__bf16__layer_norm__outer_axis__outer_strided_r129_c257.json) | 0.7625 / 0.5904 | [passed](program_peer/metal__bf16__layer_norm__outer_axis__outer_strided_r129_c257.json) | 0.8373 / 0.6016 |
| `metal/bf16/rms_norm/outer_axis/strided_r129_c257` | normalization | 129 × 257 | strided | correct | [passed](operator_peer/metal__bf16__rms_norm__outer_axis__strided_r129_c257.json) | 0.7675 / 0.6926 | [passed](program_peer/metal__bf16__rms_norm__outer_axis__strided_r129_c257.json) | 0.8418 / 0.6979 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
