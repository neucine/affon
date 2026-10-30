# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `2fc63b31561dd51740bdf65bb7bb519116a1f9265a4806e390acb24d9986a920`  
Binary SHA-256: `16d14cf1b9cd8279ae8a5766519b8506a63a19322c9728d6ba0b391f478b6bfe`  
Timing: 5 iterations × 5 samples × 3 rounds

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
| `metal/bf16/layer_norm/outer_axis/outer_strided_r129_c257` | normalization | 129 × 257 | strided | correct | [passed](operator_peer/metal__bf16__layer_norm__outer_axis__outer_strided_r129_c257.json) | 0.8676 / 0.6087 | [passed](program_peer/metal__bf16__layer_norm__outer_axis__outer_strided_r129_c257.json) | 0.9108 / 0.6248 |
| `metal/bf16/layer_norm/outer_axis/outer_offset_r129_c257` | normalization | 129 × 257 | offset | correct | [passed](operator_peer/metal__bf16__layer_norm__outer_axis__outer_offset_r129_c257.json) | 0.8866 / 0.6168 | [passed](program_peer/metal__bf16__layer_norm__outer_axis__outer_offset_r129_c257.json) | 0.9158 / 0.629 |
| `metal/bf16/rms_norm/outer_axis/strided_r129_c257` | normalization | 129 × 257 | strided | correct | [passed](operator_peer/metal__bf16__rms_norm__outer_axis__strided_r129_c257.json) | 0.8536 / 0.7528 | [passed](program_peer/metal__bf16__rms_norm__outer_axis__strided_r129_c257.json) | 0.913 / 0.7651 |
| `metal/bf16/rms_norm/outer_axis/offset_r129_c257` | normalization | 129 × 257 | offset | correct | [passed](operator_peer/metal__bf16__rms_norm__outer_axis__offset_r129_c257.json) | 0.8676 / 0.745 | [passed](program_peer/metal__bf16__rms_norm__outer_axis__offset_r129_c257.json) | 0.9126 / 0.7522 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
