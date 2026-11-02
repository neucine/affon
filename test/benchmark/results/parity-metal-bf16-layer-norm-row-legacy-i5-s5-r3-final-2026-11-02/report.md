# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `35788476901ffea5bd0f65e91f32685b79f46ff9092e66a9869494c97450266b`  
Binary SHA-256: `07d2686a6fde2c671da3d04668ac145d8ca91e78bd6eb55675701480f9735945`  
Timing: 5 iterations × 5 samples × 3 rounds

## Checks

- Cases run: 18
- Numerical check passed: 18
- Operator check and timing passed: 18
- Program check and timing passed: 18
- Numerical failures: 0
- Operator failures: 0
- Program failures: 0

## Cases

| Case | Family | Shape | Layout | Numerical | Operator | Affon / PyTorch ms | Program | Affon / PyTorch ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `metal/bf16/layer_norm/outer_axis/outer_strided_r7_c257` | normalization | 7 × 257 | strided | correct | [passed](operator_peer/metal__bf16__layer_norm__outer_axis__outer_strided_r7_c257.json) | 0.3486 / 0.2202 | [passed](program_peer/metal__bf16__layer_norm__outer_axis__outer_strided_r7_c257.json) | 0.3944 / 0.2211 |
| `metal/bf16/layer_norm/outer_axis/outer_strided_r64_c17` | normalization | 64 × 17 | strided | correct | [passed](operator_peer/metal__bf16__layer_norm__outer_axis__outer_strided_r64_c17.json) | 0.3358 / 0.2135 | [passed](program_peer/metal__bf16__layer_norm__outer_axis__outer_strided_r64_c17.json) | 0.3252 / 0.2121 |
| `metal/bf16/layer_norm/outer_axis/outer_strided_r64_c63` | normalization | 64 × 63 | strided | correct | [passed](operator_peer/metal__bf16__layer_norm__outer_axis__outer_strided_r64_c63.json) | 0.3412 / 0.2413 | [passed](program_peer/metal__bf16__layer_norm__outer_axis__outer_strided_r64_c63.json) | 0.3318 / 0.2456 |
| `metal/bf16/layer_norm/outer_axis/outer_strided_r64_c127` | normalization | 64 × 127 | strided | correct | [passed](operator_peer/metal__bf16__layer_norm__outer_axis__outer_strided_r64_c127.json) | 0.3882 / 0.278 | [passed](program_peer/metal__bf16__layer_norm__outer_axis__outer_strided_r64_c127.json) | 0.3924 / 0.2768 |
| `metal/bf16/layer_norm/outer_axis/outer_strided_r64_c257` | normalization | 64 × 257 | strided | correct | [passed](operator_peer/metal__bf16__layer_norm__outer_axis__outer_strided_r64_c257.json) | 0.5076 / 0.4353 | [passed](program_peer/metal__bf16__layer_norm__outer_axis__outer_strided_r64_c257.json) | 0.5098 / 0.4277 |
| `metal/bf16/layer_norm/outer_axis/outer_strided_r129_c17` | normalization | 129 × 17 | strided | correct | [passed](operator_peer/metal__bf16__layer_norm__outer_axis__outer_strided_r129_c17.json) | 0.4616 / 0.2251 | [passed](program_peer/metal__bf16__layer_norm__outer_axis__outer_strided_r129_c17.json) | 0.4994 / 0.229 |
| `metal/bf16/layer_norm/outer_axis/outer_strided_r129_c63` | normalization | 129 × 63 | strided | correct | [passed](operator_peer/metal__bf16__layer_norm__outer_axis__outer_strided_r129_c63.json) | 0.4586 / 0.2828 | [passed](program_peer/metal__bf16__layer_norm__outer_axis__outer_strided_r129_c63.json) | 0.4264 / 0.2847 |
| `metal/bf16/layer_norm/outer_axis/outer_strided_r129_c127` | normalization | 129 × 127 | strided | correct | [passed](operator_peer/metal__bf16__layer_norm__outer_axis__outer_strided_r129_c127.json) | 0.5702 / 0.4019 | [passed](program_peer/metal__bf16__layer_norm__outer_axis__outer_strided_r129_c127.json) | 0.5914 / 0.4029 |
| `metal/bf16/layer_norm/outer_axis/outer_strided_r129_c257` | normalization | 129 × 257 | strided | correct | [passed](operator_peer/metal__bf16__layer_norm__outer_axis__outer_strided_r129_c257.json) | 0.879 / 0.632 | [passed](program_peer/metal__bf16__layer_norm__outer_axis__outer_strided_r129_c257.json) | 0.9048 / 0.6505 |
| `metal/bf16/layer_norm/outer_axis/outer_offset_r7_c257` | normalization | 7 × 257 | offset | correct | [passed](operator_peer/metal__bf16__layer_norm__outer_axis__outer_offset_r7_c257.json) | 0.2742 / 0.2219 | [passed](program_peer/metal__bf16__layer_norm__outer_axis__outer_offset_r7_c257.json) | 0.3796 / 0.2236 |
| `metal/bf16/layer_norm/outer_axis/outer_offset_r64_c17` | normalization | 64 × 17 | offset | correct | [passed](operator_peer/metal__bf16__layer_norm__outer_axis__outer_offset_r64_c17.json) | 0.2836 / 0.2107 | [passed](program_peer/metal__bf16__layer_norm__outer_axis__outer_offset_r64_c17.json) | 0.3798 / 0.2152 |
| `metal/bf16/layer_norm/outer_axis/outer_offset_r64_c63` | normalization | 64 × 63 | offset | correct | [passed](operator_peer/metal__bf16__layer_norm__outer_axis__outer_offset_r64_c63.json) | 0.546 / 0.2395 | [passed](program_peer/metal__bf16__layer_norm__outer_axis__outer_offset_r64_c63.json) | 0.3254 / 0.2453 |
| `metal/bf16/layer_norm/outer_axis/outer_offset_r64_c127` | normalization | 64 × 127 | offset | correct | [passed](operator_peer/metal__bf16__layer_norm__outer_axis__outer_offset_r64_c127.json) | 0.4048 / 0.2771 | [passed](program_peer/metal__bf16__layer_norm__outer_axis__outer_offset_r64_c127.json) | 0.3948 / 0.2817 |
| `metal/bf16/layer_norm/outer_axis/outer_offset_r64_c257` | normalization | 64 × 257 | offset | correct | [passed](operator_peer/metal__bf16__layer_norm__outer_axis__outer_offset_r64_c257.json) | 0.5102 / 0.4391 | [passed](program_peer/metal__bf16__layer_norm__outer_axis__outer_offset_r64_c257.json) | 0.5194 / 0.3969 |
| `metal/bf16/layer_norm/outer_axis/outer_offset_r129_c17` | normalization | 129 × 17 | offset | correct | [passed](operator_peer/metal__bf16__layer_norm__outer_axis__outer_offset_r129_c17.json) | 0.4112 / 0.2258 | [passed](program_peer/metal__bf16__layer_norm__outer_axis__outer_offset_r129_c17.json) | 0.3982 / 0.2253 |
| `metal/bf16/layer_norm/outer_axis/outer_offset_r129_c63` | normalization | 129 × 63 | offset | correct | [passed](operator_peer/metal__bf16__layer_norm__outer_axis__outer_offset_r129_c63.json) | 0.4146 / 0.2796 | [passed](program_peer/metal__bf16__layer_norm__outer_axis__outer_offset_r129_c63.json) | 0.4202 / 0.284 |
| `metal/bf16/layer_norm/outer_axis/outer_offset_r129_c127` | normalization | 129 × 127 | offset | correct | [passed](operator_peer/metal__bf16__layer_norm__outer_axis__outer_offset_r129_c127.json) | 0.5634 / 0.4374 | [passed](program_peer/metal__bf16__layer_norm__outer_axis__outer_offset_r129_c127.json) | 0.5746 / 0.384 |
| `metal/bf16/layer_norm/outer_axis/outer_offset_r129_c257` | normalization | 129 × 257 | offset | correct | [passed](operator_peer/metal__bf16__layer_norm__outer_axis__outer_offset_r129_c257.json) | 0.8786 / 0.639 | [passed](program_peer/metal__bf16__layer_norm__outer_axis__outer_offset_r129_c257.json) | 0.8998 / 0.6413 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
