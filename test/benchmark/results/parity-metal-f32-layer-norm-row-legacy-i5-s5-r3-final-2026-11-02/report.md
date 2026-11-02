# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `e9a6ef581f7f65267b8ea4d1f6983cbf10621aa0c299a874980a4f403018887b`  
Binary SHA-256: `1263f72409ea8b8e7c7c5ea44ec4bc78639d602290b1885391b28c52332c724e`  
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
| `metal/f32/layer_norm/outer_axis/outer_strided_r7_c257` | normalization | 7 × 257 | strided | correct | [passed](operator_peer/metal__f32__layer_norm__outer_axis__outer_strided_r7_c257.json) | 0.4142 / 0.2203 | [passed](program_peer/metal__f32__layer_norm__outer_axis__outer_strided_r7_c257.json) | 0.4182 / 0.2117 |
| `metal/f32/layer_norm/outer_axis/outer_strided_r64_c17` | normalization | 64 × 17 | strided | correct | [passed](operator_peer/metal__f32__layer_norm__outer_axis__outer_strided_r64_c17.json) | 0.27 / 0.2139 | [passed](program_peer/metal__f32__layer_norm__outer_axis__outer_strided_r64_c17.json) | 0.2774 / 0.2121 |
| `metal/f32/layer_norm/outer_axis/outer_strided_r64_c63` | normalization | 64 × 63 | strided | correct | [passed](operator_peer/metal__f32__layer_norm__outer_axis__outer_strided_r64_c63.json) | 0.342 / 0.238 | [passed](program_peer/metal__f32__layer_norm__outer_axis__outer_strided_r64_c63.json) | 0.4376 / 0.2437 |
| `metal/f32/layer_norm/outer_axis/outer_strided_r64_c127` | normalization | 64 × 127 | strided | correct | [passed](operator_peer/metal__f32__layer_norm__outer_axis__outer_strided_r64_c127.json) | 0.424 / 0.2662 | [passed](program_peer/metal__f32__layer_norm__outer_axis__outer_strided_r64_c127.json) | 0.4262 / 0.2777 |
| `metal/f32/layer_norm/outer_axis/outer_strided_r64_c257` | normalization | 64 × 257 | strided | correct | [passed](operator_peer/metal__f32__layer_norm__outer_axis__outer_strided_r64_c257.json) | 0.5778 / 0.3641 | [passed](program_peer/metal__f32__layer_norm__outer_axis__outer_strided_r64_c257.json) | 0.5828 / 0.3642 |
| `metal/f32/layer_norm/outer_axis/outer_strided_r129_c17` | normalization | 129 × 17 | strided | correct | [passed](operator_peer/metal__f32__layer_norm__outer_axis__outer_strided_r129_c17.json) | 0.3238 / 0.2232 | [passed](program_peer/metal__f32__layer_norm__outer_axis__outer_strided_r129_c17.json) | 0.4934 / 0.2294 |
| `metal/f32/layer_norm/outer_axis/outer_strided_r129_c63` | normalization | 129 × 63 | strided | correct | [passed](operator_peer/metal__f32__layer_norm__outer_axis__outer_strided_r129_c63.json) | 0.457 / 0.2771 | [passed](program_peer/metal__f32__layer_norm__outer_axis__outer_strided_r129_c63.json) | 0.4636 / 0.2879 |
| `metal/f32/layer_norm/outer_axis/outer_strided_r129_c127` | normalization | 129 × 127 | strided | correct | [passed](operator_peer/metal__f32__layer_norm__outer_axis__outer_strided_r129_c127.json) | 0.6466 / 0.414 | [passed](program_peer/metal__f32__layer_norm__outer_axis__outer_strided_r129_c127.json) | 0.6518 / 0.4077 |
| `metal/f32/layer_norm/outer_axis/outer_strided_r129_c257` | normalization | 129 × 257 | strided | correct | [passed](operator_peer/metal__f32__layer_norm__outer_axis__outer_strided_r129_c257.json) | 1.068 / 0.6174 | [passed](program_peer/metal__f32__layer_norm__outer_axis__outer_strided_r129_c257.json) | 1.058 / 0.6285 |
| `metal/f32/layer_norm/outer_axis/outer_offset_r7_c257` | normalization | 7 × 257 | offset | correct | [passed](operator_peer/metal__f32__layer_norm__outer_axis__outer_offset_r7_c257.json) | 0.37 / 0.2178 | [passed](program_peer/metal__f32__layer_norm__outer_axis__outer_offset_r7_c257.json) | 0.4042 / 0.2182 |
| `metal/f32/layer_norm/outer_axis/outer_offset_r64_c17` | normalization | 64 × 17 | offset | correct | [passed](operator_peer/metal__f32__layer_norm__outer_axis__outer_offset_r64_c17.json) | 0.371 / 0.2147 | [passed](program_peer/metal__f32__layer_norm__outer_axis__outer_offset_r64_c17.json) | 0.3814 / 0.2009 |
| `metal/f32/layer_norm/outer_axis/outer_offset_r64_c63` | normalization | 64 × 63 | offset | correct | [passed](operator_peer/metal__f32__layer_norm__outer_axis__outer_offset_r64_c63.json) | 0.633 / 0.2385 | [passed](program_peer/metal__f32__layer_norm__outer_axis__outer_offset_r64_c63.json) | 0.3382 / 0.2436 |
| `metal/f32/layer_norm/outer_axis/outer_offset_r64_c127` | normalization | 64 × 127 | offset | correct | [passed](operator_peer/metal__f32__layer_norm__outer_axis__outer_offset_r64_c127.json) | 0.5606 / 0.2781 | [passed](program_peer/metal__f32__layer_norm__outer_axis__outer_offset_r64_c127.json) | 0.4264 / 0.2792 |
| `metal/f32/layer_norm/outer_axis/outer_offset_r64_c257` | normalization | 64 × 257 | offset | correct | [passed](operator_peer/metal__f32__layer_norm__outer_axis__outer_offset_r64_c257.json) | 0.577 / 0.3837 | [passed](program_peer/metal__f32__layer_norm__outer_axis__outer_offset_r64_c257.json) | 0.5796 / 0.3953 |
| `metal/f32/layer_norm/outer_axis/outer_offset_r129_c17` | normalization | 129 × 17 | offset | correct | [passed](operator_peer/metal__f32__layer_norm__outer_axis__outer_offset_r129_c17.json) | 0.5304 / 0.226 | [passed](program_peer/metal__f32__layer_norm__outer_axis__outer_offset_r129_c17.json) | 0.3206 / 0.2282 |
| `metal/f32/layer_norm/outer_axis/outer_offset_r129_c63` | normalization | 129 × 63 | offset | correct | [passed](operator_peer/metal__f32__layer_norm__outer_axis__outer_offset_r129_c63.json) | 0.4638 / 0.28 | [passed](program_peer/metal__f32__layer_norm__outer_axis__outer_offset_r129_c63.json) | 0.463 / 0.2872 |
| `metal/f32/layer_norm/outer_axis/outer_offset_r129_c127` | normalization | 129 × 127 | offset | correct | [passed](operator_peer/metal__f32__layer_norm__outer_axis__outer_offset_r129_c127.json) | 0.6524 / 0.4454 | [passed](program_peer/metal__f32__layer_norm__outer_axis__outer_offset_r129_c127.json) | 0.6468 / 0.4455 |
| `metal/f32/layer_norm/outer_axis/outer_offset_r129_c257` | normalization | 129 × 257 | offset | correct | [passed](operator_peer/metal__f32__layer_norm__outer_axis__outer_offset_r129_c257.json) | 1.086 / 0.6374 | [passed](program_peer/metal__f32__layer_norm__outer_axis__outer_offset_r129_c257.json) | 1.051 / 0.6514 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
