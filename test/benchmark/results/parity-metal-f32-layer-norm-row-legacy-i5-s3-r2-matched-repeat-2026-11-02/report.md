# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `81f732e915d327d158b49d832ba2f63e580ff0540787272d6c97589bec633793`  
Binary SHA-256: `1263f72409ea8b8e7c7c5ea44ec4bc78639d602290b1885391b28c52332c724e`  
Timing: 5 iterations × 3 samples × 2 rounds

## Checks

- Cases run: 15
- Numerical check passed: 15
- Operator check and timing passed: 15
- Program check and timing passed: 15
- Numerical failures: 0
- Operator failures: 0
- Program failures: 0

## Cases

| Case | Family | Shape | Layout | Numerical | Operator | Affon / PyTorch ms | Program | Affon / PyTorch ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `metal/f32/layer_norm/strided/inner_strided_r2_c63` | normalization | 2 × 63 | strided | correct | [passed](operator_peer/metal__f32__layer_norm__strided__inner_strided_r2_c63.json) | 0.2824 / 0.2444 | [passed](program_peer/metal__f32__layer_norm__strided__inner_strided_r2_c63.json) | 0.2993 / 0.2328 |
| `metal/f32/layer_norm/strided/inner_strided_r2_c127` | normalization | 2 × 127 | strided | correct | [passed](operator_peer/metal__f32__layer_norm__strided__inner_strided_r2_c127.json) | 0.3018 / 0.19 | [passed](program_peer/metal__f32__layer_norm__strided__inner_strided_r2_c127.json) | 0.298 / 0.202 |
| `metal/f32/layer_norm/strided/inner_strided_r2_c257` | normalization | 2 × 257 | strided | correct | [passed](operator_peer/metal__f32__layer_norm__strided__inner_strided_r2_c257.json) | 0.313 / 0.1843 | [passed](program_peer/metal__f32__layer_norm__strided__inner_strided_r2_c257.json) | 0.3396 / 0.1895 |
| `metal/f32/layer_norm/strided/inner_strided_r7_c17` | normalization | 7 × 17 | strided | correct | [passed](operator_peer/metal__f32__layer_norm__strided__inner_strided_r7_c17.json) | 0.3008 / 0.1819 | [passed](program_peer/metal__f32__layer_norm__strided__inner_strided_r7_c17.json) | 0.2847 / 0.227 |
| `metal/f32/layer_norm/strided/inner_strided_r7_c63` | normalization | 7 × 63 | strided | correct | [passed](operator_peer/metal__f32__layer_norm__strided__inner_strided_r7_c63.json) | 0.3632 / 0.2239 | [passed](program_peer/metal__f32__layer_norm__strided__inner_strided_r7_c63.json) | 0.3499 / 0.2029 |
| `metal/f32/layer_norm/strided/inner_strided_r7_c257` | normalization | 7 × 257 | strided | correct | [passed](operator_peer/metal__f32__layer_norm__strided__inner_strided_r7_c257.json) | 0.3105 / 0.2189 | [passed](program_peer/metal__f32__layer_norm__strided__inner_strided_r7_c257.json) | 0.324 / 0.2411 |
| `metal/f32/layer_norm/strided/inner_offset_r2_c257` | normalization | 2 × 257 | offset | correct | [passed](operator_peer/metal__f32__layer_norm__strided__inner_offset_r2_c257.json) | 0.3552 / 0.2001 | [passed](program_peer/metal__f32__layer_norm__strided__inner_offset_r2_c257.json) | 0.2587 / 0.2166 |
| `metal/f32/layer_norm/strided/inner_offset_r64_c127` | normalization | 64 × 127 | offset | correct | [passed](operator_peer/metal__f32__layer_norm__strided__inner_offset_r64_c127.json) | 0.3788 / 0.2742 | [passed](program_peer/metal__f32__layer_norm__strided__inner_offset_r64_c127.json) | 0.4134 / 0.2682 |
| `metal/f32/layer_norm/strided/inner_offset_r64_c257` | normalization | 64 × 257 | offset | correct | [passed](operator_peer/metal__f32__layer_norm__strided__inner_offset_r64_c257.json) | 0.4966 / 0.4072 | [passed](program_peer/metal__f32__layer_norm__strided__inner_offset_r64_c257.json) | 0.4424 / 0.4057 |
| `metal/f32/layer_norm/outer_axis/outer_strided_r2_c17` | normalization | 2 × 17 | strided | correct | [passed](operator_peer/metal__f32__layer_norm__outer_axis__outer_strided_r2_c17.json) | 0.3916 / 0.1895 | [passed](program_peer/metal__f32__layer_norm__outer_axis__outer_strided_r2_c17.json) | 0.3413 / 0.1949 |
| `metal/f32/layer_norm/outer_axis/outer_strided_r2_c257` | normalization | 2 × 257 | strided | correct | [passed](operator_peer/metal__f32__layer_norm__outer_axis__outer_strided_r2_c257.json) | 0.3028 / 0.203 | [passed](program_peer/metal__f32__layer_norm__outer_axis__outer_strided_r2_c257.json) | 0.3545 / 0.1941 |
| `metal/f32/layer_norm/outer_axis/outer_strided_r64_c63` | normalization | 64 × 63 | strided | correct | [passed](operator_peer/metal__f32__layer_norm__outer_axis__outer_strided_r64_c63.json) | 0.6147 / 0.2405 | [passed](program_peer/metal__f32__layer_norm__outer_axis__outer_strided_r64_c63.json) | 0.5786 / 0.2424 |
| `metal/f32/layer_norm/outer_axis/outer_offset_r2_c17` | normalization | 2 × 17 | offset | correct | [passed](operator_peer/metal__f32__layer_norm__outer_axis__outer_offset_r2_c17.json) | 0.4229 / 0.1988 | [passed](program_peer/metal__f32__layer_norm__outer_axis__outer_offset_r2_c17.json) | 0.4486 / 0.1959 |
| `metal/f32/layer_norm/outer_axis/outer_offset_r64_c63` | normalization | 64 × 63 | offset | correct | [passed](operator_peer/metal__f32__layer_norm__outer_axis__outer_offset_r64_c63.json) | 0.3366 / 0.2374 | [passed](program_peer/metal__f32__layer_norm__outer_axis__outer_offset_r64_c63.json) | 0.3371 / 0.2409 |
| `metal/f32/layer_norm/outer_axis/outer_offset_r64_c127` | normalization | 64 × 127 | offset | correct | [passed](operator_peer/metal__f32__layer_norm__outer_axis__outer_offset_r64_c127.json) | 0.4285 / 0.2761 | [passed](program_peer/metal__f32__layer_norm__outer_axis__outer_offset_r64_c127.json) | 0.6021 / 0.2683 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
