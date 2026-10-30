# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `b8a74ce7b43dec6c59440be6a93d3324fcc5cfe33f39fafb8affaa86f9ba9787`  
Binary SHA-256: `94e3c1795a7cbcae6e6202b022ce83d5f4d5a69f03d81ab1e9de230af9c7bc4e`  
Timing: 5 iterations × 5 samples × 3 rounds

## Checks

- Cases run: 12
- Numerical check passed: 12
- Operator check and timing passed: 12
- Program check and timing passed: 12
- Numerical failures: 0
- Operator failures: 0
- Program failures: 0

## Cases

| Case | Family | Shape | Layout | Numerical | Operator | Affon / PyTorch ms | Program | Affon / PyTorch ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `metal/bf16/layer_norm/outer_axis/outer_strided_r3_c129` | normalization | 3 × 129 | strided | correct | [passed](operator_peer/metal__bf16__layer_norm__outer_axis__outer_strided_r3_c129.json) | 0.2522 / 0.2055 | [passed](program_peer/metal__bf16__layer_norm__outer_axis__outer_strided_r3_c129.json) | 0.3064 / 0.1967 |
| `metal/bf16/layer_norm/outer_axis/outer_strided_r16_c32` | normalization | 16 × 32 | strided | correct | [passed](operator_peer/metal__bf16__layer_norm__outer_axis__outer_strided_r16_c32.json) | 0.268 / 0.2398 | [passed](program_peer/metal__bf16__layer_norm__outer_axis__outer_strided_r16_c32.json) | 0.3158 / 0.2076 |
| `metal/bf16/layer_norm/outer_axis/outer_strided_r128_c5` | normalization | 128 × 5 | strided | correct | [passed](operator_peer/metal__bf16__layer_norm__outer_axis__outer_strided_r128_c5.json) | 0.2798 / 0.2035 | [passed](program_peer/metal__bf16__layer_norm__outer_axis__outer_strided_r128_c5.json) | 0.3524 / 0.2124 |
| `metal/bf16/layer_norm/outer_axis/outer_offset_r3_c129` | normalization | 3 × 129 | offset | correct | [passed](operator_peer/metal__bf16__layer_norm__outer_axis__outer_offset_r3_c129.json) | 0.2676 / 0.201 | [passed](program_peer/metal__bf16__layer_norm__outer_axis__outer_offset_r3_c129.json) | 0.294 / 0.1935 |
| `metal/bf16/layer_norm/outer_axis/outer_offset_r16_c32` | normalization | 16 × 32 | offset | correct | [passed](operator_peer/metal__bf16__layer_norm__outer_axis__outer_offset_r16_c32.json) | 0.2798 / 0.2035 | [passed](program_peer/metal__bf16__layer_norm__outer_axis__outer_offset_r16_c32.json) | 0.269 / 0.1945 |
| `metal/bf16/layer_norm/outer_axis/outer_offset_r128_c5` | normalization | 128 × 5 | offset | correct | [passed](operator_peer/metal__bf16__layer_norm__outer_axis__outer_offset_r128_c5.json) | 0.3314 / 0.2013 | [passed](program_peer/metal__bf16__layer_norm__outer_axis__outer_offset_r128_c5.json) | 0.3504 / 0.2065 |
| `metal/bf16/rms_norm/outer_axis/strided_r3_c129` | normalization | 3 × 129 | strided | correct | [passed](operator_peer/metal__bf16__rms_norm__outer_axis__strided_r3_c129.json) | 0.2698 / 0.256 | [passed](program_peer/metal__bf16__rms_norm__outer_axis__strided_r3_c129.json) | 0.2582 / 0.2665 |
| `metal/bf16/rms_norm/outer_axis/strided_r16_c32` | normalization | 16 × 32 | strided | correct | [passed](operator_peer/metal__bf16__rms_norm__outer_axis__strided_r16_c32.json) | 0.2758 / 0.2584 | [passed](program_peer/metal__bf16__rms_norm__outer_axis__strided_r16_c32.json) | 0.3114 / 0.2664 |
| `metal/bf16/rms_norm/outer_axis/strided_r128_c5` | normalization | 128 × 5 | strided | correct | [passed](operator_peer/metal__bf16__rms_norm__outer_axis__strided_r128_c5.json) | 0.3044 / 0.2582 | [passed](program_peer/metal__bf16__rms_norm__outer_axis__strided_r128_c5.json) | 0.3496 / 0.2744 |
| `metal/bf16/rms_norm/outer_axis/offset_r3_c129` | normalization | 3 × 129 | offset | correct | [passed](operator_peer/metal__bf16__rms_norm__outer_axis__offset_r3_c129.json) | 0.269 / 0.2576 | [passed](program_peer/metal__bf16__rms_norm__outer_axis__offset_r3_c129.json) | 0.3022 / 0.2555 |
| `metal/bf16/rms_norm/outer_axis/offset_r16_c32` | normalization | 16 × 32 | offset | correct | [passed](operator_peer/metal__bf16__rms_norm__outer_axis__offset_r16_c32.json) | 0.2804 / 0.269 | [passed](program_peer/metal__bf16__rms_norm__outer_axis__offset_r16_c32.json) | 0.3152 / 0.257 |
| `metal/bf16/rms_norm/outer_axis/offset_r128_c5` | normalization | 128 × 5 | offset | correct | [passed](operator_peer/metal__bf16__rms_norm__outer_axis__offset_r128_c5.json) | 0.3342 / 0.266 | [passed](program_peer/metal__bf16__rms_norm__outer_axis__offset_r128_c5.json) | 0.3504 / 0.2635 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
