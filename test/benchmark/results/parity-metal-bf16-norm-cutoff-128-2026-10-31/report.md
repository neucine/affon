# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `7e102992aabc663cbe4ff708c850448c54ab1a15a5527c56952a306d80745765`  
Binary SHA-256: `68869bbfc6e3e488ddede2df99d960be78880a24d273eea6f9bc10ba171d4c47`  
Timing: 10 iterations × 5 samples × 3 rounds

## Checks

- Cases run: 29
- Numerical check passed: 29
- Operator check and timing passed: 29
- Program check and timing passed: 29
- Numerical failures: 0
- Operator failures: 0
- Program failures: 0

## Cases

| Case | Family | Shape | Layout | Numerical | Operator | Affon / PyTorch ms | Program | Affon / PyTorch ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `metal/bf16/layer_norm/strided/inner_strided_r3_c129` | normalization | 3 × 129 | strided | correct | [passed](operator_peer/metal__bf16__layer_norm__strided__inner_strided_r3_c129.json) | 0.2619 / 0.2008 | [passed](program_peer/metal__bf16__layer_norm__strided__inner_strided_r3_c129.json) | 0.2952 / 0.2022 |
| `metal/bf16/layer_norm/strided/inner_strided_r16_c129` | normalization | 16 × 129 | strided | correct | [passed](operator_peer/metal__bf16__layer_norm__strided__inner_strided_r16_c129.json) | 0.2826 / 0.2232 | [passed](program_peer/metal__bf16__layer_norm__strided__inner_strided_r16_c129.json) | 0.3256 / 0.2246 |
| `metal/bf16/layer_norm/strided/inner_strided_r33_c129` | normalization | 33 × 129 | strided | correct | [passed](operator_peer/metal__bf16__layer_norm__strided__inner_strided_r33_c129.json) | 0.2965 / 0.241 | [passed](program_peer/metal__bf16__layer_norm__strided__inner_strided_r33_c129.json) | 0.3406 / 0.2435 |
| `metal/bf16/layer_norm/strided/inner_strided_r128_c129` | normalization | 128 × 129 | strided | correct | [passed](operator_peer/metal__bf16__layer_norm__strided__inner_strided_r128_c129.json) | 0.5105 / 0.4196 | [passed](program_peer/metal__bf16__layer_norm__strided__inner_strided_r128_c129.json) | 0.552 / 0.4263 |
| `metal/bf16/layer_norm/strided/inner_offset_r3_c129` | normalization | 3 × 129 | offset | correct | [passed](operator_peer/metal__bf16__layer_norm__strided__inner_offset_r3_c129.json) | 0.2558 / 0.1972 | [passed](program_peer/metal__bf16__layer_norm__strided__inner_offset_r3_c129.json) | 0.2556 / 0.195 |
| `metal/bf16/layer_norm/strided/inner_offset_r16_c129` | normalization | 16 × 129 | offset | correct | [passed](operator_peer/metal__bf16__layer_norm__strided__inner_offset_r16_c129.json) | 0.2828 / 0.2193 | [passed](program_peer/metal__bf16__layer_norm__strided__inner_offset_r16_c129.json) | 0.3152 / 0.2098 |
| `metal/bf16/layer_norm/strided/inner_offset_r33_c129` | normalization | 33 × 129 | offset | correct | [passed](operator_peer/metal__bf16__layer_norm__strided__inner_offset_r33_c129.json) | 0.3215 / 0.2412 | [passed](program_peer/metal__bf16__layer_norm__strided__inner_offset_r33_c129.json) | 0.3423 / 0.2398 |
| `metal/bf16/layer_norm/strided/inner_offset_r128_c129` | normalization | 128 × 129 | offset | correct | [passed](operator_peer/metal__bf16__layer_norm__strided__inner_offset_r128_c129.json) | 0.4158 / 0.4205 | [passed](program_peer/metal__bf16__layer_norm__strided__inner_offset_r128_c129.json) | 0.4567 / 0.3899 |
| `metal/bf16/layer_norm/outer_axis/outer_strided_r3_c65` | normalization | 3 × 65 | strided | correct | [passed](operator_peer/metal__bf16__layer_norm__outer_axis__outer_strided_r3_c65.json) | 0.2552 / 0.1934 | [passed](program_peer/metal__bf16__layer_norm__outer_axis__outer_strided_r3_c65.json) | 0.278 / 0.203 |
| `metal/bf16/layer_norm/outer_axis/outer_strided_r33_c5` | normalization | 33 × 5 | strided | correct | [passed](operator_peer/metal__bf16__layer_norm__outer_axis__outer_strided_r33_c5.json) | 0.2461 / 0.1911 | [passed](program_peer/metal__bf16__layer_norm__outer_axis__outer_strided_r33_c5.json) | 0.3015 / 0.1987 |
| `metal/bf16/layer_norm/outer_axis/outer_offset_r3_c65` | normalization | 3 × 65 | offset | correct | [passed](operator_peer/metal__bf16__layer_norm__outer_axis__outer_offset_r3_c65.json) | 0.2551 / 0.1936 | [passed](program_peer/metal__bf16__layer_norm__outer_axis__outer_offset_r3_c65.json) | 0.2818 / 0.1912 |
| `metal/bf16/layer_norm/outer_axis/outer_offset_r33_c5` | normalization | 33 × 5 | offset | correct | [passed](operator_peer/metal__bf16__layer_norm__outer_axis__outer_offset_r33_c5.json) | 0.2645 / 0.1883 | [passed](program_peer/metal__bf16__layer_norm__outer_axis__outer_offset_r33_c5.json) | 0.3113 / 0.1962 |
| `metal/bf16/rms_norm/outer_axis/strided_r3_c65` | normalization | 3 × 65 | strided | correct | [passed](operator_peer/metal__bf16__rms_norm__outer_axis__strided_r3_c65.json) | 0.2533 / 0.2518 | [passed](program_peer/metal__bf16__rms_norm__outer_axis__strided_r3_c65.json) | 0.2595 / 0.2577 |
| `metal/bf16/rms_norm/outer_axis/strided_r33_c5` | normalization | 33 × 5 | strided | correct | [passed](operator_peer/metal__bf16__rms_norm__outer_axis__strided_r33_c5.json) | 0.2706 / 0.2514 | [passed](program_peer/metal__bf16__rms_norm__outer_axis__strided_r33_c5.json) | 0.3041 / 0.253 |
| `metal/bf16/rms_norm/outer_axis/offset_r3_c65` | normalization | 3 × 65 | offset | correct | [passed](operator_peer/metal__bf16__rms_norm__outer_axis__offset_r3_c65.json) | 0.255 / 0.2529 | [passed](program_peer/metal__bf16__rms_norm__outer_axis__offset_r3_c65.json) | 0.2902 / 0.2563 |
| `metal/bf16/rms_norm/outer_axis/offset_r33_c5` | normalization | 33 × 5 | offset | correct | [passed](operator_peer/metal__bf16__rms_norm__outer_axis__offset_r33_c5.json) | 0.2643 / 0.2514 | [passed](program_peer/metal__bf16__rms_norm__outer_axis__offset_r33_c5.json) | 0.2522 / 0.263 |
| `metal/bf16/rms_norm/strided/strided_r3_c129` | normalization | 3 × 129 | strided | correct | [passed](operator_peer/metal__bf16__rms_norm__strided__strided_r3_c129.json) | 0.2515 / 0.2564 | [passed](program_peer/metal__bf16__rms_norm__strided__strided_r3_c129.json) | 0.2386 / 0.2643 |
| `metal/bf16/rms_norm/strided/strided_r16_c129` | normalization | 16 × 129 | strided | correct | [passed](operator_peer/metal__bf16__rms_norm__strided__strided_r16_c129.json) | 0.2789 / 0.2663 | [passed](program_peer/metal__bf16__rms_norm__strided__strided_r16_c129.json) | 0.2631 / 0.2936 |
| `metal/bf16/rms_norm/strided/strided_r33_c129` | normalization | 33 × 129 | strided | correct | [passed](operator_peer/metal__bf16__rms_norm__strided__strided_r33_c129.json) | 0.3234 / 0.3071 | [passed](program_peer/metal__bf16__rms_norm__strided__strided_r33_c129.json) | 0.3253 / 0.3155 |
| `metal/bf16/rms_norm/strided/strided_r128_c129` | normalization | 128 × 129 | strided | correct | [passed](operator_peer/metal__bf16__rms_norm__strided__strided_r128_c129.json) | 0.5052 / 0.48 | [passed](program_peer/metal__bf16__rms_norm__strided__strided_r128_c129.json) | 0.5741 / 0.4459 |
| `metal/bf16/rms_norm/long_row/offset_r3_c129` | normalization | 3 × 129 | offset | correct | [passed](operator_peer/metal__bf16__rms_norm__long_row__offset_r3_c129.json) | 0.2602 / 0.2639 | [passed](program_peer/metal__bf16__rms_norm__long_row__offset_r3_c129.json) | 0.2566 / 0.251 |
| `metal/bf16/rms_norm/long_row/offset_r16_c129` | normalization | 16 × 129 | offset | correct | [passed](operator_peer/metal__bf16__rms_norm__long_row__offset_r16_c129.json) | 0.2835 / 0.2786 | [passed](program_peer/metal__bf16__rms_norm__long_row__offset_r16_c129.json) | 0.2917 / 0.2766 |
| `metal/bf16/rms_norm/long_row/offset_r33_c129` | normalization | 33 × 129 | offset | correct | [passed](operator_peer/metal__bf16__rms_norm__long_row__offset_r33_c129.json) | 0.3021 / 0.3027 | [passed](program_peer/metal__bf16__rms_norm__long_row__offset_r33_c129.json) | 0.3306 / 0.2985 |
| `metal/bf16/rms_norm/long_row/offset_r128_c129` | normalization | 128 × 129 | offset | correct | [passed](operator_peer/metal__bf16__rms_norm__long_row__offset_r128_c129.json) | 0.4097 / 0.5419 | [passed](program_peer/metal__bf16__rms_norm__long_row__offset_r128_c129.json) | 0.4409 / 0.4618 |
| `metal/bf16/layer_norm/outer_axis/outer_strided_r2_c127` | normalization | 2 × 127 | strided | correct | [passed](operator_peer/metal__bf16__layer_norm__outer_axis__outer_strided_r2_c127.json) | 0.2533 / 0.1893 | [passed](program_peer/metal__bf16__layer_norm__outer_axis__outer_strided_r2_c127.json) | 0.2557 / 0.1908 |
| `metal/bf16/layer_norm/outer_axis/outer_offset_r2_c127` | normalization | 2 × 127 | offset | correct | [passed](operator_peer/metal__bf16__layer_norm__outer_axis__outer_offset_r2_c127.json) | 0.2232 / 0.1937 | [passed](program_peer/metal__bf16__layer_norm__outer_axis__outer_offset_r2_c127.json) | 0.2484 / 0.1855 |
| `metal/bf16/rms_norm/outer_axis/strided_r2_c127` | normalization | 2 × 127 | strided | correct | [passed](operator_peer/metal__bf16__rms_norm__outer_axis__strided_r2_c127.json) | 0.2528 / 0.2528 | [passed](program_peer/metal__bf16__rms_norm__outer_axis__strided_r2_c127.json) | 0.2689 / 0.2474 |
| `metal/bf16/rms_norm/outer_axis/offset_r2_c127` | normalization | 2 × 127 | offset | correct | [passed](operator_peer/metal__bf16__rms_norm__outer_axis__offset_r2_c127.json) | 0.2071 / 0.2566 | [passed](program_peer/metal__bf16__rms_norm__outer_axis__offset_r2_c127.json) | 0.2727 / 0.2641 |
| `metal/bf16/rms_norm/long_row/r64_c255` | normalization | 64 × 255 | dense | correct | [passed](operator_peer/metal__bf16__rms_norm__long_row__r64_c255.json) | 0.4112 / 0.5222 | [passed](program_peer/metal__bf16__rms_norm__long_row__r64_c255.json) | 0.3937 / 0.458 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
