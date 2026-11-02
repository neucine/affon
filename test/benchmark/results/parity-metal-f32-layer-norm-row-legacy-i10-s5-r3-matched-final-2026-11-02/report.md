# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `7b60329ad5192420e3e88ccc94f25b3fc2ed69dcf9c0513b84ac3d451524527a`  
Binary SHA-256: `1263f72409ea8b8e7c7c5ea44ec4bc78639d602290b1885391b28c52332c724e`  
Timing: 10 iterations × 5 samples × 3 rounds

## Checks

- Cases run: 34
- Numerical check passed: 34
- Operator check and timing passed: 34
- Program check and timing passed: 34
- Numerical failures: 0
- Operator failures: 0
- Program failures: 0

## Cases

| Case | Family | Shape | Layout | Numerical | Operator | Affon / PyTorch ms | Program | Affon / PyTorch ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `metal/f32/layer_norm/short_row` | normalization | 4 × 32 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__short_row.json) | 0.2999 / 0.1968 | [passed](program_peer/metal__f32__layer_norm__short_row.json) | 0.293 / 0.1936 |
| `metal/f32/layer_norm/long_row` | normalization | 128 × 512 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__long_row.json) | 0.8003 / 0.8666 | [passed](program_peer/metal__f32__layer_norm__long_row.json) | 0.8027 / 0.8908 |
| `metal/f32/layer_norm/strided` | normalization | 16 × 64 | strided | correct | [passed](operator_peer/metal__f32__layer_norm__strided.json) | 0.3452 / 0.2038 | [passed](program_peer/metal__f32__layer_norm__strided.json) | 0.3519 / 0.2045 |
| `metal/f32/layer_norm/ragged` | normalization | 17 × 65 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__ragged.json) | 0.2789 / 0.2014 | [passed](program_peer/metal__f32__layer_norm__ragged.json) | 0.2587 / 0.2023 |
| `metal/f32/layer_norm/outer_axis` | normalization | 32 × 64 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__outer_axis.json) | 0.3398 / 0.2184 | [passed](program_peer/metal__f32__layer_norm__outer_axis.json) | 0.3021 / 0.2185 |
| `metal/f32/layer_norm/short_row/r32_c31` | normalization | 32 × 31 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__short_row__r32_c31.json) | 0.3162 / 0.1953 | [passed](program_peer/metal__f32__layer_norm__short_row__r32_c31.json) | 0.3268 / 0.2029 |
| `metal/f32/layer_norm/short_row/r32_c32` | normalization | 32 × 32 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__short_row__r32_c32.json) | 0.317 / 0.2018 | [passed](program_peer/metal__f32__layer_norm__short_row__r32_c32.json) | 0.2481 / 0.2011 |
| `metal/f32/layer_norm/short_row/r32_c33` | normalization | 32 × 33 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__short_row__r32_c33.json) | 0.2551 / 0.205 | [passed](program_peer/metal__f32__layer_norm__short_row__r32_c33.json) | 0.3279 / 0.2053 |
| `metal/f32/layer_norm/long_row/r32_c63` | normalization | 32 × 63 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__long_row__r32_c63.json) | 0.2833 / 0.2086 | [passed](program_peer/metal__f32__layer_norm__long_row__r32_c63.json) | 0.3114 / 0.2141 |
| `metal/f32/layer_norm/long_row/r32_c64` | normalization | 32 × 64 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__long_row__r32_c64.json) | 0.3177 / 0.2181 | [passed](program_peer/metal__f32__layer_norm__long_row__r32_c64.json) | 0.2833 / 0.2183 |
| `metal/f32/layer_norm/long_row/r32_c65` | normalization | 32 × 65 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__long_row__r32_c65.json) | 0.3981 / 0.2181 | [passed](program_peer/metal__f32__layer_norm__long_row__r32_c65.json) | 0.4135 / 0.2193 |
| `metal/f32/layer_norm/long_row/r32_c127` | normalization | 32 × 127 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__long_row__r32_c127.json) | 0.3281 / 0.2334 | [passed](program_peer/metal__f32__layer_norm__long_row__r32_c127.json) | 0.3271 / 0.2377 |
| `metal/f32/layer_norm/long_row/r32_c128` | normalization | 32 × 128 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__long_row__r32_c128.json) | 0.3185 / 0.234 | [passed](program_peer/metal__f32__layer_norm__long_row__r32_c128.json) | 0.3275 / 0.2387 |
| `metal/f32/layer_norm/long_row/r32_c129` | normalization | 32 × 129 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__long_row__r32_c129.json) | 0.2599 / 0.2347 | [passed](program_peer/metal__f32__layer_norm__long_row__r32_c129.json) | 0.3417 / 0.2381 |
| `metal/f32/layer_norm/long_row/r256_c64` | normalization | 256 × 64 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__long_row__r256_c64.json) | 0.4609 / 0.3935 | [passed](program_peer/metal__f32__layer_norm__long_row__r256_c64.json) | 0.4882 / 0.4058 |
| `metal/f32/layer_norm/strided/r32_c127` | normalization | 32 × 127 | strided | correct | [passed](operator_peer/metal__f32__layer_norm__strided__r32_c127.json) | 0.3334 / 0.2373 | [passed](program_peer/metal__f32__layer_norm__strided__r32_c127.json) | 0.3184 / 0.2396 |
| `metal/f32/layer_norm/strided/r64_c128` | normalization | 64 × 128 | strided | correct | [passed](operator_peer/metal__f32__layer_norm__strided__r64_c128.json) | 0.3962 / 0.275 | [passed](program_peer/metal__f32__layer_norm__strided__r64_c128.json) | 0.402 / 0.2777 |
| `metal/f32/layer_norm/ragged/r17_c65` | normalization | 17 × 65 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__ragged__r17_c65.json) | 0.2839 / 0.203 | [passed](program_peer/metal__f32__layer_norm__ragged__r17_c65.json) | 0.32 / 0.203 |
| `metal/f32/layer_norm/ragged/r33_c129` | normalization | 33 × 129 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__ragged__r33_c129.json) | 0.3431 / 0.237 | [passed](program_peer/metal__f32__layer_norm__ragged__r33_c129.json) | 0.3198 / 0.2395 |
| `metal/f32/layer_norm/outer_axis/r127_c64` | normalization | 127 × 64 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__outer_axis__r127_c64.json) | 0.472 / 0.2761 | [passed](program_peer/metal__f32__layer_norm__outer_axis__r127_c64.json) | 0.4666 / 0.2744 |
| `metal/f32/layer_norm/outer_axis/r64_c129` | normalization | 64 × 129 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__outer_axis__r64_c129.json) | 0.42 / 0.2736 | [passed](program_peer/metal__f32__layer_norm__outer_axis__r64_c129.json) | 0.4203 / 0.2742 |
| `metal/f32/layer_norm/long_row/r256_c31` | normalization | 256 × 31 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__long_row__r256_c31.json) | 0.4488 / 0.284 | [passed](program_peer/metal__f32__layer_norm__long_row__r256_c31.json) | 0.4267 / 0.2778 |
| `metal/f32/layer_norm/long_row/r128_c129` | normalization | 128 × 129 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__long_row__r128_c129.json) | 0.4149 / 0.3842 | [passed](program_peer/metal__f32__layer_norm__long_row__r128_c129.json) | 0.4295 / 0.4113 |
| `metal/f32/layer_norm/long_row/r33_c255` | normalization | 33 × 255 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__long_row__r33_c255.json) | 0.4001 / 0.2739 | [passed](program_peer/metal__f32__layer_norm__long_row__r33_c255.json) | 0.3084 / 0.2776 |
| `metal/f32/layer_norm/long_row/r64_c255` | normalization | 64 × 255 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__long_row__r64_c255.json) | 0.3873 / 0.3986 | [passed](program_peer/metal__f32__layer_norm__long_row__r64_c255.json) | 0.394 / 0.4091 |
| `metal/f32/layer_norm/short_row/r31_c16` | normalization | 31 × 16 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__short_row__r31_c16.json) | 0.276 / 0.2034 | [passed](program_peer/metal__f32__layer_norm__short_row__r31_c16.json) | 0.2794 / 0.1955 |
| `metal/f32/layer_norm/short_row/r8_c7` | normalization | 8 × 7 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__short_row__r8_c7.json) | 0.2309 / 0.1785 | [passed](program_peer/metal__f32__layer_norm__short_row__r8_c7.json) | 0.258 / 0.194 |
| `metal/f32/layer_norm/short_row/r8_c15` | normalization | 8 × 15 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__short_row__r8_c15.json) | 0.2637 / 0.1917 | [passed](program_peer/metal__f32__layer_norm__short_row__r8_c15.json) | 0.2644 / 0.1871 |
| `metal/f32/layer_norm/short_row/r8_c16` | normalization | 8 × 16 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__short_row__r8_c16.json) | 0.2596 / 0.1923 | [passed](program_peer/metal__f32__layer_norm__short_row__r8_c16.json) | 0.2692 / 0.1906 |
| `metal/f32/layer_norm/short_row/r8_c17` | normalization | 8 × 17 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__short_row__r8_c17.json) | 0.2625 / 0.1911 | [passed](program_peer/metal__f32__layer_norm__short_row__r8_c17.json) | 0.2818 / 0.1909 |
| `metal/f32/layer_norm/short_row/r16_c31` | normalization | 16 × 31 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__short_row__r16_c31.json) | 0.3059 / 0.1963 | [passed](program_peer/metal__f32__layer_norm__short_row__r16_c31.json) | 0.3169 / 0.194 |
| `metal/f32/layer_norm/short_row/r16_c32` | normalization | 16 × 32 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__short_row__r16_c32.json) | 0.2668 / 0.1942 | [passed](program_peer/metal__f32__layer_norm__short_row__r16_c32.json) | 0.3106 / 0.1967 |
| `metal/f32/layer_norm/short_row/r16_c33` | normalization | 16 × 33 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__short_row__r16_c33.json) | 0.2565 / 0.195 | [passed](program_peer/metal__f32__layer_norm__short_row__r16_c33.json) | 0.3144 / 0.1968 |
| `metal/f32/layer_norm/long_row/r32_c257` | normalization | 32 × 257 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__long_row__r32_c257.json) | 0.3081 / 0.2735 | [passed](program_peer/metal__f32__layer_norm__long_row__r32_c257.json) | 0.3692 / 0.2729 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
