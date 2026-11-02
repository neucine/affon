# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `763ec48dc4b4109b7298287e34e435c7dd3e35402924fdfc8ea27589ded78199`  
Binary SHA-256: `07d2686a6fde2c671da3d04668ac145d8ca91e78bd6eb55675701480f9735945`  
Timing: 5 iterations × 3 samples × 2 rounds

## Checks

- Cases run: 32
- Numerical check passed: 32
- Operator check and timing passed: 32
- Program check and timing passed: 32
- Numerical failures: 0
- Operator failures: 0
- Program failures: 0

## Cases

| Case | Family | Shape | Layout | Numerical | Operator | Affon / PyTorch ms | Program | Affon / PyTorch ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `metal/bf16/layer_norm/strided/inner_strided_r2_c17` | normalization | 2 × 17 | strided | correct | [passed](operator_peer/metal__bf16__layer_norm__strided__inner_strided_r2_c17.json) | 0.2611 / 0.2472 | [passed](program_peer/metal__bf16__layer_norm__strided__inner_strided_r2_c17.json) | 0.2637 / 0.1956 |
| `metal/bf16/layer_norm/strided/inner_strided_r2_c63` | normalization | 2 × 63 | strided | correct | [passed](operator_peer/metal__bf16__layer_norm__strided__inner_strided_r2_c63.json) | 0.3281 / 0.2434 | [passed](program_peer/metal__bf16__layer_norm__strided__inner_strided_r2_c63.json) | 0.3248 / 0.1923 |
| `metal/bf16/layer_norm/strided/inner_strided_r2_c257` | normalization | 2 × 257 | strided | correct | [passed](operator_peer/metal__bf16__layer_norm__strided__inner_strided_r2_c257.json) | 0.2605 / 0.2099 | [passed](program_peer/metal__bf16__layer_norm__strided__inner_strided_r2_c257.json) | 0.2951 / 0.1829 |
| `metal/bf16/layer_norm/strided/inner_strided_r7_c17` | normalization | 7 × 17 | strided | correct | [passed](operator_peer/metal__bf16__layer_norm__strided__inner_strided_r7_c17.json) | 0.3228 / 0.1977 | [passed](program_peer/metal__bf16__layer_norm__strided__inner_strided_r7_c17.json) | 0.2381 / 0.1829 |
| `metal/bf16/layer_norm/strided/inner_strided_r7_c63` | normalization | 7 × 63 | strided | correct | [passed](operator_peer/metal__bf16__layer_norm__strided__inner_strided_r7_c63.json) | 0.3334 / 0.2231 | [passed](program_peer/metal__bf16__layer_norm__strided__inner_strided_r7_c63.json) | 0.3556 / 0.2075 |
| `metal/bf16/layer_norm/strided/inner_strided_r7_c257` | normalization | 7 × 257 | strided | correct | [passed](operator_peer/metal__bf16__layer_norm__strided__inner_strided_r7_c257.json) | 0.2706 / 0.2172 | [passed](program_peer/metal__bf16__layer_norm__strided__inner_strided_r7_c257.json) | 0.2497 / 0.2195 |
| `metal/bf16/layer_norm/strided/inner_strided_r64_c17` | normalization | 64 × 17 | strided | correct | [passed](operator_peer/metal__bf16__layer_norm__strided__inner_strided_r64_c17.json) | 0.3018 / 0.2009 | [passed](program_peer/metal__bf16__layer_norm__strided__inner_strided_r64_c17.json) | 0.3421 / 0.207 |
| `metal/bf16/layer_norm/strided/inner_strided_r64_c63` | normalization | 64 × 63 | strided | correct | [passed](operator_peer/metal__bf16__layer_norm__strided__inner_strided_r64_c63.json) | 0.4171 / 0.2407 | [passed](program_peer/metal__bf16__layer_norm__strided__inner_strided_r64_c63.json) | 0.433 / 0.2451 |
| `metal/bf16/layer_norm/strided/inner_strided_r64_c257` | normalization | 64 × 257 | strided | correct | [passed](operator_peer/metal__bf16__layer_norm__strided__inner_strided_r64_c257.json) | 0.3742 / 0.4096 | [passed](program_peer/metal__bf16__layer_norm__strided__inner_strided_r64_c257.json) | 0.4608 / 0.4156 |
| `metal/bf16/layer_norm/strided/inner_strided_r129_c17` | normalization | 129 × 17 | strided | correct | [passed](operator_peer/metal__bf16__layer_norm__strided__inner_strided_r129_c17.json) | 0.3169 / 0.2286 | [passed](program_peer/metal__bf16__layer_norm__strided__inner_strided_r129_c17.json) | 0.3589 / 0.2273 |
| `metal/bf16/layer_norm/strided/inner_strided_r129_c63` | normalization | 129 × 63 | strided | correct | [passed](operator_peer/metal__bf16__layer_norm__strided__inner_strided_r129_c63.json) | 0.4556 / 0.3306 | [passed](program_peer/metal__bf16__layer_norm__strided__inner_strided_r129_c63.json) | 0.5323 / 0.3254 |
| `metal/bf16/layer_norm/strided/inner_strided_r129_c257` | normalization | 129 × 257 | strided | correct | [passed](operator_peer/metal__bf16__layer_norm__strided__inner_strided_r129_c257.json) | 0.5715 / 0.6009 | [passed](program_peer/metal__bf16__layer_norm__strided__inner_strided_r129_c257.json) | 0.6747 / 0.613 |
| `metal/bf16/layer_norm/strided/inner_offset_r2_c17` | normalization | 2 × 17 | offset | correct | [passed](operator_peer/metal__bf16__layer_norm__strided__inner_offset_r2_c17.json) | 0.2579 / 0.1776 | [passed](program_peer/metal__bf16__layer_norm__strided__inner_offset_r2_c17.json) | 0.2774 / 0.2237 |
| `metal/bf16/layer_norm/strided/inner_offset_r2_c63` | normalization | 2 × 63 | offset | correct | [passed](operator_peer/metal__bf16__layer_norm__strided__inner_offset_r2_c63.json) | 0.3662 / 0.1951 | [passed](program_peer/metal__bf16__layer_norm__strided__inner_offset_r2_c63.json) | 0.3425 / 0.1989 |
| `metal/bf16/layer_norm/strided/inner_offset_r2_c257` | normalization | 2 × 257 | offset | correct | [passed](operator_peer/metal__bf16__layer_norm__strided__inner_offset_r2_c257.json) | 0.2837 / 0.2261 | [passed](program_peer/metal__bf16__layer_norm__strided__inner_offset_r2_c257.json) | 0.3394 / 0.2049 |
| `metal/bf16/layer_norm/strided/inner_offset_r7_c17` | normalization | 7 × 17 | offset | correct | [passed](operator_peer/metal__bf16__layer_norm__strided__inner_offset_r7_c17.json) | 0.2645 / 0.2007 | [passed](program_peer/metal__bf16__layer_norm__strided__inner_offset_r7_c17.json) | 0.321 / 0.2257 |
| `metal/bf16/layer_norm/strided/inner_offset_r7_c63` | normalization | 7 × 63 | offset | correct | [passed](operator_peer/metal__bf16__layer_norm__strided__inner_offset_r7_c63.json) | 0.3357 / 0.2483 | [passed](program_peer/metal__bf16__layer_norm__strided__inner_offset_r7_c63.json) | 0.3482 / 0.1909 |
| `metal/bf16/layer_norm/strided/inner_offset_r7_c257` | normalization | 7 × 257 | offset | correct | [passed](operator_peer/metal__bf16__layer_norm__strided__inner_offset_r7_c257.json) | 0.2717 / 0.2186 | [passed](program_peer/metal__bf16__layer_norm__strided__inner_offset_r7_c257.json) | 0.2752 / 0.187 |
| `metal/bf16/layer_norm/strided/inner_offset_r64_c17` | normalization | 64 × 17 | offset | correct | [passed](operator_peer/metal__bf16__layer_norm__strided__inner_offset_r64_c17.json) | 0.2814 / 0.2028 | [passed](program_peer/metal__bf16__layer_norm__strided__inner_offset_r64_c17.json) | 0.3107 / 0.1986 |
| `metal/bf16/layer_norm/strided/inner_offset_r64_c63` | normalization | 64 × 63 | offset | correct | [passed](operator_peer/metal__bf16__layer_norm__strided__inner_offset_r64_c63.json) | 0.3154 / 0.2415 | [passed](program_peer/metal__bf16__layer_norm__strided__inner_offset_r64_c63.json) | 0.4462 / 0.2279 |
| `metal/bf16/layer_norm/strided/inner_offset_r64_c257` | normalization | 64 × 257 | offset | correct | [passed](operator_peer/metal__bf16__layer_norm__strided__inner_offset_r64_c257.json) | 0.3996 / 0.4246 | [passed](program_peer/metal__bf16__layer_norm__strided__inner_offset_r64_c257.json) | 0.4492 / 0.42 |
| `metal/bf16/layer_norm/strided/inner_offset_r129_c17` | normalization | 129 × 17 | offset | correct | [passed](operator_peer/metal__bf16__layer_norm__strided__inner_offset_r129_c17.json) | 0.3338 / 0.2271 | [passed](program_peer/metal__bf16__layer_norm__strided__inner_offset_r129_c17.json) | 0.3593 / 0.2249 |
| `metal/bf16/layer_norm/strided/inner_offset_r129_c63` | normalization | 129 × 63 | offset | correct | [passed](operator_peer/metal__bf16__layer_norm__strided__inner_offset_r129_c63.json) | 0.4391 / 0.2869 | [passed](program_peer/metal__bf16__layer_norm__strided__inner_offset_r129_c63.json) | 0.498 / 0.3134 |
| `metal/bf16/layer_norm/strided/inner_offset_r129_c257` | normalization | 129 × 257 | offset | correct | [passed](operator_peer/metal__bf16__layer_norm__strided__inner_offset_r129_c257.json) | 0.5739 / 0.6058 | [passed](program_peer/metal__bf16__layer_norm__strided__inner_offset_r129_c257.json) | 0.6952 / 0.5964 |
| `metal/bf16/layer_norm/outer_axis/outer_strided_r2_c17` | normalization | 2 × 17 | strided | correct | [passed](operator_peer/metal__bf16__layer_norm__outer_axis__outer_strided_r2_c17.json) | 0.516 / 0.1973 | [passed](program_peer/metal__bf16__layer_norm__outer_axis__outer_strided_r2_c17.json) | 0.5377 / 0.194 |
| `metal/bf16/layer_norm/outer_axis/outer_strided_r2_c257` | normalization | 2 × 257 | strided | correct | [passed](operator_peer/metal__bf16__layer_norm__outer_axis__outer_strided_r2_c257.json) | 0.2641 / 0.2032 | [passed](program_peer/metal__bf16__layer_norm__outer_axis__outer_strided_r2_c257.json) | 0.3064 / 0.2052 |
| `metal/bf16/layer_norm/outer_axis/outer_strided_r7_c63` | normalization | 7 × 63 | strided | correct | [passed](operator_peer/metal__bf16__layer_norm__outer_axis__outer_strided_r7_c63.json) | 0.2811 / 0.2059 | [passed](program_peer/metal__bf16__layer_norm__outer_axis__outer_strided_r7_c63.json) | 0.3524 / 0.2422 |
| `metal/bf16/layer_norm/outer_axis/outer_strided_r7_c127` | normalization | 7 × 127 | strided | correct | [passed](operator_peer/metal__bf16__layer_norm__outer_axis__outer_strided_r7_c127.json) | 0.3085 / 0.2014 | [passed](program_peer/metal__bf16__layer_norm__outer_axis__outer_strided_r7_c127.json) | 0.3348 / 0.24 |
| `metal/bf16/layer_norm/outer_axis/outer_offset_r2_c17` | normalization | 2 × 17 | offset | correct | [passed](operator_peer/metal__bf16__layer_norm__outer_axis__outer_offset_r2_c17.json) | 0.4916 / 0.1983 | [passed](program_peer/metal__bf16__layer_norm__outer_axis__outer_offset_r2_c17.json) | 0.5218 / 0.1966 |
| `metal/bf16/layer_norm/outer_axis/outer_offset_r2_c257` | normalization | 2 × 257 | offset | correct | [passed](operator_peer/metal__bf16__layer_norm__outer_axis__outer_offset_r2_c257.json) | 0.226 / 0.1935 | [passed](program_peer/metal__bf16__layer_norm__outer_axis__outer_offset_r2_c257.json) | 0.2407 / 0.1825 |
| `metal/bf16/layer_norm/outer_axis/outer_offset_r7_c63` | normalization | 7 × 63 | offset | correct | [passed](operator_peer/metal__bf16__layer_norm__outer_axis__outer_offset_r7_c63.json) | 0.3588 / 0.2036 | [passed](program_peer/metal__bf16__layer_norm__outer_axis__outer_offset_r7_c63.json) | 0.2978 / 0.2078 |
| `metal/bf16/layer_norm/outer_axis/outer_offset_r7_c127` | normalization | 7 × 127 | offset | correct | [passed](operator_peer/metal__bf16__layer_norm__outer_axis__outer_offset_r7_c127.json) | 0.2651 / 0.2418 | [passed](program_peer/metal__bf16__layer_norm__outer_axis__outer_offset_r7_c127.json) | 0.3534 / 0.2107 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
