# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `870a9f7e3b670e3110505e3cf676bf463953ad97cdeaac5a31ceb9fab1eb2a55`  
Binary SHA-256: `07d2686a6fde2c671da3d04668ac145d8ca91e78bd6eb55675701480f9735945`  
Timing: 10 iterations × 5 samples × 3 rounds

## Checks

- Cases run: 48
- Numerical check passed: 48
- Operator check and timing passed: 48
- Program check and timing passed: 48
- Numerical failures: 0
- Operator failures: 0
- Program failures: 0

## Cases

| Case | Family | Shape | Layout | Numerical | Operator | Affon / PyTorch ms | Program | Affon / PyTorch ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `metal/bf16/layer_norm/short_row` | normalization | 4 × 32 | dense | correct | [passed](operator_peer/metal__bf16__layer_norm__short_row.json) | 0.2855 / 0.1852 | [passed](program_peer/metal__bf16__layer_norm__short_row.json) | 0.3114 / 0.1911 |
| `metal/bf16/layer_norm/long_row` | normalization | 128 × 512 | dense | correct | [passed](operator_peer/metal__bf16__layer_norm__long_row.json) | 0.8005 / 0.8971 | [passed](program_peer/metal__bf16__layer_norm__long_row.json) | 0.8293 / 0.8971 |
| `metal/bf16/layer_norm/strided` | normalization | 16 × 64 | strided | correct | [passed](operator_peer/metal__bf16__layer_norm__strided.json) | 0.2625 / 0.2068 | [passed](program_peer/metal__bf16__layer_norm__strided.json) | 0.2967 / 0.2077 |
| `metal/bf16/layer_norm/ragged` | normalization | 17 × 65 | dense | correct | [passed](operator_peer/metal__bf16__layer_norm__ragged.json) | 0.269 / 0.202 | [passed](program_peer/metal__bf16__layer_norm__ragged.json) | 0.2889 / 0.2041 |
| `metal/bf16/layer_norm/outer_axis` | normalization | 32 × 64 | dense | correct | [passed](operator_peer/metal__bf16__layer_norm__outer_axis.json) | 0.3452 / 0.2066 | [passed](program_peer/metal__bf16__layer_norm__outer_axis.json) | 0.4346 / 0.2168 |
| `metal/bf16/layer_norm/short_row/r32_c31` | normalization | 32 × 31 | dense | correct | [passed](operator_peer/metal__bf16__layer_norm__short_row__r32_c31.json) | 0.3115 / 0.1996 | [passed](program_peer/metal__bf16__layer_norm__short_row__r32_c31.json) | 0.3277 / 0.2009 |
| `metal/bf16/layer_norm/short_row/r32_c32` | normalization | 32 × 32 | dense | correct | [passed](operator_peer/metal__bf16__layer_norm__short_row__r32_c32.json) | 0.312 / 0.2015 | [passed](program_peer/metal__bf16__layer_norm__short_row__r32_c32.json) | 0.3345 / 0.1999 |
| `metal/bf16/layer_norm/short_row/r32_c33` | normalization | 32 × 33 | dense | correct | [passed](operator_peer/metal__bf16__layer_norm__short_row__r32_c33.json) | 0.3105 / 0.2066 | [passed](program_peer/metal__bf16__layer_norm__short_row__r32_c33.json) | 0.2741 / 0.2034 |
| `metal/bf16/layer_norm/long_row/r32_c63` | normalization | 32 × 63 | dense | correct | [passed](operator_peer/metal__bf16__layer_norm__long_row__r32_c63.json) | 0.3715 / 0.2169 | [passed](program_peer/metal__bf16__layer_norm__long_row__r32_c63.json) | 0.3032 / 0.219 |
| `metal/bf16/layer_norm/long_row/r32_c64` | normalization | 32 × 64 | dense | correct | [passed](operator_peer/metal__bf16__layer_norm__long_row__r32_c64.json) | 0.279 / 0.2167 | [passed](program_peer/metal__bf16__layer_norm__long_row__r32_c64.json) | 0.2797 / 0.2193 |
| `metal/bf16/layer_norm/long_row/r32_c65` | normalization | 32 × 65 | dense | correct | [passed](operator_peer/metal__bf16__layer_norm__long_row__r32_c65.json) | 0.2817 / 0.2166 | [passed](program_peer/metal__bf16__layer_norm__long_row__r32_c65.json) | 0.3062 / 0.214 |
| `metal/bf16/layer_norm/long_row/r32_c127` | normalization | 32 × 127 | dense | correct | [passed](operator_peer/metal__bf16__layer_norm__long_row__r32_c127.json) | 0.2884 / 0.2369 | [passed](program_peer/metal__bf16__layer_norm__long_row__r32_c127.json) | 0.3201 / 0.2384 |
| `metal/bf16/layer_norm/long_row/r32_c128` | normalization | 32 × 128 | dense | correct | [passed](operator_peer/metal__bf16__layer_norm__long_row__r32_c128.json) | 0.297 / 0.2337 | [passed](program_peer/metal__bf16__layer_norm__long_row__r32_c128.json) | 0.3181 / 0.2332 |
| `metal/bf16/layer_norm/long_row/r32_c129` | normalization | 32 × 129 | dense | correct | [passed](operator_peer/metal__bf16__layer_norm__long_row__r32_c129.json) | 0.2956 / 0.2372 | [passed](program_peer/metal__bf16__layer_norm__long_row__r32_c129.json) | 0.3165 / 0.2397 |
| `metal/bf16/layer_norm/long_row/r256_c64` | normalization | 256 × 64 | dense | correct | [passed](operator_peer/metal__bf16__layer_norm__long_row__r256_c64.json) | 0.511 / 0.3922 | [passed](program_peer/metal__bf16__layer_norm__long_row__r256_c64.json) | 0.4658 / 0.4137 |
| `metal/bf16/layer_norm/strided/r32_c127` | normalization | 32 × 127 | strided | correct | [passed](operator_peer/metal__bf16__layer_norm__strided__r32_c127.json) | 0.2948 / 0.2327 | [passed](program_peer/metal__bf16__layer_norm__strided__r32_c127.json) | 0.3235 / 0.2402 |
| `metal/bf16/layer_norm/strided/r64_c128` | normalization | 64 × 128 | strided | correct | [passed](operator_peer/metal__bf16__layer_norm__strided__r64_c128.json) | 0.3256 / 0.2769 | [passed](program_peer/metal__bf16__layer_norm__strided__r64_c128.json) | 0.3053 / 0.2772 |
| `metal/bf16/layer_norm/ragged/r17_c65` | normalization | 17 × 65 | dense | correct | [passed](operator_peer/metal__bf16__layer_norm__ragged__r17_c65.json) | 0.2643 / 0.2008 | [passed](program_peer/metal__bf16__layer_norm__ragged__r17_c65.json) | 0.2878 / 0.2054 |
| `metal/bf16/layer_norm/ragged/r33_c129` | normalization | 33 × 129 | dense | correct | [passed](operator_peer/metal__bf16__layer_norm__ragged__r33_c129.json) | 0.2949 / 0.2392 | [passed](program_peer/metal__bf16__layer_norm__ragged__r33_c129.json) | 0.2799 / 0.2382 |
| `metal/bf16/layer_norm/outer_axis/r127_c64` | normalization | 127 × 64 | dense | correct | [passed](operator_peer/metal__bf16__layer_norm__outer_axis__r127_c64.json) | 0.4149 / 0.2748 | [passed](program_peer/metal__bf16__layer_norm__outer_axis__r127_c64.json) | 0.4206 / 0.2651 |
| `metal/bf16/layer_norm/outer_axis/r64_c129` | normalization | 64 × 129 | dense | correct | [passed](operator_peer/metal__bf16__layer_norm__outer_axis__r64_c129.json) | 0.382 / 0.2773 | [passed](program_peer/metal__bf16__layer_norm__outer_axis__r64_c129.json) | 0.3869 / 0.2759 |
| `metal/bf16/layer_norm/long_row/r256_c31` | normalization | 256 × 31 | dense | correct | [passed](operator_peer/metal__bf16__layer_norm__long_row__r256_c31.json) | 0.4177 / 0.2801 | [passed](program_peer/metal__bf16__layer_norm__long_row__r256_c31.json) | 0.4663 / 0.2758 |
| `metal/bf16/layer_norm/long_row/r128_c129` | normalization | 128 × 129 | dense | correct | [passed](operator_peer/metal__bf16__layer_norm__long_row__r128_c129.json) | 0.3917 / 0.3902 | [passed](program_peer/metal__bf16__layer_norm__long_row__r128_c129.json) | 0.4266 / 0.418 |
| `metal/bf16/layer_norm/long_row/r33_c255` | normalization | 33 × 255 | dense | correct | [passed](operator_peer/metal__bf16__layer_norm__long_row__r33_c255.json) | 0.322 / 0.2798 | [passed](program_peer/metal__bf16__layer_norm__long_row__r33_c255.json) | 0.3495 / 0.271 |
| `metal/bf16/layer_norm/long_row/r64_c255` | normalization | 64 × 255 | dense | correct | [passed](operator_peer/metal__bf16__layer_norm__long_row__r64_c255.json) | 0.3991 / 0.4016 | [passed](program_peer/metal__bf16__layer_norm__long_row__r64_c255.json) | 0.4086 / 0.4136 |
| `metal/bf16/layer_norm/short_row/r31_c16` | normalization | 31 × 16 | dense | correct | [passed](operator_peer/metal__bf16__layer_norm__short_row__r31_c16.json) | 0.2741 / 0.2006 | [passed](program_peer/metal__bf16__layer_norm__short_row__r31_c16.json) | 0.2998 / 0.1922 |
| `metal/bf16/layer_norm/short_row/r8_c7` | normalization | 8 × 7 | dense | correct | [passed](operator_peer/metal__bf16__layer_norm__short_row__r8_c7.json) | 0.2523 / 0.185 | [passed](program_peer/metal__bf16__layer_norm__short_row__r8_c7.json) | 0.2697 / 0.1949 |
| `metal/bf16/layer_norm/short_row/r8_c15` | normalization | 8 × 15 | dense | correct | [passed](operator_peer/metal__bf16__layer_norm__short_row__r8_c15.json) | 0.2583 / 0.1833 | [passed](program_peer/metal__bf16__layer_norm__short_row__r8_c15.json) | 0.2797 / 0.1905 |
| `metal/bf16/layer_norm/short_row/r8_c16` | normalization | 8 × 16 | dense | correct | [passed](operator_peer/metal__bf16__layer_norm__short_row__r8_c16.json) | 0.2643 / 0.1922 | [passed](program_peer/metal__bf16__layer_norm__short_row__r8_c16.json) | 0.2837 / 0.1893 |
| `metal/bf16/layer_norm/short_row/r8_c17` | normalization | 8 × 17 | dense | correct | [passed](operator_peer/metal__bf16__layer_norm__short_row__r8_c17.json) | 0.2259 / 0.1964 | [passed](program_peer/metal__bf16__layer_norm__short_row__r8_c17.json) | 0.2409 / 0.1852 |
| `metal/bf16/layer_norm/short_row/r16_c31` | normalization | 16 × 31 | dense | correct | [passed](operator_peer/metal__bf16__layer_norm__short_row__r16_c31.json) | 0.2901 / 0.1954 | [passed](program_peer/metal__bf16__layer_norm__short_row__r16_c31.json) | 0.3047 / 0.1924 |
| `metal/bf16/layer_norm/short_row/r16_c32` | normalization | 16 × 32 | dense | correct | [passed](operator_peer/metal__bf16__layer_norm__short_row__r16_c32.json) | 0.2691 / 0.1969 | [passed](program_peer/metal__bf16__layer_norm__short_row__r16_c32.json) | 0.3083 / 0.1975 |
| `metal/bf16/layer_norm/short_row/r16_c33` | normalization | 16 × 33 | dense | correct | [passed](operator_peer/metal__bf16__layer_norm__short_row__r16_c33.json) | 0.3073 / 0.1981 | [passed](program_peer/metal__bf16__layer_norm__short_row__r16_c33.json) | 0.2455 / 0.1967 |
| `metal/bf16/layer_norm/long_row/r32_c257` | normalization | 32 × 257 | dense | correct | [passed](operator_peer/metal__bf16__layer_norm__long_row__r32_c257.json) | 0.3264 / 0.2741 | [passed](program_peer/metal__bf16__layer_norm__long_row__r32_c257.json) | 0.3377 / 0.2783 |
| `metal/bf16/layer_norm/strided/inner_strided_r2_c127` | normalization | 2 × 127 | strided | correct | [passed](operator_peer/metal__bf16__layer_norm__strided__inner_strided_r2_c127.json) | 0.2506 / 0.1943 | [passed](program_peer/metal__bf16__layer_norm__strided__inner_strided_r2_c127.json) | 0.2636 / 0.1966 |
| `metal/bf16/layer_norm/strided/inner_strided_r7_c127` | normalization | 7 × 127 | strided | correct | [passed](operator_peer/metal__bf16__layer_norm__strided__inner_strided_r7_c127.json) | 0.2604 / 0.1991 | [passed](program_peer/metal__bf16__layer_norm__strided__inner_strided_r7_c127.json) | 0.2835 / 0.2054 |
| `metal/bf16/layer_norm/strided/inner_strided_r64_c127` | normalization | 64 × 127 | strided | correct | [passed](operator_peer/metal__bf16__layer_norm__strided__inner_strided_r64_c127.json) | 0.3432 / 0.2759 | [passed](program_peer/metal__bf16__layer_norm__strided__inner_strided_r64_c127.json) | 0.3185 / 0.2755 |
| `metal/bf16/layer_norm/strided/inner_strided_r129_c127` | normalization | 129 × 127 | strided | correct | [passed](operator_peer/metal__bf16__layer_norm__strided__inner_strided_r129_c127.json) | 0.4402 / 0.4254 | [passed](program_peer/metal__bf16__layer_norm__strided__inner_strided_r129_c127.json) | 0.4746 / 0.431 |
| `metal/bf16/layer_norm/strided/inner_offset_r2_c127` | normalization | 2 × 127 | offset | correct | [passed](operator_peer/metal__bf16__layer_norm__strided__inner_offset_r2_c127.json) | 0.2462 / 0.1834 | [passed](program_peer/metal__bf16__layer_norm__strided__inner_offset_r2_c127.json) | 0.2594 / 0.1979 |
| `metal/bf16/layer_norm/strided/inner_offset_r7_c127` | normalization | 7 × 127 | offset | correct | [passed](operator_peer/metal__bf16__layer_norm__strided__inner_offset_r7_c127.json) | 0.2599 / 0.2093 | [passed](program_peer/metal__bf16__layer_norm__strided__inner_offset_r7_c127.json) | 0.2783 / 0.2084 |
| `metal/bf16/layer_norm/strided/inner_offset_r64_c127` | normalization | 64 × 127 | offset | correct | [passed](operator_peer/metal__bf16__layer_norm__strided__inner_offset_r64_c127.json) | 0.3005 / 0.2758 | [passed](program_peer/metal__bf16__layer_norm__strided__inner_offset_r64_c127.json) | 0.3041 / 0.2797 |
| `metal/bf16/layer_norm/strided/inner_offset_r129_c127` | normalization | 129 × 127 | offset | correct | [passed](operator_peer/metal__bf16__layer_norm__strided__inner_offset_r129_c127.json) | 0.4266 / 0.4293 | [passed](program_peer/metal__bf16__layer_norm__strided__inner_offset_r129_c127.json) | 0.4648 / 0.4225 |
| `metal/bf16/layer_norm/outer_axis/outer_strided_r2_c63` | normalization | 2 × 63 | strided | correct | [passed](operator_peer/metal__bf16__layer_norm__outer_axis__outer_strided_r2_c63.json) | 0.2497 / 0.1851 | [passed](program_peer/metal__bf16__layer_norm__outer_axis__outer_strided_r2_c63.json) | 0.2672 / 0.1986 |
| `metal/bf16/layer_norm/outer_axis/outer_strided_r2_c127` | normalization | 2 × 127 | strided | correct | [passed](operator_peer/metal__bf16__layer_norm__outer_axis__outer_strided_r2_c127.json) | 0.2156 / 0.1959 | [passed](program_peer/metal__bf16__layer_norm__outer_axis__outer_strided_r2_c127.json) | 0.2817 / 0.1965 |
| `metal/bf16/layer_norm/outer_axis/outer_strided_r7_c17` | normalization | 7 × 17 | strided | correct | [passed](operator_peer/metal__bf16__layer_norm__outer_axis__outer_strided_r7_c17.json) | 0.2646 / 0.1979 | [passed](program_peer/metal__bf16__layer_norm__outer_axis__outer_strided_r7_c17.json) | 0.2864 / 0.1964 |
| `metal/bf16/layer_norm/outer_axis/outer_offset_r2_c63` | normalization | 2 × 63 | offset | correct | [passed](operator_peer/metal__bf16__layer_norm__outer_axis__outer_offset_r2_c63.json) | 0.2588 / 0.1947 | [passed](program_peer/metal__bf16__layer_norm__outer_axis__outer_offset_r2_c63.json) | 0.2327 / 0.1972 |
| `metal/bf16/layer_norm/outer_axis/outer_offset_r2_c127` | normalization | 2 × 127 | offset | correct | [passed](operator_peer/metal__bf16__layer_norm__outer_axis__outer_offset_r2_c127.json) | 0.2602 / 0.1944 | [passed](program_peer/metal__bf16__layer_norm__outer_axis__outer_offset_r2_c127.json) | 0.2653 / 0.2014 |
| `metal/bf16/layer_norm/outer_axis/outer_offset_r7_c17` | normalization | 7 × 17 | offset | correct | [passed](operator_peer/metal__bf16__layer_norm__outer_axis__outer_offset_r7_c17.json) | 0.262 / 0.2003 | [passed](program_peer/metal__bf16__layer_norm__outer_axis__outer_offset_r7_c17.json) | 0.2894 / 0.1906 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
