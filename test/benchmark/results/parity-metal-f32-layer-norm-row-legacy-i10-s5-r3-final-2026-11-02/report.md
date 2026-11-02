# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `3a23a74a3cd7625ae99e8d375e6072c9126c4e1cc076285327dc24335a366da2`  
Binary SHA-256: `1263f72409ea8b8e7c7c5ea44ec4bc78639d602290b1885391b28c52332c724e`  
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
| `metal/f32/layer_norm/short_row` | normalization | 4 × 32 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__short_row.json) | 0.2397 / 0.1886 | [passed](program_peer/metal__f32__layer_norm__short_row.json) | 0.2689 / 0.1947 |
| `metal/f32/layer_norm/long_row` | normalization | 128 × 512 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__long_row.json) | 0.8025 / 0.8801 | [passed](program_peer/metal__f32__layer_norm__long_row.json) | 0.814 / 0.9209 |
| `metal/f32/layer_norm/strided` | normalization | 16 × 64 | strided | correct | [passed](operator_peer/metal__f32__layer_norm__strided.json) | 0.3169 / 0.2058 | [passed](program_peer/metal__f32__layer_norm__strided.json) | 0.2641 / 0.2053 |
| `metal/f32/layer_norm/ragged` | normalization | 17 × 65 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__ragged.json) | 0.2611 / 0.2014 | [passed](program_peer/metal__f32__layer_norm__ragged.json) | 0.3138 / 0.1987 |
| `metal/f32/layer_norm/outer_axis` | normalization | 32 × 64 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__outer_axis.json) | 0.274 / 0.2159 | [passed](program_peer/metal__f32__layer_norm__outer_axis.json) | 0.3443 / 0.217 |
| `metal/f32/layer_norm/short_row/r32_c31` | normalization | 32 × 31 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__short_row__r32_c31.json) | 0.3201 / 0.1931 | [passed](program_peer/metal__f32__layer_norm__short_row__r32_c31.json) | 0.3164 / 0.1984 |
| `metal/f32/layer_norm/short_row/r32_c32` | normalization | 32 × 32 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__short_row__r32_c32.json) | 0.3167 / 0.1994 | [passed](program_peer/metal__f32__layer_norm__short_row__r32_c32.json) | 0.3208 / 0.2061 |
| `metal/f32/layer_norm/short_row/r32_c33` | normalization | 32 × 33 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__short_row__r32_c33.json) | 0.3121 / 0.2048 | [passed](program_peer/metal__f32__layer_norm__short_row__r32_c33.json) | 0.3241 / 0.2031 |
| `metal/f32/layer_norm/long_row/r32_c63` | normalization | 32 × 63 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__long_row__r32_c63.json) | 0.312 / 0.2169 | [passed](program_peer/metal__f32__layer_norm__long_row__r32_c63.json) | 0.322 / 0.2187 |
| `metal/f32/layer_norm/long_row/r32_c64` | normalization | 32 × 64 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__long_row__r32_c64.json) | 0.3355 / 0.2131 | [passed](program_peer/metal__f32__layer_norm__long_row__r32_c64.json) | 0.3144 / 0.2185 |
| `metal/f32/layer_norm/long_row/r32_c65` | normalization | 32 × 65 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__long_row__r32_c65.json) | 0.278 / 0.2178 | [passed](program_peer/metal__f32__layer_norm__long_row__r32_c65.json) | 0.2796 / 0.2193 |
| `metal/f32/layer_norm/long_row/r32_c127` | normalization | 32 × 127 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__long_row__r32_c127.json) | 0.28 / 0.2325 | [passed](program_peer/metal__f32__layer_norm__long_row__r32_c127.json) | 0.3422 / 0.2355 |
| `metal/f32/layer_norm/long_row/r32_c128` | normalization | 32 × 128 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__long_row__r32_c128.json) | 0.3209 / 0.2357 | [passed](program_peer/metal__f32__layer_norm__long_row__r32_c128.json) | 0.3074 / 0.2365 |
| `metal/f32/layer_norm/long_row/r32_c129` | normalization | 32 × 129 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__long_row__r32_c129.json) | 0.3341 / 0.2326 | [passed](program_peer/metal__f32__layer_norm__long_row__r32_c129.json) | 0.3406 / 0.2364 |
| `metal/f32/layer_norm/long_row/r256_c64` | normalization | 256 × 64 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__long_row__r256_c64.json) | 0.4725 / 0.3928 | [passed](program_peer/metal__f32__layer_norm__long_row__r256_c64.json) | 0.5309 / 0.4141 |
| `metal/f32/layer_norm/strided/r32_c127` | normalization | 32 × 127 | strided | correct | [passed](operator_peer/metal__f32__layer_norm__strided__r32_c127.json) | 0.3381 / 0.2379 | [passed](program_peer/metal__f32__layer_norm__strided__r32_c127.json) | 0.3427 / 0.2412 |
| `metal/f32/layer_norm/strided/r64_c128` | normalization | 64 × 128 | strided | correct | [passed](operator_peer/metal__f32__layer_norm__strided__r64_c128.json) | 0.4008 / 0.2755 | [passed](program_peer/metal__f32__layer_norm__strided__r64_c128.json) | 0.3952 / 0.2752 |
| `metal/f32/layer_norm/ragged/r17_c65` | normalization | 17 × 65 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__ragged__r17_c65.json) | 0.3421 / 0.2026 | [passed](program_peer/metal__f32__layer_norm__ragged__r17_c65.json) | 0.3363 / 0.1988 |
| `metal/f32/layer_norm/ragged/r33_c129` | normalization | 33 × 129 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__ragged__r33_c129.json) | 0.3436 / 0.2356 | [passed](program_peer/metal__f32__layer_norm__ragged__r33_c129.json) | 0.3435 / 0.2397 |
| `metal/f32/layer_norm/outer_axis/r127_c64` | normalization | 127 × 64 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__outer_axis__r127_c64.json) | 0.4526 / 0.2739 | [passed](program_peer/metal__f32__layer_norm__outer_axis__r127_c64.json) | 0.4638 / 0.2766 |
| `metal/f32/layer_norm/outer_axis/r64_c129` | normalization | 64 × 129 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__outer_axis__r64_c129.json) | 0.424 / 0.277 | [passed](program_peer/metal__f32__layer_norm__outer_axis__r64_c129.json) | 0.4207 / 0.2749 |
| `metal/f32/layer_norm/long_row/r256_c31` | normalization | 256 × 31 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__long_row__r256_c31.json) | 0.416 / 0.2725 | [passed](program_peer/metal__f32__layer_norm__long_row__r256_c31.json) | 0.4509 / 0.278 |
| `metal/f32/layer_norm/long_row/r128_c129` | normalization | 128 × 129 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__long_row__r128_c129.json) | 0.4187 / 0.3848 | [passed](program_peer/metal__f32__layer_norm__long_row__r128_c129.json) | 0.4421 / 0.3976 |
| `metal/f32/layer_norm/long_row/r33_c255` | normalization | 33 × 255 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__long_row__r33_c255.json) | 0.397 / 0.2716 | [passed](program_peer/metal__f32__layer_norm__long_row__r33_c255.json) | 0.3729 / 0.2748 |
| `metal/f32/layer_norm/long_row/r64_c255` | normalization | 64 × 255 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__long_row__r64_c255.json) | 0.4948 / 0.4018 | [passed](program_peer/metal__f32__layer_norm__long_row__r64_c255.json) | 0.4028 / 0.3924 |
| `metal/f32/layer_norm/short_row/r31_c16` | normalization | 31 × 16 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__short_row__r31_c16.json) | 0.2872 / 0.1973 | [passed](program_peer/metal__f32__layer_norm__short_row__r31_c16.json) | 0.2778 / 0.2011 |
| `metal/f32/layer_norm/short_row/r8_c7` | normalization | 8 × 7 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__short_row__r8_c7.json) | 0.2545 / 0.1869 | [passed](program_peer/metal__f32__layer_norm__short_row__r8_c7.json) | 0.2598 / 0.1954 |
| `metal/f32/layer_norm/short_row/r8_c15` | normalization | 8 × 15 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__short_row__r8_c15.json) | 0.2647 / 0.1924 | [passed](program_peer/metal__f32__layer_norm__short_row__r8_c15.json) | 0.2562 / 0.1854 |
| `metal/f32/layer_norm/short_row/r8_c16` | normalization | 8 × 16 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__short_row__r8_c16.json) | 0.2644 / 0.1891 | [passed](program_peer/metal__f32__layer_norm__short_row__r8_c16.json) | 0.2692 / 0.19 |
| `metal/f32/layer_norm/short_row/r8_c17` | normalization | 8 × 17 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__short_row__r8_c17.json) | 0.2709 / 0.1883 | [passed](program_peer/metal__f32__layer_norm__short_row__r8_c17.json) | 0.2805 / 0.1949 |
| `metal/f32/layer_norm/short_row/r16_c31` | normalization | 16 × 31 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__short_row__r16_c31.json) | 0.314 / 0.1944 | [passed](program_peer/metal__f32__layer_norm__short_row__r16_c31.json) | 0.3154 / 0.1949 |
| `metal/f32/layer_norm/short_row/r16_c32` | normalization | 16 × 32 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__short_row__r16_c32.json) | 0.311 / 0.1997 | [passed](program_peer/metal__f32__layer_norm__short_row__r16_c32.json) | 0.3019 / 0.1993 |
| `metal/f32/layer_norm/short_row/r16_c33` | normalization | 16 × 33 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__short_row__r16_c33.json) | 0.3147 / 0.1933 | [passed](program_peer/metal__f32__layer_norm__short_row__r16_c33.json) | 0.3182 / 0.1952 |
| `metal/f32/layer_norm/long_row/r32_c257` | normalization | 32 × 257 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__long_row__r32_c257.json) | 0.3521 / 0.2739 | [passed](program_peer/metal__f32__layer_norm__long_row__r32_c257.json) | 0.3944 / 0.2774 |
| `metal/f32/layer_norm/strided/inner_strided_r2_c127` | normalization | 2 × 127 | strided | correct | [passed](operator_peer/metal__f32__layer_norm__strided__inner_strided_r2_c127.json) | 0.2783 / 0.193 | [passed](program_peer/metal__f32__layer_norm__strided__inner_strided_r2_c127.json) | 0.2998 / 0.1936 |
| `metal/f32/layer_norm/strided/inner_strided_r7_c127` | normalization | 7 × 127 | strided | correct | [passed](operator_peer/metal__f32__layer_norm__strided__inner_strided_r7_c127.json) | 0.297 / 0.203 | [passed](program_peer/metal__f32__layer_norm__strided__inner_strided_r7_c127.json) | 0.2467 / 0.2037 |
| `metal/f32/layer_norm/strided/inner_strided_r64_c127` | normalization | 64 × 127 | strided | correct | [passed](operator_peer/metal__f32__layer_norm__strided__inner_strided_r64_c127.json) | 0.4037 / 0.277 | [passed](program_peer/metal__f32__layer_norm__strided__inner_strided_r64_c127.json) | 0.3904 / 0.2765 |
| `metal/f32/layer_norm/strided/inner_strided_r129_c127` | normalization | 129 × 127 | strided | correct | [passed](operator_peer/metal__f32__layer_norm__strided__inner_strided_r129_c127.json) | 0.5038 / 0.4253 | [passed](program_peer/metal__f32__layer_norm__strided__inner_strided_r129_c127.json) | 0.4277 / 0.4263 |
| `metal/f32/layer_norm/strided/inner_offset_r2_c127` | normalization | 2 × 127 | offset | correct | [passed](operator_peer/metal__f32__layer_norm__strided__inner_offset_r2_c127.json) | 0.297 / 0.1956 | [passed](program_peer/metal__f32__layer_norm__strided__inner_offset_r2_c127.json) | 0.2898 / 0.1961 |
| `metal/f32/layer_norm/strided/inner_offset_r7_c127` | normalization | 7 × 127 | offset | correct | [passed](operator_peer/metal__f32__layer_norm__strided__inner_offset_r7_c127.json) | 0.2661 / 0.2043 | [passed](program_peer/metal__f32__layer_norm__strided__inner_offset_r7_c127.json) | 0.2211 / 0.2036 |
| `metal/f32/layer_norm/strided/inner_offset_r64_c127` | normalization | 64 × 127 | offset | correct | [passed](operator_peer/metal__f32__layer_norm__strided__inner_offset_r64_c127.json) | 0.3617 / 0.2736 | [passed](program_peer/metal__f32__layer_norm__strided__inner_offset_r64_c127.json) | 0.3811 / 0.2779 |
| `metal/f32/layer_norm/strided/inner_offset_r129_c127` | normalization | 129 × 127 | offset | correct | [passed](operator_peer/metal__f32__layer_norm__strided__inner_offset_r129_c127.json) | 0.4328 / 0.4286 | [passed](program_peer/metal__f32__layer_norm__strided__inner_offset_r129_c127.json) | 0.4194 / 0.4312 |
| `metal/f32/layer_norm/outer_axis/outer_strided_r2_c63` | normalization | 2 × 63 | strided | correct | [passed](operator_peer/metal__f32__layer_norm__outer_axis__outer_strided_r2_c63.json) | 0.2943 / 0.1978 | [passed](program_peer/metal__f32__layer_norm__outer_axis__outer_strided_r2_c63.json) | 0.2897 / 0.1944 |
| `metal/f32/layer_norm/outer_axis/outer_strided_r2_c127` | normalization | 2 × 127 | strided | correct | [passed](operator_peer/metal__f32__layer_norm__outer_axis__outer_strided_r2_c127.json) | 0.2804 / 0.1942 | [passed](program_peer/metal__f32__layer_norm__outer_axis__outer_strided_r2_c127.json) | 0.2946 / 0.1936 |
| `metal/f32/layer_norm/outer_axis/outer_strided_r7_c17` | normalization | 7 × 17 | strided | correct | [passed](operator_peer/metal__f32__layer_norm__outer_axis__outer_strided_r7_c17.json) | 0.2533 / 0.1854 | [passed](program_peer/metal__f32__layer_norm__outer_axis__outer_strided_r7_c17.json) | 0.2928 / 0.1995 |
| `metal/f32/layer_norm/outer_axis/outer_offset_r2_c63` | normalization | 2 × 63 | offset | correct | [passed](operator_peer/metal__f32__layer_norm__outer_axis__outer_offset_r2_c63.json) | 0.292 / 0.1954 | [passed](program_peer/metal__f32__layer_norm__outer_axis__outer_offset_r2_c63.json) | 0.2893 / 0.1962 |
| `metal/f32/layer_norm/outer_axis/outer_offset_r2_c127` | normalization | 2 × 127 | offset | correct | [passed](operator_peer/metal__f32__layer_norm__outer_axis__outer_offset_r2_c127.json) | 0.2873 / 0.1908 | [passed](program_peer/metal__f32__layer_norm__outer_axis__outer_offset_r2_c127.json) | 0.2861 / 0.1923 |
| `metal/f32/layer_norm/outer_axis/outer_offset_r7_c17` | normalization | 7 × 17 | offset | correct | [passed](operator_peer/metal__f32__layer_norm__outer_axis__outer_offset_r7_c17.json) | 0.2365 / 0.1972 | [passed](program_peer/metal__f32__layer_norm__outer_axis__outer_offset_r7_c17.json) | 0.2458 / 0.1976 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
