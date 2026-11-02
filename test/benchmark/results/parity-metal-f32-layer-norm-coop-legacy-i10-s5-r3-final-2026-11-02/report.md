# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `7b60329ad5192420e3e88ccc94f25b3fc2ed69dcf9c0513b84ac3d451524527a`  
Binary SHA-256: `921ae28ce07860990ec9e503afd1d4472af2c309c647eb11d084cd7d329cdb62`  
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
| `metal/f32/layer_norm/short_row` | normalization | 4 × 32 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__short_row.json) | 0.2479 / 0.192 | [passed](program_peer/metal__f32__layer_norm__short_row.json) | 0.3016 / 0.1965 |
| `metal/f32/layer_norm/long_row` | normalization | 128 × 512 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__long_row.json) | 0.8581 / 0.8833 | [passed](program_peer/metal__f32__layer_norm__long_row.json) | 0.8669 / 0.9034 |
| `metal/f32/layer_norm/strided` | normalization | 16 × 64 | strided | correct | [passed](operator_peer/metal__f32__layer_norm__strided.json) | 0.2586 / 0.2057 | [passed](program_peer/metal__f32__layer_norm__strided.json) | 0.3201 / 0.2126 |
| `metal/f32/layer_norm/ragged` | normalization | 17 × 65 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__ragged.json) | 0.3756 / 0.2077 | [passed](program_peer/metal__f32__layer_norm__ragged.json) | 0.3505 / 0.2021 |
| `metal/f32/layer_norm/outer_axis` | normalization | 32 × 64 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__outer_axis.json) | 0.422 / 0.2165 | [passed](program_peer/metal__f32__layer_norm__outer_axis.json) | 0.2829 / 0.215 |
| `metal/f32/layer_norm/short_row/r32_c31` | normalization | 32 × 31 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__short_row__r32_c31.json) | 0.3198 / 0.2019 | [passed](program_peer/metal__f32__layer_norm__short_row__r32_c31.json) | 0.2731 / 0.1891 |
| `metal/f32/layer_norm/short_row/r32_c32` | normalization | 32 × 32 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__short_row__r32_c32.json) | 0.2371 / 0.2082 | [passed](program_peer/metal__f32__layer_norm__short_row__r32_c32.json) | 0.2552 / 0.1756 |
| `metal/f32/layer_norm/short_row/r32_c33` | normalization | 32 × 33 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__short_row__r32_c33.json) | 0.3232 / 0.1999 | [passed](program_peer/metal__f32__layer_norm__short_row__r32_c33.json) | 0.2728 / 0.1672 |
| `metal/f32/layer_norm/long_row/r32_c63` | normalization | 32 × 63 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__long_row__r32_c63.json) | 0.2489 / 0.2169 | [passed](program_peer/metal__f32__layer_norm__long_row__r32_c63.json) | 0.2398 / 0.1888 |
| `metal/f32/layer_norm/long_row/r32_c64` | normalization | 32 × 64 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__long_row__r32_c64.json) | 0.3583 / 0.2167 | [passed](program_peer/metal__f32__layer_norm__long_row__r32_c64.json) | 0.3346 / 0.214 |
| `metal/f32/layer_norm/long_row/r32_c65` | normalization | 32 × 65 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__long_row__r32_c65.json) | 0.3879 / 0.2051 | [passed](program_peer/metal__f32__layer_norm__long_row__r32_c65.json) | 0.3994 / 0.2192 |
| `metal/f32/layer_norm/long_row/r32_c127` | normalization | 32 × 127 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__long_row__r32_c127.json) | 0.302 / 0.2289 | [passed](program_peer/metal__f32__layer_norm__long_row__r32_c127.json) | 0.3054 / 0.2369 |
| `metal/f32/layer_norm/long_row/r32_c128` | normalization | 32 × 128 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__long_row__r32_c128.json) | 0.3023 / 0.2364 | [passed](program_peer/metal__f32__layer_norm__long_row__r32_c128.json) | 0.3059 / 0.2351 |
| `metal/f32/layer_norm/long_row/r32_c129` | normalization | 32 × 129 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__long_row__r32_c129.json) | 0.3041 / 0.2324 | [passed](program_peer/metal__f32__layer_norm__long_row__r32_c129.json) | 0.3116 / 0.2369 |
| `metal/f32/layer_norm/long_row/r256_c64` | normalization | 256 × 64 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__long_row__r256_c64.json) | 0.545 / 0.3872 | [passed](program_peer/metal__f32__layer_norm__long_row__r256_c64.json) | 0.5536 / 0.413 |
| `metal/f32/layer_norm/strided/r32_c127` | normalization | 32 × 127 | strided | correct | [passed](operator_peer/metal__f32__layer_norm__strided__r32_c127.json) | 0.2408 / 0.238 | [passed](program_peer/metal__f32__layer_norm__strided__r32_c127.json) | 0.3065 / 0.2394 |
| `metal/f32/layer_norm/strided/r64_c128` | normalization | 64 × 128 | strided | correct | [passed](operator_peer/metal__f32__layer_norm__strided__r64_c128.json) | 0.3459 / 0.2728 | [passed](program_peer/metal__f32__layer_norm__strided__r64_c128.json) | 0.3259 / 0.2763 |
| `metal/f32/layer_norm/ragged/r17_c65` | normalization | 17 × 65 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__ragged__r17_c65.json) | 0.2641 / 0.2053 | [passed](program_peer/metal__f32__layer_norm__ragged__r17_c65.json) | 0.2621 / 0.203 |
| `metal/f32/layer_norm/ragged/r33_c129` | normalization | 33 × 129 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__ragged__r33_c129.json) | 0.3035 / 0.2342 | [passed](program_peer/metal__f32__layer_norm__ragged__r33_c129.json) | 0.2931 / 0.2407 |
| `metal/f32/layer_norm/outer_axis/r127_c64` | normalization | 127 × 64 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__outer_axis__r127_c64.json) | 0.4621 / 0.2728 | [passed](program_peer/metal__f32__layer_norm__outer_axis__r127_c64.json) | 0.4671 / 0.2747 |
| `metal/f32/layer_norm/outer_axis/r64_c129` | normalization | 64 × 129 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__outer_axis__r64_c129.json) | 0.4813 / 0.2622 | [passed](program_peer/metal__f32__layer_norm__outer_axis__r64_c129.json) | 0.4223 / 0.275 |
| `metal/f32/layer_norm/long_row/r256_c31` | normalization | 256 × 31 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__long_row__r256_c31.json) | 0.4075 / 0.3037 | [passed](program_peer/metal__f32__layer_norm__long_row__r256_c31.json) | 0.432 / 0.2775 |
| `metal/f32/layer_norm/long_row/r128_c129` | normalization | 128 × 129 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__long_row__r128_c129.json) | 0.4523 / 0.3848 | [passed](program_peer/metal__f32__layer_norm__long_row__r128_c129.json) | 0.4551 / 0.4139 |
| `metal/f32/layer_norm/long_row/r33_c255` | normalization | 33 × 255 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__long_row__r33_c255.json) | 0.3359 / 0.2723 | [passed](program_peer/metal__f32__layer_norm__long_row__r33_c255.json) | 0.3397 / 0.2783 |
| `metal/f32/layer_norm/long_row/r64_c255` | normalization | 64 × 255 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__long_row__r64_c255.json) | 0.3796 / 0.3705 | [passed](program_peer/metal__f32__layer_norm__long_row__r64_c255.json) | 0.3859 / 0.4083 |
| `metal/f32/layer_norm/short_row/r31_c16` | normalization | 31 × 16 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__short_row__r31_c16.json) | 0.2551 / 0.184 | [passed](program_peer/metal__f32__layer_norm__short_row__r31_c16.json) | 0.2497 / 0.1964 |
| `metal/f32/layer_norm/short_row/r8_c7` | normalization | 8 × 7 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__short_row__r8_c7.json) | 0.2345 / 0.1804 | [passed](program_peer/metal__f32__layer_norm__short_row__r8_c7.json) | 0.2535 / 0.18 |
| `metal/f32/layer_norm/short_row/r8_c15` | normalization | 8 × 15 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__short_row__r8_c15.json) | 0.2639 / 0.19 | [passed](program_peer/metal__f32__layer_norm__short_row__r8_c15.json) | 0.2682 / 0.1888 |
| `metal/f32/layer_norm/short_row/r8_c16` | normalization | 8 × 16 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__short_row__r8_c16.json) | 0.2625 / 0.1907 | [passed](program_peer/metal__f32__layer_norm__short_row__r8_c16.json) | 0.2687 / 0.1906 |
| `metal/f32/layer_norm/short_row/r8_c17` | normalization | 8 × 17 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__short_row__r8_c17.json) | 0.2695 / 0.1903 | [passed](program_peer/metal__f32__layer_norm__short_row__r8_c17.json) | 0.2815 / 0.1908 |
| `metal/f32/layer_norm/short_row/r16_c31` | normalization | 16 × 31 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__short_row__r16_c31.json) | 0.2976 / 0.1956 | [passed](program_peer/metal__f32__layer_norm__short_row__r16_c31.json) | 0.3147 / 0.197 |
| `metal/f32/layer_norm/short_row/r16_c32` | normalization | 16 × 32 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__short_row__r16_c32.json) | 0.3087 / 0.2018 | [passed](program_peer/metal__f32__layer_norm__short_row__r16_c32.json) | 0.3138 / 0.1955 |
| `metal/f32/layer_norm/short_row/r16_c33` | normalization | 16 × 33 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__short_row__r16_c33.json) | 0.3187 / 0.1956 | [passed](program_peer/metal__f32__layer_norm__short_row__r16_c33.json) | 0.3176 / 0.2033 |
| `metal/f32/layer_norm/long_row/r32_c257` | normalization | 32 × 257 | dense | correct | [passed](operator_peer/metal__f32__layer_norm__long_row__r32_c257.json) | 0.3342 / 0.271 | [passed](program_peer/metal__f32__layer_norm__long_row__r32_c257.json) | 0.3425 / 0.2747 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
