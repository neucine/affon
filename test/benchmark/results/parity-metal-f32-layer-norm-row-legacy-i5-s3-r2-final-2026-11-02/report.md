# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `9172ae82e66f41efb72a199559989a65fd08a467d653945c5fdae1cef689e6a3`  
Binary SHA-256: `1263f72409ea8b8e7c7c5ea44ec4bc78639d602290b1885391b28c52332c724e`  
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
| `metal/f32/layer_norm/strided/inner_strided_r2_c17` | normalization | 2 × 17 | strided | correct | [passed](operator_peer/metal__f32__layer_norm__strided__inner_strided_r2_c17.json) | 0.3364 / 0.2072 | [passed](program_peer/metal__f32__layer_norm__strided__inner_strided_r2_c17.json) | 0.279 / 0.2187 |
| `metal/f32/layer_norm/strided/inner_strided_r2_c63` | normalization | 2 × 63 | strided | correct | [passed](operator_peer/metal__f32__layer_norm__strided__inner_strided_r2_c63.json) | 0.2929 / 0.2147 | [passed](program_peer/metal__f32__layer_norm__strided__inner_strided_r2_c63.json) | 0.3371 / 0.2338 |
| `metal/f32/layer_norm/strided/inner_strided_r2_c257` | normalization | 2 × 257 | strided | correct | [passed](operator_peer/metal__f32__layer_norm__strided__inner_strided_r2_c257.json) | 0.2361 / 0.1997 | [passed](program_peer/metal__f32__layer_norm__strided__inner_strided_r2_c257.json) | 0.3063 / 0.2015 |
| `metal/f32/layer_norm/strided/inner_strided_r7_c17` | normalization | 7 × 17 | strided | correct | [passed](operator_peer/metal__f32__layer_norm__strided__inner_strided_r7_c17.json) | 0.2948 / 0.1916 | [passed](program_peer/metal__f32__layer_norm__strided__inner_strided_r7_c17.json) | 0.248 / 0.2061 |
| `metal/f32/layer_norm/strided/inner_strided_r7_c63` | normalization | 7 × 63 | strided | correct | [passed](operator_peer/metal__f32__layer_norm__strided__inner_strided_r7_c63.json) | 0.3173 / 0.203 | [passed](program_peer/metal__f32__layer_norm__strided__inner_strided_r7_c63.json) | 0.2596 / 0.2167 |
| `metal/f32/layer_norm/strided/inner_strided_r7_c257` | normalization | 7 × 257 | strided | correct | [passed](operator_peer/metal__f32__layer_norm__strided__inner_strided_r7_c257.json) | 0.2607 / 0.2213 | [passed](program_peer/metal__f32__layer_norm__strided__inner_strided_r7_c257.json) | 0.289 / 0.2475 |
| `metal/f32/layer_norm/strided/inner_strided_r64_c17` | normalization | 64 × 17 | strided | correct | [passed](operator_peer/metal__f32__layer_norm__strided__inner_strided_r64_c17.json) | 0.3229 / 0.2118 | [passed](program_peer/metal__f32__layer_norm__strided__inner_strided_r64_c17.json) | 0.2624 / 0.2129 |
| `metal/f32/layer_norm/strided/inner_strided_r64_c63` | normalization | 64 × 63 | strided | correct | [passed](operator_peer/metal__f32__layer_norm__strided__inner_strided_r64_c63.json) | 0.325 / 0.2422 | [passed](program_peer/metal__f32__layer_norm__strided__inner_strided_r64_c63.json) | 0.3405 / 0.2391 |
| `metal/f32/layer_norm/strided/inner_strided_r64_c257` | normalization | 64 × 257 | strided | correct | [passed](operator_peer/metal__f32__layer_norm__strided__inner_strided_r64_c257.json) | 0.5368 / 0.4186 | [passed](program_peer/metal__f32__layer_norm__strided__inner_strided_r64_c257.json) | 0.5056 / 0.4244 |
| `metal/f32/layer_norm/strided/inner_strided_r129_c17` | normalization | 129 × 17 | strided | correct | [passed](operator_peer/metal__f32__layer_norm__strided__inner_strided_r129_c17.json) | 0.3367 / 0.2239 | [passed](program_peer/metal__f32__layer_norm__strided__inner_strided_r129_c17.json) | 0.3371 / 0.2536 |
| `metal/f32/layer_norm/strided/inner_strided_r129_c63` | normalization | 129 × 63 | strided | correct | [passed](operator_peer/metal__f32__layer_norm__strided__inner_strided_r129_c63.json) | 0.3965 / 0.2798 | [passed](program_peer/metal__f32__layer_norm__strided__inner_strided_r129_c63.json) | 0.4441 / 0.2813 |
| `metal/f32/layer_norm/strided/inner_strided_r129_c257` | normalization | 129 × 257 | strided | correct | [passed](operator_peer/metal__f32__layer_norm__strided__inner_strided_r129_c257.json) | 0.7432 / 0.5933 | [passed](program_peer/metal__f32__layer_norm__strided__inner_strided_r129_c257.json) | 0.8482 / 0.6083 |
| `metal/f32/layer_norm/strided/inner_offset_r2_c17` | normalization | 2 × 17 | offset | correct | [passed](operator_peer/metal__f32__layer_norm__strided__inner_offset_r2_c17.json) | 0.2606 / 0.229 | [passed](program_peer/metal__f32__layer_norm__strided__inner_offset_r2_c17.json) | 0.3082 / 0.189 |
| `metal/f32/layer_norm/strided/inner_offset_r2_c63` | normalization | 2 × 63 | offset | correct | [passed](operator_peer/metal__f32__layer_norm__strided__inner_offset_r2_c63.json) | 0.3192 / 0.2098 | [passed](program_peer/metal__f32__layer_norm__strided__inner_offset_r2_c63.json) | 0.2707 / 0.1911 |
| `metal/f32/layer_norm/strided/inner_offset_r2_c257` | normalization | 2 × 257 | offset | correct | [passed](operator_peer/metal__f32__layer_norm__strided__inner_offset_r2_c257.json) | 0.3245 / 0.1866 | [passed](program_peer/metal__f32__layer_norm__strided__inner_offset_r2_c257.json) | 0.3241 / 0.1914 |
| `metal/f32/layer_norm/strided/inner_offset_r7_c17` | normalization | 7 × 17 | offset | correct | [passed](operator_peer/metal__f32__layer_norm__strided__inner_offset_r7_c17.json) | 0.272 / 0.1974 | [passed](program_peer/metal__f32__layer_norm__strided__inner_offset_r7_c17.json) | 0.2657 / 0.2044 |
| `metal/f32/layer_norm/strided/inner_offset_r7_c63` | normalization | 7 × 63 | offset | correct | [passed](operator_peer/metal__f32__layer_norm__strided__inner_offset_r7_c63.json) | 0.3048 / 0.2081 | [passed](program_peer/metal__f32__layer_norm__strided__inner_offset_r7_c63.json) | 0.3372 / 0.2359 |
| `metal/f32/layer_norm/strided/inner_offset_r7_c257` | normalization | 7 × 257 | offset | correct | [passed](operator_peer/metal__f32__layer_norm__strided__inner_offset_r7_c257.json) | 0.2826 / 0.206 | [passed](program_peer/metal__f32__layer_norm__strided__inner_offset_r7_c257.json) | 0.3152 / 0.2199 |
| `metal/f32/layer_norm/strided/inner_offset_r64_c17` | normalization | 64 × 17 | offset | correct | [passed](operator_peer/metal__f32__layer_norm__strided__inner_offset_r64_c17.json) | 0.3096 / 0.2532 | [passed](program_peer/metal__f32__layer_norm__strided__inner_offset_r64_c17.json) | 0.2875 / 0.197 |
| `metal/f32/layer_norm/strided/inner_offset_r64_c63` | normalization | 64 × 63 | offset | correct | [passed](operator_peer/metal__f32__layer_norm__strided__inner_offset_r64_c63.json) | 0.2936 / 0.2412 | [passed](program_peer/metal__f32__layer_norm__strided__inner_offset_r64_c63.json) | 0.355 / 0.249 |
| `metal/f32/layer_norm/strided/inner_offset_r64_c257` | normalization | 64 × 257 | offset | correct | [passed](operator_peer/metal__f32__layer_norm__strided__inner_offset_r64_c257.json) | 0.4525 / 0.3967 | [passed](program_peer/metal__f32__layer_norm__strided__inner_offset_r64_c257.json) | 0.5153 / 0.4226 |
| `metal/f32/layer_norm/strided/inner_offset_r129_c17` | normalization | 129 × 17 | offset | correct | [passed](operator_peer/metal__f32__layer_norm__strided__inner_offset_r129_c17.json) | 0.3284 / 0.2283 | [passed](program_peer/metal__f32__layer_norm__strided__inner_offset_r129_c17.json) | 0.3362 / 0.2266 |
| `metal/f32/layer_norm/strided/inner_offset_r129_c63` | normalization | 129 × 63 | offset | correct | [passed](operator_peer/metal__f32__layer_norm__strided__inner_offset_r129_c63.json) | 0.4306 / 0.2786 | [passed](program_peer/metal__f32__layer_norm__strided__inner_offset_r129_c63.json) | 0.4313 / 0.2819 |
| `metal/f32/layer_norm/strided/inner_offset_r129_c257` | normalization | 129 × 257 | offset | correct | [passed](operator_peer/metal__f32__layer_norm__strided__inner_offset_r129_c257.json) | 0.5831 / 0.568 | [passed](program_peer/metal__f32__layer_norm__strided__inner_offset_r129_c257.json) | 0.748 / 0.6139 |
| `metal/f32/layer_norm/outer_axis/outer_strided_r2_c17` | normalization | 2 × 17 | strided | correct | [passed](operator_peer/metal__f32__layer_norm__outer_axis__outer_strided_r2_c17.json) | 0.3334 / 0.2084 | [passed](program_peer/metal__f32__layer_norm__outer_axis__outer_strided_r2_c17.json) | 0.2725 / 0.1976 |
| `metal/f32/layer_norm/outer_axis/outer_strided_r2_c257` | normalization | 2 × 257 | strided | correct | [passed](operator_peer/metal__f32__layer_norm__outer_axis__outer_strided_r2_c257.json) | 0.3131 / 0.1888 | [passed](program_peer/metal__f32__layer_norm__outer_axis__outer_strided_r2_c257.json) | 0.2622 / 0.2021 |
| `metal/f32/layer_norm/outer_axis/outer_strided_r7_c63` | normalization | 7 × 63 | strided | correct | [passed](operator_peer/metal__f32__layer_norm__outer_axis__outer_strided_r7_c63.json) | 0.3201 / 0.1991 | [passed](program_peer/metal__f32__layer_norm__outer_axis__outer_strided_r7_c63.json) | 0.3539 / 0.2145 |
| `metal/f32/layer_norm/outer_axis/outer_strided_r7_c127` | normalization | 7 × 127 | strided | correct | [passed](operator_peer/metal__f32__layer_norm__outer_axis__outer_strided_r7_c127.json) | 0.3396 / 0.212 | [passed](program_peer/metal__f32__layer_norm__outer_axis__outer_strided_r7_c127.json) | 0.2546 / 0.2221 |
| `metal/f32/layer_norm/outer_axis/outer_offset_r2_c17` | normalization | 2 × 17 | offset | correct | [passed](operator_peer/metal__f32__layer_norm__outer_axis__outer_offset_r2_c17.json) | 0.4098 / 0.2046 | [passed](program_peer/metal__f32__layer_norm__outer_axis__outer_offset_r2_c17.json) | 0.4099 / 0.1841 |
| `metal/f32/layer_norm/outer_axis/outer_offset_r2_c257` | normalization | 2 × 257 | offset | correct | [passed](operator_peer/metal__f32__layer_norm__outer_axis__outer_offset_r2_c257.json) | 0.2421 / 0.2047 | [passed](program_peer/metal__f32__layer_norm__outer_axis__outer_offset_r2_c257.json) | 0.3139 / 0.2005 |
| `metal/f32/layer_norm/outer_axis/outer_offset_r7_c63` | normalization | 7 × 63 | offset | correct | [passed](operator_peer/metal__f32__layer_norm__outer_axis__outer_offset_r7_c63.json) | 0.3548 / 0.2013 | [passed](program_peer/metal__f32__layer_norm__outer_axis__outer_offset_r7_c63.json) | 0.3141 / 0.243 |
| `metal/f32/layer_norm/outer_axis/outer_offset_r7_c127` | normalization | 7 × 127 | offset | correct | [passed](operator_peer/metal__f32__layer_norm__outer_axis__outer_offset_r7_c127.json) | 0.3357 / 0.1931 | [passed](program_peer/metal__f32__layer_norm__outer_axis__outer_offset_r7_c127.json) | 0.3364 / 0.2057 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
