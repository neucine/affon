# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `57f414f20c89518ff964c9b6717d7fe35b06ed4c10210607a33a1c014566c7d2`  
Binary SHA-256: `94e3c1795a7cbcae6e6202b022ce83d5f4d5a69f03d81ab1e9de230af9c7bc4e`  
Timing: 5 iterations × 3 samples × 2 rounds

## Checks

- Cases run: 28
- Numerical check passed: 28
- Operator check and timing passed: 28
- Program check and timing passed: 28
- Numerical failures: 0
- Operator failures: 0
- Program failures: 0

## Cases

| Case | Family | Shape | Layout | Numerical | Operator | Affon / PyTorch ms | Program | Affon / PyTorch ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `metal/bf16/layer_norm/strided/inner_strided_r2_c257` | normalization | 2 × 257 | strided | correct | [passed](operator_peer/metal__bf16__layer_norm__strided__inner_strided_r2_c257.json) | 0.2655 / 0.2373 | [passed](program_peer/metal__bf16__layer_norm__strided__inner_strided_r2_c257.json) | 0.3529 / 0.1911 |
| `metal/bf16/layer_norm/strided/inner_strided_r7_c257` | normalization | 7 × 257 | strided | correct | [passed](operator_peer/metal__bf16__layer_norm__strided__inner_strided_r7_c257.json) | 0.2976 / 0.2189 | [passed](program_peer/metal__bf16__layer_norm__strided__inner_strided_r7_c257.json) | 0.2868 / 0.2194 |
| `metal/bf16/layer_norm/strided/inner_strided_r64_c257` | normalization | 64 × 257 | strided | correct | [passed](operator_peer/metal__bf16__layer_norm__strided__inner_strided_r64_c257.json) | 0.4788 / 0.411 | [passed](program_peer/metal__bf16__layer_norm__strided__inner_strided_r64_c257.json) | 0.5451 / 0.4171 |
| `metal/bf16/layer_norm/strided/inner_strided_r129_c257` | normalization | 129 × 257 | strided | correct | [passed](operator_peer/metal__bf16__layer_norm__strided__inner_strided_r129_c257.json) | 0.7278 / 0.5832 | [passed](program_peer/metal__bf16__layer_norm__strided__inner_strided_r129_c257.json) | 0.8202 / 0.5991 |
| `metal/bf16/layer_norm/strided/inner_offset_r2_c257` | normalization | 2 × 257 | offset | correct | [passed](operator_peer/metal__bf16__layer_norm__strided__inner_offset_r2_c257.json) | 0.2647 / 0.192 | [passed](program_peer/metal__bf16__layer_norm__strided__inner_offset_r2_c257.json) | 0.2623 / 0.195 |
| `metal/bf16/layer_norm/strided/inner_offset_r7_c257` | normalization | 7 × 257 | offset | correct | [passed](operator_peer/metal__bf16__layer_norm__strided__inner_offset_r7_c257.json) | 0.2967 / 0.2157 | [passed](program_peer/metal__bf16__layer_norm__strided__inner_offset_r7_c257.json) | 0.3095 / 0.2138 |
| `metal/bf16/layer_norm/strided/inner_offset_r64_c257` | normalization | 64 × 257 | offset | correct | [passed](operator_peer/metal__bf16__layer_norm__strided__inner_offset_r64_c257.json) | 0.3984 / 0.3836 | [passed](program_peer/metal__bf16__layer_norm__strided__inner_offset_r64_c257.json) | 0.458 / 0.4091 |
| `metal/bf16/layer_norm/strided/inner_offset_r129_c257` | normalization | 129 × 257 | offset | correct | [passed](operator_peer/metal__bf16__layer_norm__strided__inner_offset_r129_c257.json) | 0.723 / 0.5775 | [passed](program_peer/metal__bf16__layer_norm__strided__inner_offset_r129_c257.json) | 0.8588 / 0.5741 |
| `metal/bf16/layer_norm/outer_axis/outer_strided_r2_c257` | normalization | 2 × 257 | strided | correct | [passed](operator_peer/metal__bf16__layer_norm__outer_axis__outer_strided_r2_c257.json) | 0.2643 / 0.2357 | [passed](program_peer/metal__bf16__layer_norm__outer_axis__outer_strided_r2_c257.json) | 0.3017 / 0.2012 |
| `metal/bf16/layer_norm/outer_axis/outer_strided_r7_c63` | normalization | 7 × 63 | strided | correct | [passed](operator_peer/metal__bf16__layer_norm__outer_axis__outer_strided_r7_c63.json) | 0.3242 / 0.1915 | [passed](program_peer/metal__bf16__layer_norm__outer_axis__outer_strided_r7_c63.json) | 0.3498 / 0.2073 |
| `metal/bf16/layer_norm/outer_axis/outer_strided_r7_c127` | normalization | 7 × 127 | strided | correct | [passed](operator_peer/metal__bf16__layer_norm__outer_axis__outer_strided_r7_c127.json) | 0.3537 / 0.2079 | [passed](program_peer/metal__bf16__layer_norm__outer_axis__outer_strided_r7_c127.json) | 0.3851 / 0.2122 |
| `metal/bf16/layer_norm/outer_axis/outer_offset_r2_c257` | normalization | 2 × 257 | offset | correct | [passed](operator_peer/metal__bf16__layer_norm__outer_axis__outer_offset_r2_c257.json) | 0.2981 / 0.1898 | [passed](program_peer/metal__bf16__layer_norm__outer_axis__outer_offset_r2_c257.json) | 0.311 / 0.1914 |
| `metal/bf16/layer_norm/outer_axis/outer_offset_r7_c63` | normalization | 7 × 63 | offset | correct | [passed](operator_peer/metal__bf16__layer_norm__outer_axis__outer_offset_r7_c63.json) | 0.2983 / 0.225 | [passed](program_peer/metal__bf16__layer_norm__outer_axis__outer_offset_r7_c63.json) | 0.264 / 0.203 |
| `metal/bf16/layer_norm/outer_axis/outer_offset_r7_c127` | normalization | 7 × 127 | offset | correct | [passed](operator_peer/metal__bf16__layer_norm__outer_axis__outer_offset_r7_c127.json) | 0.3507 / 0.2138 | [passed](program_peer/metal__bf16__layer_norm__outer_axis__outer_offset_r7_c127.json) | 0.2555 / 0.2138 |
| `metal/bf16/rms_norm/strided/strided_r2_c257` | normalization | 2 × 257 | strided | correct | [passed](operator_peer/metal__bf16__rms_norm__strided__strided_r2_c257.json) | 0.245 / 0.2916 | [passed](program_peer/metal__bf16__rms_norm__strided__strided_r2_c257.json) | 0.2868 / 0.25 |
| `metal/bf16/rms_norm/strided/strided_r7_c257` | normalization | 7 × 257 | strided | correct | [passed](operator_peer/metal__bf16__rms_norm__strided__strided_r7_c257.json) | 0.273 / 0.2614 | [passed](program_peer/metal__bf16__rms_norm__strided__strided_r7_c257.json) | 0.3113 / 0.292 |
| `metal/bf16/rms_norm/strided/strided_r64_c257` | normalization | 64 × 257 | strided | correct | [passed](operator_peer/metal__bf16__rms_norm__strided__strided_r64_c257.json) | 0.4786 / 0.5381 | [passed](program_peer/metal__bf16__rms_norm__strided__strided_r64_c257.json) | 0.533 / 0.5286 |
| `metal/bf16/rms_norm/strided/strided_r129_c257` | normalization | 129 × 257 | strided | correct | [passed](operator_peer/metal__bf16__rms_norm__strided__strided_r129_c257.json) | 0.7153 / 0.7687 | [passed](program_peer/metal__bf16__rms_norm__strided__strided_r129_c257.json) | 0.7511 / 0.7669 |
| `metal/bf16/rms_norm/long_row/offset_r2_c257` | normalization | 2 × 257 | offset | correct | [passed](operator_peer/metal__bf16__rms_norm__long_row__offset_r2_c257.json) | 0.2341 / 0.2729 | [passed](program_peer/metal__bf16__rms_norm__long_row__offset_r2_c257.json) | 0.2953 / 0.2696 |
| `metal/bf16/rms_norm/long_row/offset_r7_c257` | normalization | 7 × 257 | offset | correct | [passed](operator_peer/metal__bf16__rms_norm__long_row__offset_r7_c257.json) | 0.281 / 0.288 | [passed](program_peer/metal__bf16__rms_norm__long_row__offset_r7_c257.json) | 0.3266 / 0.2892 |
| `metal/bf16/rms_norm/long_row/offset_r64_c257` | normalization | 64 × 257 | offset | correct | [passed](operator_peer/metal__bf16__rms_norm__long_row__offset_r64_c257.json) | 0.4876 / 0.538 | [passed](program_peer/metal__bf16__rms_norm__long_row__offset_r64_c257.json) | 0.5256 / 0.5491 |
| `metal/bf16/rms_norm/long_row/offset_r129_c257` | normalization | 129 × 257 | offset | correct | [passed](operator_peer/metal__bf16__rms_norm__long_row__offset_r129_c257.json) | 0.7201 / 0.7461 | [passed](program_peer/metal__bf16__rms_norm__long_row__offset_r129_c257.json) | 0.725 / 0.7342 |
| `metal/bf16/rms_norm/outer_axis/strided_r2_c257` | normalization | 2 × 257 | strided | correct | [passed](operator_peer/metal__bf16__rms_norm__outer_axis__strided_r2_c257.json) | 0.2767 / 0.2619 | [passed](program_peer/metal__bf16__rms_norm__outer_axis__strided_r2_c257.json) | 0.2891 / 0.2674 |
| `metal/bf16/rms_norm/outer_axis/strided_r7_c63` | normalization | 7 × 63 | strided | correct | [passed](operator_peer/metal__bf16__rms_norm__outer_axis__strided_r7_c63.json) | 0.2838 / 0.2579 | [passed](program_peer/metal__bf16__rms_norm__outer_axis__strided_r7_c63.json) | 0.3401 / 0.3077 |
| `metal/bf16/rms_norm/outer_axis/strided_r7_c127` | normalization | 7 × 127 | strided | correct | [passed](operator_peer/metal__bf16__rms_norm__outer_axis__strided_r7_c127.json) | 0.3238 / 0.2586 | [passed](program_peer/metal__bf16__rms_norm__outer_axis__strided_r7_c127.json) | 0.333 / 0.3001 |
| `metal/bf16/rms_norm/outer_axis/offset_r2_c257` | normalization | 2 × 257 | offset | correct | [passed](operator_peer/metal__bf16__rms_norm__outer_axis__offset_r2_c257.json) | 0.3139 / 0.2605 | [passed](program_peer/metal__bf16__rms_norm__outer_axis__offset_r2_c257.json) | 0.3645 / 0.2674 |
| `metal/bf16/rms_norm/outer_axis/offset_r7_c63` | normalization | 7 × 63 | offset | correct | [passed](operator_peer/metal__bf16__rms_norm__outer_axis__offset_r7_c63.json) | 0.2341 / 0.2796 | [passed](program_peer/metal__bf16__rms_norm__outer_axis__offset_r7_c63.json) | 0.313 / 0.256 |
| `metal/bf16/rms_norm/outer_axis/offset_r7_c127` | normalization | 7 × 127 | offset | correct | [passed](operator_peer/metal__bf16__rms_norm__outer_axis__offset_r7_c127.json) | 0.3153 / 0.2698 | [passed](program_peer/metal__bf16__rms_norm__outer_axis__offset_r7_c127.json) | 0.3582 / 0.2505 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
