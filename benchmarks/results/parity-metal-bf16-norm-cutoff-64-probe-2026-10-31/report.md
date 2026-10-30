# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `afe5f9afd66b4a3b7ef86f538a08648719f8322374731cf751a65842581f2dfe`  
Binary SHA-256: `420e5315d554f3c9d08baf19a6f5517b0471f8b2a96377cacc0a6d30be0eda16`  
Timing: 10 iterations × 5 samples × 3 rounds

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
| `metal/bf16/rms_norm/outer_axis` | normalization | 32 × 64 | dense | correct | [passed](operator_peer/metal__bf16__rms_norm__outer_axis.json) | 0.3766 / 0.2889 | [passed](program_peer/metal__bf16__rms_norm__outer_axis.json) | 0.2835 / 0.2938 |
| `metal/bf16/layer_norm/outer_axis` | normalization | 32 × 64 | dense | correct | [passed](operator_peer/metal__bf16__layer_norm__outer_axis.json) | 0.3466 / 0.2075 | [passed](program_peer/metal__bf16__layer_norm__outer_axis.json) | 0.3218 / 0.2199 |
| `metal/bf16/rms_norm/strided/r64_c128` | normalization | 64 × 128 | strided | correct | [passed](operator_peer/metal__bf16__rms_norm__strided__r64_c128.json) | 0.3367 / 0.3419 | [passed](program_peer/metal__bf16__rms_norm__strided__r64_c128.json) | 0.367 / 0.3492 |
| `metal/bf16/rms_norm/outer_axis/r64_c129` | normalization | 64 × 129 | dense | correct | [passed](operator_peer/metal__bf16__rms_norm__outer_axis__r64_c129.json) | 0.3905 / 0.361 | [passed](program_peer/metal__bf16__rms_norm__outer_axis__r64_c129.json) | 0.3877 / 0.3749 |
| `metal/bf16/rms_norm/long_row/r128_c129` | normalization | 128 × 129 | dense | correct | [passed](operator_peer/metal__bf16__rms_norm__long_row__r128_c129.json) | 0.4187 / 0.5141 | [passed](program_peer/metal__bf16__rms_norm__long_row__r128_c129.json) | 0.4143 / 0.4548 |
| `metal/bf16/log_softmax/long_row/r128_c129` | normalization | 128 × 129 | dense | correct | [passed](operator_peer/metal__bf16__log_softmax__long_row__r128_c129.json) | 0.4004 / 0.466 | [passed](program_peer/metal__bf16__log_softmax__long_row__r128_c129.json) | 0.5283 / 0.4877 |
| `metal/f32/rms_norm/long_row/r128_c129` | normalization | 128 × 129 | dense | correct | [passed](operator_peer/metal__f32__rms_norm__long_row__r128_c129.json) | 0.4996 / 0.509 | [passed](program_peer/metal__f32__rms_norm__long_row__r128_c129.json) | 0.4312 / 0.4425 |
| `metal/bf16/rms_norm/ragged/r33_c129` | normalization | 33 × 129 | dense | correct | [passed](operator_peer/metal__bf16__rms_norm__ragged__r33_c129.json) | 0.3131 / 0.3126 | [passed](program_peer/metal__bf16__rms_norm__ragged__r33_c129.json) | 0.3465 / 0.3154 |
| `metal/f32/rms_norm/outer_axis/r127_c64` | normalization | 127 × 64 | dense | correct | [passed](operator_peer/metal__f32__rms_norm__outer_axis__r127_c64.json) | 0.464 / 0.3308 | [passed](program_peer/metal__f32__rms_norm__outer_axis__r127_c64.json) | 0.4647 / 0.3347 |
| `metal/f32/log_softmax/long_row/r256_c64` | normalization | 256 × 64 | dense | correct | [passed](operator_peer/metal__f32__log_softmax__long_row__r256_c64.json) | 0.4517 / 0.4609 | [passed](program_peer/metal__f32__log_softmax__long_row__r256_c64.json) | 0.4675 / 0.4587 |
| `metal/f32/log_softmax/long_row/r64_c255` | normalization | 64 × 255 | dense | correct | [passed](operator_peer/metal__f32__log_softmax__long_row__r64_c255.json) | 0.3901 / 0.4577 | [passed](program_peer/metal__f32__log_softmax__long_row__r64_c255.json) | 0.3946 / 0.441 |
| `metal/bf16/rms_norm/long_row/r32_c129` | normalization | 32 × 129 | dense | correct | [passed](operator_peer/metal__bf16__rms_norm__long_row__r32_c129.json) | 0.2938 / 0.3063 | [passed](program_peer/metal__bf16__rms_norm__long_row__r32_c129.json) | 0.287 / 0.3159 |
| `metal/bf16/rms_norm/long_row/r32_c127` | normalization | 32 × 127 | dense | correct | [passed](operator_peer/metal__bf16__rms_norm__long_row__r32_c127.json) | 0.3113 / 0.3126 | [passed](program_peer/metal__bf16__rms_norm__long_row__r32_c127.json) | 0.2632 / 0.2964 |
| `metal/bf16/log_softmax/long_row/r64_c255` | normalization | 64 × 255 | dense | correct | [passed](operator_peer/metal__bf16__log_softmax__long_row__r64_c255.json) | 0.3996 / 0.504 | [passed](program_peer/metal__bf16__log_softmax__long_row__r64_c255.json) | 0.43 / 0.505 |
| `metal/bf16/rms_norm/long_row/r32_c128` | normalization | 32 × 128 | dense | correct | [passed](operator_peer/metal__bf16__rms_norm__long_row__r32_c128.json) | 0.313 / 0.3059 | [passed](program_peer/metal__bf16__rms_norm__long_row__r32_c128.json) | 0.2736 / 0.3126 |
| `metal/f32/rms_norm/long_row/r64_c255` | normalization | 64 × 255 | dense | correct | [passed](operator_peer/metal__f32__rms_norm__long_row__r64_c255.json) | 0.3974 / 0.4881 | [passed](program_peer/metal__f32__rms_norm__long_row__r64_c255.json) | 0.4241 / 0.4844 |
| `metal/bf16/rms_norm/outer_axis/r127_c64` | normalization | 127 × 64 | dense | correct | [passed](operator_peer/metal__bf16__rms_norm__outer_axis__r127_c64.json) | 0.4151 / 0.3626 | [passed](program_peer/metal__bf16__rms_norm__outer_axis__r127_c64.json) | 0.4212 / 0.3667 |
| `metal/bf16/rms_norm/strided/r32_c127` | normalization | 32 × 127 | strided | correct | [passed](operator_peer/metal__bf16__rms_norm__strided__r32_c127.json) | 0.3293 / 0.3056 | [passed](program_peer/metal__bf16__rms_norm__strided__r32_c127.json) | 0.3462 / 0.3106 |
| `metal/f32/rms_norm/long_row/r256_c31` | normalization | 256 × 31 | dense | correct | [passed](operator_peer/metal__f32__rms_norm__long_row__r256_c31.json) | 0.4603 / 0.3143 | [passed](program_peer/metal__f32__rms_norm__long_row__r256_c31.json) | 0.4671 / 0.3196 |
| `metal/bf16/layer_norm/long_row/r32_c257` | normalization | 32 × 257 | dense | correct | [passed](operator_peer/metal__bf16__layer_norm__long_row__r32_c257.json) | 0.3275 / 0.2736 | [passed](program_peer/metal__bf16__layer_norm__long_row__r32_c257.json) | 0.3472 / 0.2748 |
| `metal/bf16/rms_norm/long_row/r33_c255` | normalization | 33 × 255 | dense | correct | [passed](operator_peer/metal__bf16__rms_norm__long_row__r33_c255.json) | 0.3018 / 0.3523 | [passed](program_peer/metal__bf16__rms_norm__long_row__r33_c255.json) | 0.388 / 0.3642 |
| `metal/bf16/log_softmax/long_row/r256_c31` | normalization | 256 × 31 | dense | correct | [passed](operator_peer/metal__bf16__log_softmax__long_row__r256_c31.json) | 0.4262 / 0.3305 | [passed](program_peer/metal__bf16__log_softmax__long_row__r256_c31.json) | 0.4253 / 0.3363 |
| `metal/bf16/rms_norm/long_row/r32_c257` | normalization | 32 × 257 | dense | correct | [passed](operator_peer/metal__bf16__rms_norm__long_row__r32_c257.json) | 0.3494 / 0.3567 | [passed](program_peer/metal__bf16__rms_norm__long_row__r32_c257.json) | 0.3592 / 0.3753 |
| `metal/bf16/rms_norm/long_row/r256_c31` | normalization | 256 × 31 | dense | correct | [passed](operator_peer/metal__bf16__rms_norm__long_row__r256_c31.json) | 0.4667 / 0.3533 | [passed](program_peer/metal__bf16__rms_norm__long_row__r256_c31.json) | 0.4678 / 0.3582 |
| `metal/f32/log_softmax/long_row/r256_c31` | normalization | 256 × 31 | dense | correct | [passed](operator_peer/metal__f32__log_softmax__long_row__r256_c31.json) | 0.5063 / 0.2992 | [passed](program_peer/metal__f32__log_softmax__long_row__r256_c31.json) | 0.4959 / 0.301 |
| `metal/bf16/layer_norm/strided/r64_c128` | normalization | 64 × 128 | strided | correct | [passed](operator_peer/metal__bf16__layer_norm__strided__r64_c128.json) | 0.3658 / 0.2768 | [passed](program_peer/metal__bf16__layer_norm__strided__r64_c128.json) | 0.3718 / 0.2752 |
| `metal/f32/rms_norm/long_row/r33_c255` | normalization | 33 × 255 | dense | correct | [passed](operator_peer/metal__f32__rms_norm__long_row__r33_c255.json) | 0.3798 / 0.3165 | [passed](program_peer/metal__f32__rms_norm__long_row__r33_c255.json) | 0.4023 / 0.3254 |
| `metal/bf16/log_softmax/strided/r64_c128` | normalization | 64 × 128 | strided | correct | [passed](operator_peer/metal__bf16__log_softmax__strided__r64_c128.json) | 0.3173 / 0.3202 | [passed](program_peer/metal__bf16__log_softmax__strided__r64_c128.json) | 0.3181 / 0.3077 |
| `metal/bf16/log_softmax/outer_axis/r64_c129` | normalization | 64 × 129 | dense | correct | [passed](operator_peer/metal__bf16__log_softmax__outer_axis__r64_c129.json) | 0.3723 / 0.3345 | [passed](program_peer/metal__bf16__log_softmax__outer_axis__r64_c129.json) | 0.4033 / 0.336 |
| `metal/bf16/log_softmax/long_row/r33_c255` | normalization | 33 × 255 | dense | correct | [passed](operator_peer/metal__bf16__log_softmax__long_row__r33_c255.json) | 0.336 / 0.3385 | [passed](program_peer/metal__bf16__log_softmax__long_row__r33_c255.json) | 0.3242 / 0.3384 |
| `metal/f32/log_softmax/strided/r64_c128` | normalization | 64 × 128 | strided | correct | [passed](operator_peer/metal__f32__log_softmax__strided__r64_c128.json) | 0.3525 / 0.3029 | [passed](program_peer/metal__f32__log_softmax__strided__r64_c128.json) | 0.3703 / 0.3017 |
| `metal/f32/layer_norm/strided/r64_c128` | normalization | 64 × 128 | strided | correct | [passed](operator_peer/metal__f32__layer_norm__strided__r64_c128.json) | 0.3924 / 0.2694 | [passed](program_peer/metal__f32__layer_norm__strided__r64_c128.json) | 0.3678 / 0.2795 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
