# Parity batch report

Target: `Mac16,1/26.6.2/arm64`  
Catalog SHA-256: `afe5f9afd66b4a3b7ef86f538a08648719f8322374731cf751a65842581f2dfe`  
Binary SHA-256: `68869bbfc6e3e488ddede2df99d960be78880a24d273eea6f9bc10ba171d4c47`  
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
| `metal/bf16/rms_norm/outer_axis` | normalization | 32 × 64 | dense | correct | [passed](operator_peer/metal__bf16__rms_norm__outer_axis.json) | 0.3707 / 0.2915 | [passed](program_peer/metal__bf16__rms_norm__outer_axis.json) | 0.4212 / 0.2919 |
| `metal/bf16/layer_norm/outer_axis` | normalization | 32 × 64 | dense | correct | [passed](operator_peer/metal__bf16__layer_norm__outer_axis.json) | 0.3527 / 0.2174 | [passed](program_peer/metal__bf16__layer_norm__outer_axis.json) | 0.2753 / 0.2204 |
| `metal/bf16/rms_norm/strided/r64_c128` | normalization | 64 × 128 | strided | correct | [passed](operator_peer/metal__bf16__rms_norm__strided__r64_c128.json) | 0.3304 / 0.3526 | [passed](program_peer/metal__bf16__rms_norm__strided__r64_c128.json) | 0.3309 / 0.3508 |
| `metal/bf16/rms_norm/outer_axis/r64_c129` | normalization | 64 × 129 | dense | correct | [passed](operator_peer/metal__bf16__rms_norm__outer_axis__r64_c129.json) | 0.3892 / 0.3727 | [passed](program_peer/metal__bf16__rms_norm__outer_axis__r64_c129.json) | 0.385 / 0.365 |
| `metal/bf16/rms_norm/long_row/r128_c129` | normalization | 128 × 129 | dense | correct | [passed](operator_peer/metal__bf16__rms_norm__long_row__r128_c129.json) | 0.4715 / 0.5127 | [passed](program_peer/metal__bf16__rms_norm__long_row__r128_c129.json) | 0.4376 / 0.494 |
| `metal/bf16/log_softmax/long_row/r128_c129` | normalization | 128 × 129 | dense | correct | [passed](operator_peer/metal__bf16__log_softmax__long_row__r128_c129.json) | 0.4724 / 0.4945 | [passed](program_peer/metal__bf16__log_softmax__long_row__r128_c129.json) | 0.4291 / 0.5147 |
| `metal/f32/rms_norm/long_row/r128_c129` | normalization | 128 × 129 | dense | correct | [passed](operator_peer/metal__f32__rms_norm__long_row__r128_c129.json) | 0.4375 / 0.5068 | [passed](program_peer/metal__f32__rms_norm__long_row__r128_c129.json) | 0.4211 / 0.511 |
| `metal/bf16/rms_norm/ragged/r33_c129` | normalization | 33 × 129 | dense | correct | [passed](operator_peer/metal__bf16__rms_norm__ragged__r33_c129.json) | 0.3193 / 0.3145 | [passed](program_peer/metal__bf16__rms_norm__ragged__r33_c129.json) | 0.2761 / 0.3204 |
| `metal/f32/rms_norm/outer_axis/r127_c64` | normalization | 127 × 64 | dense | correct | [passed](operator_peer/metal__f32__rms_norm__outer_axis__r127_c64.json) | 0.4579 / 0.3255 | [passed](program_peer/metal__f32__rms_norm__outer_axis__r127_c64.json) | 0.4652 / 0.3354 |
| `metal/f32/log_softmax/long_row/r256_c64` | normalization | 256 × 64 | dense | correct | [passed](operator_peer/metal__f32__log_softmax__long_row__r256_c64.json) | 0.4518 / 0.4563 | [passed](program_peer/metal__f32__log_softmax__long_row__r256_c64.json) | 0.4708 / 0.4695 |
| `metal/f32/log_softmax/long_row/r64_c255` | normalization | 64 × 255 | dense | correct | [passed](operator_peer/metal__f32__log_softmax__long_row__r64_c255.json) | 0.3885 / 0.4548 | [passed](program_peer/metal__f32__log_softmax__long_row__r64_c255.json) | 0.3618 / 0.4642 |
| `metal/bf16/rms_norm/long_row/r32_c129` | normalization | 32 × 129 | dense | correct | [passed](operator_peer/metal__bf16__rms_norm__long_row__r32_c129.json) | 0.3127 / 0.3055 | [passed](program_peer/metal__bf16__rms_norm__long_row__r32_c129.json) | 0.3462 / 0.3164 |
| `metal/bf16/rms_norm/long_row/r32_c127` | normalization | 32 × 127 | dense | correct | [passed](operator_peer/metal__bf16__rms_norm__long_row__r32_c127.json) | 0.6053 / 0.31 | [passed](program_peer/metal__bf16__rms_norm__long_row__r32_c127.json) | 0.6274 / 0.3081 |
| `metal/bf16/log_softmax/long_row/r64_c255` | normalization | 64 × 255 | dense | correct | [passed](operator_peer/metal__bf16__log_softmax__long_row__r64_c255.json) | 0.4141 / 0.4893 | [passed](program_peer/metal__bf16__log_softmax__long_row__r64_c255.json) | 0.374 / 0.4541 |
| `metal/bf16/rms_norm/long_row/r32_c128` | normalization | 32 × 128 | dense | correct | [passed](operator_peer/metal__bf16__rms_norm__long_row__r32_c128.json) | 0.3109 / 0.3057 | [passed](program_peer/metal__bf16__rms_norm__long_row__r32_c128.json) | 0.3328 / 0.3114 |
| `metal/f32/rms_norm/long_row/r64_c255` | normalization | 64 × 255 | dense | correct | [passed](operator_peer/metal__f32__rms_norm__long_row__r64_c255.json) | 0.3764 / 0.4706 | [passed](program_peer/metal__f32__rms_norm__long_row__r64_c255.json) | 0.3816 / 0.5099 |
| `metal/bf16/rms_norm/outer_axis/r127_c64` | normalization | 127 × 64 | dense | correct | [passed](operator_peer/metal__bf16__rms_norm__outer_axis__r127_c64.json) | 0.4128 / 0.3668 | [passed](program_peer/metal__bf16__rms_norm__outer_axis__r127_c64.json) | 0.4232 / 0.3685 |
| `metal/bf16/rms_norm/strided/r32_c127` | normalization | 32 × 127 | strided | correct | [passed](operator_peer/metal__bf16__rms_norm__strided__r32_c127.json) | 0.6722 / 0.3048 | [passed](program_peer/metal__bf16__rms_norm__strided__r32_c127.json) | 0.7085 / 0.3116 |
| `metal/f32/rms_norm/long_row/r256_c31` | normalization | 256 × 31 | dense | correct | [passed](operator_peer/metal__f32__rms_norm__long_row__r256_c31.json) | 0.4715 / 0.3225 | [passed](program_peer/metal__f32__rms_norm__long_row__r256_c31.json) | 0.4717 / 0.3329 |
| `metal/bf16/layer_norm/long_row/r32_c257` | normalization | 32 × 257 | dense | correct | [passed](operator_peer/metal__bf16__layer_norm__long_row__r32_c257.json) | 0.314 / 0.2721 | [passed](program_peer/metal__bf16__layer_norm__long_row__r32_c257.json) | 0.3513 / 0.2741 |
| `metal/bf16/rms_norm/long_row/r33_c255` | normalization | 33 × 255 | dense | correct | [passed](operator_peer/metal__bf16__rms_norm__long_row__r33_c255.json) | 0.3563 / 0.3569 | [passed](program_peer/metal__bf16__rms_norm__long_row__r33_c255.json) | 0.3993 / 0.3631 |
| `metal/bf16/log_softmax/long_row/r256_c31` | normalization | 256 × 31 | dense | correct | [passed](operator_peer/metal__bf16__log_softmax__long_row__r256_c31.json) | 0.5049 / 0.3361 | [passed](program_peer/metal__bf16__log_softmax__long_row__r256_c31.json) | 0.4037 / 0.3398 |
| `metal/bf16/rms_norm/long_row/r32_c257` | normalization | 32 × 257 | dense | correct | [passed](operator_peer/metal__bf16__rms_norm__long_row__r32_c257.json) | 0.3173 / 0.3549 | [passed](program_peer/metal__bf16__rms_norm__long_row__r32_c257.json) | 0.3387 / 0.3618 |
| `metal/bf16/rms_norm/long_row/r256_c31` | normalization | 256 × 31 | dense | correct | [passed](operator_peer/metal__bf16__rms_norm__long_row__r256_c31.json) | 0.4613 / 0.3575 | [passed](program_peer/metal__bf16__rms_norm__long_row__r256_c31.json) | 0.466 / 0.3626 |
| `metal/f32/log_softmax/long_row/r256_c31` | normalization | 256 × 31 | dense | correct | [passed](operator_peer/metal__f32__log_softmax__long_row__r256_c31.json) | 0.4104 / 0.3003 | [passed](program_peer/metal__f32__log_softmax__long_row__r256_c31.json) | 0.4387 / 0.3033 |
| `metal/bf16/layer_norm/strided/r64_c128` | normalization | 64 × 128 | strided | correct | [passed](operator_peer/metal__bf16__layer_norm__strided__r64_c128.json) | 0.3777 / 0.2754 | [passed](program_peer/metal__bf16__layer_norm__strided__r64_c128.json) | 0.3997 / 0.2768 |
| `metal/f32/rms_norm/long_row/r33_c255` | normalization | 33 × 255 | dense | correct | [passed](operator_peer/metal__f32__rms_norm__long_row__r33_c255.json) | 0.3709 / 0.315 | [passed](program_peer/metal__f32__rms_norm__long_row__r33_c255.json) | 0.3287 / 0.3176 |
| `metal/bf16/log_softmax/strided/r64_c128` | normalization | 64 × 128 | strided | correct | [passed](operator_peer/metal__bf16__log_softmax__strided__r64_c128.json) | 0.3493 / 0.3013 | [passed](program_peer/metal__bf16__log_softmax__strided__r64_c128.json) | 0.3199 / 0.3227 |
| `metal/bf16/log_softmax/outer_axis/r64_c129` | normalization | 64 × 129 | dense | correct | [passed](operator_peer/metal__bf16__log_softmax__outer_axis__r64_c129.json) | 0.3402 / 0.3399 | [passed](program_peer/metal__bf16__log_softmax__outer_axis__r64_c129.json) | 0.3371 / 0.34 |
| `metal/bf16/log_softmax/long_row/r33_c255` | normalization | 33 × 255 | dense | correct | [passed](operator_peer/metal__bf16__log_softmax__long_row__r33_c255.json) | 0.3194 / 0.339 | [passed](program_peer/metal__bf16__log_softmax__long_row__r33_c255.json) | 0.365 / 0.3446 |
| `metal/f32/log_softmax/strided/r64_c128` | normalization | 64 × 128 | strided | correct | [passed](operator_peer/metal__f32__log_softmax__strided__r64_c128.json) | 0.3665 / 0.2986 | [passed](program_peer/metal__f32__log_softmax__strided__r64_c128.json) | 0.3495 / 0.3007 |
| `metal/f32/layer_norm/strided/r64_c128` | normalization | 64 × 128 | strided | correct | [passed](operator_peer/metal__f32__layer_norm__strided__r64_c128.json) | 0.327 / 0.2803 | [passed](program_peer/metal__f32__layer_norm__strided__r64_c128.json) | 0.4031 / 0.2776 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
