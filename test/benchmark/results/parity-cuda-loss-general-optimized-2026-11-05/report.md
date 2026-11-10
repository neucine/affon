# Parity batch report

Target: `NVIDIA GeForce RTX 3090/7.0.0-38-generic/x86_64`  
Catalog SHA-256: `00d7899204cf66090a5e12fa91ced2107932950aa5349732f83a03cea8280ec6`  
Binary SHA-256: `23be7b4de8c44b10e0c1236e74a4c559029d7c52ff14d1b9120b13f1d0988d00`  
Timing: 30 iterations × 5 samples × 3 rounds

## Checks

- Cases run: 24
- Numerical check passed: 24
- Operator check and timing passed: 24
- Program check and timing passed: 24
- Numerical failures: 0
- Operator failures: 0
- Program failures: 0

## Cases

| Case | Family | Shape | Layout | Numerical | Operator | Affon / PyTorch ms | Program | Affon / PyTorch ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `cuda/f32/cross_entropy/small/general_r1_c3` | loss | 1 × 3 | dense | correct | [passed](operator_peer/cuda__f32__cross_entropy__small__general_r1_c3.json) | 0.02024 / 0.02442 | [passed](program_peer/cuda__f32__cross_entropy__small__general_r1_c3.json) | 0.02487 / 0.02758 |
| `cuda/f32/cross_entropy/large/offset_general_r5_c97` | loss | 5 × 97 | offset | correct | [passed](operator_peer/cuda__f32__cross_entropy__large__offset_general_r5_c97.json) | 0.02711 / 0.03336 | [passed](program_peer/cuda__f32__cross_entropy__large__offset_general_r5_c97.json) | 0.03221 / 0.03634 |
| `cuda/f32/cross_entropy/strided/cols_general_r19_c257` | loss | 19 × 257 | strided | correct | [passed](operator_peer/cuda__f32__cross_entropy__strided__cols_general_r19_c257.json) | 0.07143 / 0.03213 | [passed](program_peer/cuda__f32__cross_entropy__strided__cols_general_r19_c257.json) | 0.07687 / 0.03482 |
| `cuda/f32/cross_entropy/batched/general_b2_c37_s5` | loss | 2 × 37 × 5 | dense | correct | [passed](operator_peer/cuda__f32__cross_entropy__batched__general_b2_c37_s5.json) | 0.04047 / 0.02581 | [passed](program_peer/cuda__f32__cross_entropy__batched__general_b2_c37_s5.json) | 0.04588 / 0.02807 |
| `cuda/f32/log_softmax_nll/small/general_r1_c3` | loss | 1 × 3 | dense | correct | [passed](operator_peer/cuda__f32__log_softmax_nll__small__general_r1_c3.json) | 0.02021 / 0.0246 | [passed](program_peer/cuda__f32__log_softmax_nll__small__general_r1_c3.json) | 0.02463 / 0.02784 |
| `cuda/f32/log_softmax_nll/large/offset_general_r5_c97` | loss | 5 × 97 | offset | correct | [passed](operator_peer/cuda__f32__log_softmax_nll__large__offset_general_r5_c97.json) | 0.02649 / 0.03272 | [passed](program_peer/cuda__f32__log_softmax_nll__large__offset_general_r5_c97.json) | 0.03194 / 0.03679 |
| `cuda/f32/log_softmax_nll/strided/cols_general_r19_c257` | loss | 19 × 257 | strided | correct | [passed](operator_peer/cuda__f32__log_softmax_nll__strided__cols_general_r19_c257.json) | 0.0685 / 0.03297 | [passed](program_peer/cuda__f32__log_softmax_nll__strided__cols_general_r19_c257.json) | 0.07629 / 0.03475 |
| `cuda/f32/log_softmax_nll/batched/general_b2_c37_s5` | loss | 2 × 37 × 5 | dense | correct | [passed](operator_peer/cuda__f32__log_softmax_nll__batched__general_b2_c37_s5.json) | 0.04087 / 0.03191 | [passed](program_peer/cuda__f32__log_softmax_nll__batched__general_b2_c37_s5.json) | 0.04591 / 0.02801 |
| `cuda/bf16/cross_entropy/small/general_r1_c3` | loss | 1 × 3 | dense | correct | [passed](operator_peer/cuda__bf16__cross_entropy__small__general_r1_c3.json) | 0.02003 / 0.03626 | [passed](program_peer/cuda__bf16__cross_entropy__small__general_r1_c3.json) | 0.02228 / 0.03855 |
| `cuda/bf16/cross_entropy/large/offset_general_r5_c97` | loss | 5 × 97 | offset | correct | [passed](operator_peer/cuda__bf16__cross_entropy__large__offset_general_r5_c97.json) | 0.02701 / 0.04036 | [passed](program_peer/cuda__bf16__cross_entropy__large__offset_general_r5_c97.json) | 0.02966 / 0.04392 |
| `cuda/bf16/cross_entropy/strided/cols_general_r19_c257` | loss | 19 × 257 | strided | correct | [passed](operator_peer/cuda__bf16__cross_entropy__strided__cols_general_r19_c257.json) | 0.07207 / 0.03906 | [passed](program_peer/cuda__bf16__cross_entropy__strided__cols_general_r19_c257.json) | 0.07334 / 0.0429 |
| `cuda/bf16/cross_entropy/batched/general_b2_c37_s5` | loss | 2 × 37 × 5 | dense | correct | [passed](operator_peer/cuda__bf16__cross_entropy__batched__general_b2_c37_s5.json) | 0.03561 / 0.03537 | [passed](program_peer/cuda__bf16__cross_entropy__batched__general_b2_c37_s5.json) | 0.03743 / 0.03938 |
| `cuda/bf16/log_softmax_nll/small/general_r1_c3` | loss | 1 × 3 | dense | correct | [passed](operator_peer/cuda__bf16__log_softmax_nll__small__general_r1_c3.json) | 0.02018 / 0.03469 | [passed](program_peer/cuda__bf16__log_softmax_nll__small__general_r1_c3.json) | 0.02208 / 0.0379 |
| `cuda/bf16/log_softmax_nll/large/offset_general_r5_c97` | loss | 5 × 97 | offset | correct | [passed](operator_peer/cuda__bf16__log_softmax_nll__large__offset_general_r5_c97.json) | 0.0272 / 0.04054 | [passed](program_peer/cuda__bf16__log_softmax_nll__large__offset_general_r5_c97.json) | 0.02925 / 0.04354 |
| `cuda/bf16/log_softmax_nll/strided/cols_general_r19_c257` | loss | 19 × 257 | strided | correct | [passed](operator_peer/cuda__bf16__log_softmax_nll__strided__cols_general_r19_c257.json) | 0.07081 / 0.03904 | [passed](program_peer/cuda__bf16__log_softmax_nll__strided__cols_general_r19_c257.json) | 0.07408 / 0.04232 |
| `cuda/bf16/log_softmax_nll/batched/general_b2_c37_s5` | loss | 2 × 37 × 5 | dense | correct | [passed](operator_peer/cuda__bf16__log_softmax_nll__batched__general_b2_c37_s5.json) | 0.03531 / 0.03595 | [passed](program_peer/cuda__bf16__log_softmax_nll__batched__general_b2_c37_s5.json) | 0.03693 / 0.03905 |
| `cuda/f32/cross_entropy/large/general_r1_c8193` | loss | 1 × 8193 | dense | correct | [passed](operator_peer/cuda__f32__cross_entropy__large__general_r1_c8193.json) | 0.2308 / 0.02502 | [passed](program_peer/cuda__f32__cross_entropy__large__general_r1_c8193.json) | 0.2352 / 0.02749 |
| `cuda/f32/cross_entropy/large/general_r11_c1031` | loss | 11 × 1031 | dense | correct | [passed](operator_peer/cuda__f32__cross_entropy__large__general_r11_c1031.json) | 0.1147 / 0.02426 | [passed](program_peer/cuda__f32__cross_entropy__large__general_r11_c1031.json) | 0.1239 / 0.02814 |
| `cuda/f32/log_softmax_nll/large/general_r1_c8193` | loss | 1 × 8193 | dense | correct | [passed](operator_peer/cuda__f32__log_softmax_nll__large__general_r1_c8193.json) | 0.2191 / 0.02449 | [passed](program_peer/cuda__f32__log_softmax_nll__large__general_r1_c8193.json) | 0.234 / 0.02794 |
| `cuda/f32/log_softmax_nll/large/general_r11_c1031` | loss | 11 × 1031 | dense | correct | [passed](operator_peer/cuda__f32__log_softmax_nll__large__general_r11_c1031.json) | 0.1198 / 0.02473 | [passed](program_peer/cuda__f32__log_softmax_nll__large__general_r11_c1031.json) | 0.1245 / 0.02744 |
| `cuda/bf16/cross_entropy/large/general_r1_c8193` | loss | 1 × 8193 | dense | correct | [passed](operator_peer/cuda__bf16__cross_entropy__large__general_r1_c8193.json) | 0.2328 / 0.03475 | [passed](program_peer/cuda__bf16__cross_entropy__large__general_r1_c8193.json) | 0.2328 / 0.038 |
| `cuda/bf16/cross_entropy/large/general_r11_c1031` | loss | 11 × 1031 | dense | correct | [passed](operator_peer/cuda__bf16__cross_entropy__large__general_r11_c1031.json) | 0.1201 / 0.03456 | [passed](program_peer/cuda__bf16__cross_entropy__large__general_r11_c1031.json) | 0.1223 / 0.038 |
| `cuda/bf16/log_softmax_nll/large/general_r1_c8193` | loss | 1 × 8193 | dense | correct | [passed](operator_peer/cuda__bf16__log_softmax_nll__large__general_r1_c8193.json) | 0.2274 / 0.03494 | [passed](program_peer/cuda__bf16__log_softmax_nll__large__general_r1_c8193.json) | 0.2347 / 0.03833 |
| `cuda/bf16/log_softmax_nll/large/general_r11_c1031` | loss | 11 × 1031 | dense | correct | [passed](operator_peer/cuda__bf16__log_softmax_nll__large__general_r11_c1031.json) | 0.1198 / 0.03534 | [passed](program_peer/cuda__bf16__log_softmax_nll__large__general_r11_c1031.json) | 0.1216 / 0.03809 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
