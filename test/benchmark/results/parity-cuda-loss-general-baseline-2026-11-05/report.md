# Parity batch report

Target: `NVIDIA GeForce RTX 3090/7.0.0-38-generic/x86_64`  
Catalog SHA-256: `00d7899204cf66090a5e12fa91ced2107932950aa5349732f83a03cea8280ec6`  
Binary SHA-256: `9f92b876a43d0ec46a772a25cb9fb696750e171e16da75e9830a077464bad7f9`  
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
| `cuda/f32/cross_entropy/small/general_r1_c3` | loss | 1 × 3 | dense | correct | [passed](operator_peer/cuda__f32__cross_entropy__small__general_r1_c3.json) | 0.01787 / 0.02441 | [passed](program_peer/cuda__f32__cross_entropy__small__general_r1_c3.json) | 0.0224 / 0.02795 |
| `cuda/f32/cross_entropy/large/offset_general_r5_c97` | loss | 5 × 97 | offset | correct | [passed](operator_peer/cuda__f32__cross_entropy__large__offset_general_r5_c97.json) | 0.06963 / 0.03268 | [passed](program_peer/cuda__f32__cross_entropy__large__offset_general_r5_c97.json) | 0.07549 / 0.0358 |
| `cuda/f32/cross_entropy/strided/cols_general_r19_c257` | loss | 19 × 257 | strided | correct | [passed](operator_peer/cuda__f32__cross_entropy__strided__cols_general_r19_c257.json) | 0.6123 / 0.03179 | [passed](program_peer/cuda__f32__cross_entropy__strided__cols_general_r19_c257.json) | 0.617 / 0.03447 |
| `cuda/f32/cross_entropy/batched/general_b2_c37_s5` | loss | 2 × 37 × 5 | dense | correct | [passed](operator_peer/cuda__f32__cross_entropy__batched__general_b2_c37_s5.json) | 0.06654 / 0.02534 | [passed](program_peer/cuda__f32__cross_entropy__batched__general_b2_c37_s5.json) | 0.07127 / 0.02827 |
| `cuda/f32/log_softmax_nll/small/general_r1_c3` | loss | 1 × 3 | dense | correct | [passed](operator_peer/cuda__f32__log_softmax_nll__small__general_r1_c3.json) | 0.01753 / 0.02564 | [passed](program_peer/cuda__f32__log_softmax_nll__small__general_r1_c3.json) | 0.02237 / 0.02782 |
| `cuda/f32/log_softmax_nll/large/offset_general_r5_c97` | loss | 5 × 97 | offset | correct | [passed](operator_peer/cuda__f32__log_softmax_nll__large__offset_general_r5_c97.json) | 0.07147 / 0.03336 | [passed](program_peer/cuda__f32__log_softmax_nll__large__offset_general_r5_c97.json) | 0.07334 / 0.03562 |
| `cuda/f32/log_softmax_nll/strided/cols_general_r19_c257` | loss | 19 × 257 | strided | correct | [passed](operator_peer/cuda__f32__log_softmax_nll__strided__cols_general_r19_c257.json) | 0.6122 / 0.03225 | [passed](program_peer/cuda__f32__log_softmax_nll__strided__cols_general_r19_c257.json) | 0.635 / 0.03512 |
| `cuda/f32/log_softmax_nll/batched/general_b2_c37_s5` | loss | 2 × 37 × 5 | dense | correct | [passed](operator_peer/cuda__f32__log_softmax_nll__batched__general_b2_c37_s5.json) | 0.06708 / 0.02599 | [passed](program_peer/cuda__f32__log_softmax_nll__batched__general_b2_c37_s5.json) | 0.07115 / 0.02773 |
| `cuda/bf16/cross_entropy/small/general_r1_c3` | loss | 1 × 3 | dense | correct | [passed](operator_peer/cuda__bf16__cross_entropy__small__general_r1_c3.json) | 0.02387 / 0.03561 | [passed](program_peer/cuda__bf16__cross_entropy__small__general_r1_c3.json) | 0.02622 / 0.03781 |
| `cuda/bf16/cross_entropy/large/offset_general_r5_c97` | loss | 5 × 97 | offset | correct | [passed](operator_peer/cuda__bf16__cross_entropy__large__offset_general_r5_c97.json) | 0.0528 / 0.04003 | [passed](program_peer/cuda__bf16__cross_entropy__large__offset_general_r5_c97.json) | 0.0558 / 0.04307 |
| `cuda/bf16/cross_entropy/strided/cols_general_r19_c257` | loss | 19 × 257 | strided | correct | [passed](operator_peer/cuda__bf16__cross_entropy__strided__cols_general_r19_c257.json) | 0.1848 / 0.03972 | [passed](program_peer/cuda__bf16__cross_entropy__strided__cols_general_r19_c257.json) | 0.1844 / 0.04314 |
| `cuda/bf16/cross_entropy/batched/general_b2_c37_s5` | loss | 2 × 37 × 5 | dense | correct | [passed](operator_peer/cuda__bf16__cross_entropy__batched__general_b2_c37_s5.json) | 0.09161 / 0.03612 | [passed](program_peer/cuda__bf16__cross_entropy__batched__general_b2_c37_s5.json) | 0.09413 / 0.03914 |
| `cuda/bf16/log_softmax_nll/small/general_r1_c3` | loss | 1 × 3 | dense | correct | [passed](operator_peer/cuda__bf16__log_softmax_nll__small__general_r1_c3.json) | 0.02402 / 0.03462 | [passed](program_peer/cuda__bf16__log_softmax_nll__small__general_r1_c3.json) | 0.02626 / 0.03833 |
| `cuda/bf16/log_softmax_nll/large/offset_general_r5_c97` | loss | 5 × 97 | offset | correct | [passed](operator_peer/cuda__bf16__log_softmax_nll__large__offset_general_r5_c97.json) | 0.05323 / 0.04036 | [passed](program_peer/cuda__bf16__log_softmax_nll__large__offset_general_r5_c97.json) | 0.05535 / 0.04319 |
| `cuda/bf16/log_softmax_nll/strided/cols_general_r19_c257` | loss | 19 × 257 | strided | correct | [passed](operator_peer/cuda__bf16__log_softmax_nll__strided__cols_general_r19_c257.json) | 0.182 / 0.03908 | [passed](program_peer/cuda__bf16__log_softmax_nll__strided__cols_general_r19_c257.json) | 0.1872 / 0.04193 |
| `cuda/bf16/log_softmax_nll/batched/general_b2_c37_s5` | loss | 2 × 37 × 5 | dense | correct | [passed](operator_peer/cuda__bf16__log_softmax_nll__batched__general_b2_c37_s5.json) | 0.08974 / 0.03544 | [passed](program_peer/cuda__bf16__log_softmax_nll__batched__general_b2_c37_s5.json) | 0.09423 / 0.03851 |
| `cuda/f32/cross_entropy/large/general_r1_c8193` | loss | 1 × 8193 | dense | correct | [passed](operator_peer/cuda__f32__cross_entropy__large__general_r1_c8193.json) | 0.7834 / 0.02407 | [passed](program_peer/cuda__f32__cross_entropy__large__general_r1_c8193.json) | 0.7881 / 0.02723 |
| `cuda/f32/cross_entropy/large/general_r11_c1031` | loss | 11 × 1031 | dense | correct | [passed](operator_peer/cuda__f32__cross_entropy__large__general_r11_c1031.json) | 1.129 / 0.02398 | [passed](program_peer/cuda__f32__cross_entropy__large__general_r11_c1031.json) | 1.133 / 0.02737 |
| `cuda/f32/log_softmax_nll/large/general_r1_c8193` | loss | 1 × 8193 | dense | correct | [passed](operator_peer/cuda__f32__log_softmax_nll__large__general_r1_c8193.json) | 0.7832 / 0.02405 | [passed](program_peer/cuda__f32__log_softmax_nll__large__general_r1_c8193.json) | 0.7874 / 0.02739 |
| `cuda/f32/log_softmax_nll/large/general_r11_c1031` | loss | 11 × 1031 | dense | correct | [passed](operator_peer/cuda__f32__log_softmax_nll__large__general_r11_c1031.json) | 1.129 / 0.02471 | [passed](program_peer/cuda__f32__log_softmax_nll__large__general_r11_c1031.json) | 1.133 / 0.0275 |
| `cuda/bf16/cross_entropy/large/general_r1_c8193` | loss | 1 × 8193 | dense | correct | [passed](operator_peer/cuda__bf16__cross_entropy__large__general_r1_c8193.json) | 0.08756 / 0.03491 | [passed](program_peer/cuda__bf16__cross_entropy__large__general_r1_c8193.json) | 0.08931 / 0.03832 |
| `cuda/bf16/cross_entropy/large/general_r11_c1031` | loss | 11 × 1031 | dense | correct | [passed](operator_peer/cuda__bf16__cross_entropy__large__general_r11_c1031.json) | 0.1832 / 0.03509 | [passed](program_peer/cuda__bf16__cross_entropy__large__general_r11_c1031.json) | 0.1853 / 0.03781 |
| `cuda/bf16/log_softmax_nll/large/general_r1_c8193` | loss | 1 × 8193 | dense | correct | [passed](operator_peer/cuda__bf16__log_softmax_nll__large__general_r1_c8193.json) | 0.09304 / 0.03449 | [passed](program_peer/cuda__bf16__log_softmax_nll__large__general_r1_c8193.json) | 0.09548 / 0.03835 |
| `cuda/bf16/log_softmax_nll/large/general_r11_c1031` | loss | 11 × 1031 | dense | correct | [passed](operator_peer/cuda__bf16__log_softmax_nll__large__general_r11_c1031.json) | 0.1835 / 0.03453 | [passed](program_peer/cuda__bf16__log_softmax_nll__large__general_r11_c1031.json) | 0.1854 / 0.03808 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
