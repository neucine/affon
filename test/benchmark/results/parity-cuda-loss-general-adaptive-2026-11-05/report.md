# Parity batch report

Target: `NVIDIA GeForce RTX 3090/7.0.0-38-generic/x86_64`  
Catalog SHA-256: `00d7899204cf66090a5e12fa91ced2107932950aa5349732f83a03cea8280ec6`  
Binary SHA-256: `12b96cf6f255799f8b3059588a660354d32d75c2215155388e13a4d6dd838593`  
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
| `cuda/f32/cross_entropy/small/general_r1_c3` | loss | 1 × 3 | dense | correct | [passed](operator_peer/cuda__f32__cross_entropy__small__general_r1_c3.json) | 0.02072 / 0.02476 | [passed](program_peer/cuda__f32__cross_entropy__small__general_r1_c3.json) | 0.02498 / 0.02794 |
| `cuda/f32/cross_entropy/large/offset_general_r5_c97` | loss | 5 × 97 | offset | correct | [passed](operator_peer/cuda__f32__cross_entropy__large__offset_general_r5_c97.json) | 0.03296 / 0.03397 | [passed](program_peer/cuda__f32__cross_entropy__large__offset_general_r5_c97.json) | 0.03708 / 0.03729 |
| `cuda/f32/cross_entropy/strided/cols_general_r19_c257` | loss | 19 × 257 | strided | correct | [passed](operator_peer/cuda__f32__cross_entropy__strided__cols_general_r19_c257.json) | 0.07049 / 0.03268 | [passed](program_peer/cuda__f32__cross_entropy__strided__cols_general_r19_c257.json) | 0.07523 / 0.03481 |
| `cuda/f32/cross_entropy/batched/general_b2_c37_s5` | loss | 2 × 37 × 5 | dense | correct | [passed](operator_peer/cuda__f32__cross_entropy__batched__general_b2_c37_s5.json) | 0.04073 / 0.03211 | [passed](program_peer/cuda__f32__cross_entropy__batched__general_b2_c37_s5.json) | 0.04582 / 0.02857 |
| `cuda/f32/log_softmax_nll/small/general_r1_c3` | loss | 1 × 3 | dense | correct | [passed](operator_peer/cuda__f32__log_softmax_nll__small__general_r1_c3.json) | 0.0202 / 0.02488 | [passed](program_peer/cuda__f32__log_softmax_nll__small__general_r1_c3.json) | 0.02523 / 0.02807 |
| `cuda/f32/log_softmax_nll/large/offset_general_r5_c97` | loss | 5 × 97 | offset | correct | [passed](operator_peer/cuda__f32__log_softmax_nll__large__offset_general_r5_c97.json) | 0.03261 / 0.03344 | [passed](program_peer/cuda__f32__log_softmax_nll__large__offset_general_r5_c97.json) | 0.03682 / 0.03635 |
| `cuda/f32/log_softmax_nll/strided/cols_general_r19_c257` | loss | 19 × 257 | strided | correct | [passed](operator_peer/cuda__f32__log_softmax_nll__strided__cols_general_r19_c257.json) | 0.07194 / 0.03197 | [passed](program_peer/cuda__f32__log_softmax_nll__strided__cols_general_r19_c257.json) | 0.07431 / 0.0355 |
| `cuda/f32/log_softmax_nll/batched/general_b2_c37_s5` | loss | 2 × 37 × 5 | dense | correct | [passed](operator_peer/cuda__f32__log_softmax_nll__batched__general_b2_c37_s5.json) | 0.0413 / 0.03195 | [passed](program_peer/cuda__f32__log_softmax_nll__batched__general_b2_c37_s5.json) | 0.04583 / 0.02908 |
| `cuda/bf16/cross_entropy/small/general_r1_c3` | loss | 1 × 3 | dense | correct | [passed](operator_peer/cuda__bf16__cross_entropy__small__general_r1_c3.json) | 0.02052 / 0.03623 | [passed](program_peer/cuda__bf16__cross_entropy__small__general_r1_c3.json) | 0.02274 / 0.03825 |
| `cuda/bf16/cross_entropy/large/offset_general_r5_c97` | loss | 5 × 97 | offset | correct | [passed](operator_peer/cuda__bf16__cross_entropy__large__offset_general_r5_c97.json) | 0.03254 / 0.04027 | [passed](program_peer/cuda__bf16__cross_entropy__large__offset_general_r5_c97.json) | 0.0348 / 0.04342 |
| `cuda/bf16/cross_entropy/strided/cols_general_r19_c257` | loss | 19 × 257 | strided | correct | [passed](operator_peer/cuda__bf16__cross_entropy__strided__cols_general_r19_c257.json) | 0.07203 / 0.04006 | [passed](program_peer/cuda__bf16__cross_entropy__strided__cols_general_r19_c257.json) | 0.07244 / 0.04225 |
| `cuda/bf16/cross_entropy/batched/general_b2_c37_s5` | loss | 2 × 37 × 5 | dense | correct | [passed](operator_peer/cuda__bf16__cross_entropy__batched__general_b2_c37_s5.json) | 0.03551 / 0.0357 | [passed](program_peer/cuda__bf16__cross_entropy__batched__general_b2_c37_s5.json) | 0.03771 / 0.03835 |
| `cuda/bf16/log_softmax_nll/small/general_r1_c3` | loss | 1 × 3 | dense | correct | [passed](operator_peer/cuda__bf16__log_softmax_nll__small__general_r1_c3.json) | 0.02087 / 0.03497 | [passed](program_peer/cuda__bf16__log_softmax_nll__small__general_r1_c3.json) | 0.02248 / 0.0382 |
| `cuda/bf16/log_softmax_nll/large/offset_general_r5_c97` | loss | 5 × 97 | offset | correct | [passed](operator_peer/cuda__bf16__log_softmax_nll__large__offset_general_r5_c97.json) | 0.03304 / 0.04191 | [passed](program_peer/cuda__bf16__log_softmax_nll__large__offset_general_r5_c97.json) | 0.03516 / 0.04351 |
| `cuda/bf16/log_softmax_nll/strided/cols_general_r19_c257` | loss | 19 × 257 | strided | correct | [passed](operator_peer/cuda__bf16__log_softmax_nll__strided__cols_general_r19_c257.json) | 0.07127 / 0.03973 | [passed](program_peer/cuda__bf16__log_softmax_nll__strided__cols_general_r19_c257.json) | 0.0743 / 0.04213 |
| `cuda/bf16/log_softmax_nll/batched/general_b2_c37_s5` | loss | 2 × 37 × 5 | dense | correct | [passed](operator_peer/cuda__bf16__log_softmax_nll__batched__general_b2_c37_s5.json) | 0.03545 / 0.036 | [passed](program_peer/cuda__bf16__log_softmax_nll__batched__general_b2_c37_s5.json) | 0.03808 / 0.039 |
| `cuda/f32/cross_entropy/large/general_r1_c8193` | loss | 1 × 8193 | dense | correct | [passed](operator_peer/cuda__f32__cross_entropy__large__general_r1_c8193.json) | 0.09334 / 0.02445 | [passed](program_peer/cuda__f32__cross_entropy__large__general_r1_c8193.json) | 0.09797 / 0.02778 |
| `cuda/f32/cross_entropy/large/general_r11_c1031` | loss | 11 × 1031 | dense | correct | [passed](operator_peer/cuda__f32__cross_entropy__large__general_r11_c1031.json) | 0.1199 / 0.02452 | [passed](program_peer/cuda__f32__cross_entropy__large__general_r11_c1031.json) | 0.1248 / 0.02854 |
| `cuda/f32/log_softmax_nll/large/general_r1_c8193` | loss | 1 × 8193 | dense | correct | [passed](operator_peer/cuda__f32__log_softmax_nll__large__general_r1_c8193.json) | 0.09304 / 0.02592 | [passed](program_peer/cuda__f32__log_softmax_nll__large__general_r1_c8193.json) | 0.09771 / 0.02772 |
| `cuda/f32/log_softmax_nll/large/general_r11_c1031` | loss | 11 × 1031 | dense | correct | [passed](operator_peer/cuda__f32__log_softmax_nll__large__general_r11_c1031.json) | 0.1199 / 0.02537 | [passed](program_peer/cuda__f32__log_softmax_nll__large__general_r11_c1031.json) | 0.122 / 0.02813 |
| `cuda/bf16/cross_entropy/large/general_r1_c8193` | loss | 1 × 8193 | dense | correct | [passed](operator_peer/cuda__bf16__cross_entropy__large__general_r1_c8193.json) | 0.0932 / 0.03671 | [passed](program_peer/cuda__bf16__cross_entropy__large__general_r1_c8193.json) | 0.09184 / 0.03823 |
| `cuda/bf16/cross_entropy/large/general_r11_c1031` | loss | 11 × 1031 | dense | correct | [passed](operator_peer/cuda__bf16__cross_entropy__large__general_r11_c1031.json) | 0.1179 / 0.0353 | [passed](program_peer/cuda__bf16__cross_entropy__large__general_r11_c1031.json) | 0.1227 / 0.03843 |
| `cuda/bf16/log_softmax_nll/large/general_r1_c8193` | loss | 1 × 8193 | dense | correct | [passed](operator_peer/cuda__bf16__log_softmax_nll__large__general_r1_c8193.json) | 0.09157 / 0.03551 | [passed](program_peer/cuda__bf16__log_softmax_nll__large__general_r1_c8193.json) | 0.09322 / 0.03834 |
| `cuda/bf16/log_softmax_nll/large/general_r11_c1031` | loss | 11 × 1031 | dense | correct | [passed](operator_peer/cuda__bf16__log_softmax_nll__large__general_r11_c1031.json) | 0.12 / 0.03619 | [passed](program_peer/cuda__bf16__log_softmax_nll__large__general_r11_c1031.json) | 0.119 / 0.03811 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
