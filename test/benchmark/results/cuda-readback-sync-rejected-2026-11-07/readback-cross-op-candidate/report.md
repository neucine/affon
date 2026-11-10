# Parity batch report

Target: `NVIDIA GeForce RTX 3090/7.0.0-38-generic/x86_64`  
Catalog SHA-256: `49b6a93871fe8587314da62dacc4e8895896e80ea7b68d91d2e3b6ff8ecde7f7`  
Binary SHA-256: `c12eb8cc3336da1c7fa30a3b51e01e6c6a8b84c509b2fdb542fb73d69093e077`  
Timing: 10 iterations × 5 samples × 3 rounds

## Checks

- Cases run: 49
- Numerical check passed: 49
- Operator check and timing passed: 49
- Program check and timing passed: 49
- Numerical failures: 0
- Operator failures: 0
- Program failures: 0
- Operator checks pending: 0
- Program checks pending: 0

## Cases

| Case | Family | Shape | Layout | Numerical | Operator | Affon / PyTorch ms | Program | Affon / PyTorch ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `cuda/f32/matmul/batched` | matrix | 4 × 8 × 16 × 12 | dense | correct | [passed](operator_peer/cuda__f32__matmul__batched.json) | 0.04183 / 0.02032 | [passed](program_peer/cuda__f32__matmul__batched.json) | 0.04837 / 0.02404 |
| `cuda/bf16/matmul/batched` | matrix | 4 × 8 × 16 × 12 | dense | correct | [passed](operator_peer/cuda__bf16__matmul__batched.json) | 0.04552 / 0.02137 | [passed](program_peer/cuda__bf16__matmul__batched.json) | 0.04915 / 0.02483 |
| `cuda/f32/index_select/batched` | indexing | 3 × 7 × 11 | dense | correct | [passed](operator_peer/cuda__f32__index_select__batched.json) | 0.02421 / 0.01284 | [passed](program_peer/cuda__f32__index_select__batched.json) | 0.02885 / 0.01621 |
| `cuda/bf16/index_select/batched` | indexing | 3 × 7 × 11 | dense | correct | [passed](operator_peer/cuda__bf16__index_select__batched.json) | 0.02818 / 0.01311 | [passed](program_peer/cuda__bf16__index_select__batched.json) | 0.03348 / 0.016 |
| `cuda/f32/gather/batched` | indexing | 16 × 32 | dense | correct | [passed](operator_peer/cuda__f32__gather__batched.json) | 0.02321 / 0.01414 | [passed](program_peer/cuda__f32__gather__batched.json) | 0.02883 / 0.01786 |
| `cuda/bf16/gather/batched` | indexing | 16 × 32 | dense | correct | [passed](operator_peer/cuda__bf16__gather__batched.json) | 0.03049 / 0.01435 | [passed](program_peer/cuda__bf16__gather__batched.json) | 0.03546 / 0.01725 |
| `cuda/f32/layer_norm/long_row` | normalization | 128 × 512 | dense | correct | [passed](operator_peer/cuda__f32__layer_norm__long_row.json) | 1.113 / 0.5822 | [passed](program_peer/cuda__f32__layer_norm__long_row.json) | 1.129 / 0.5853 |
| `cuda/bf16/layer_norm/long_row` | normalization | 128 × 512 | dense | correct | [passed](operator_peer/cuda__bf16__layer_norm__long_row.json) | 1.112 / 0.586 | [passed](program_peer/cuda__bf16__layer_norm__long_row.json) | 1.097 / 0.5605 |
| `cuda/f32/one_hot/batched/b1_r16_c17_k2` | indexing | 1 × 16 × 17 | dense | correct | [passed](operator_peer/cuda__f32__one_hot__batched__b1_r16_c17_k2.json) | 0.06666 / 0.03109 | [passed](program_peer/cuda__f32__one_hot__batched__b1_r16_c17_k2.json) | 0.07143 / 0.03461 |
| `cuda/f32/softmax/long_row` | normalization | 128 × 512 | dense | correct | [passed](operator_peer/cuda__f32__softmax__long_row.json) | 1.076 / 0.552 | [passed](program_peer/cuda__f32__softmax__long_row.json) | 1.095 / 0.5558 |
| `cuda/bf16/softmax/long_row` | normalization | 128 × 512 | dense | correct | [passed](operator_peer/cuda__bf16__softmax__long_row.json) | 1.09 / 0.549 | [passed](program_peer/cuda__bf16__softmax__long_row.json) | 1.102 / 0.5582 |
| `cuda/f32/rms_norm/long_row` | normalization | 128 × 512 | dense | correct | [passed](operator_peer/cuda__f32__rms_norm__long_row.json) | 1.108 / 0.5592 | [passed](program_peer/cuda__f32__rms_norm__long_row.json) | 1.122 / 0.574 |
| `cuda/bf16/rms_norm/long_row` | normalization | 128 × 512 | dense | correct | [passed](operator_peer/cuda__bf16__rms_norm__long_row.json) | 1.099 / 0.6024 | [passed](program_peer/cuda__bf16__rms_norm__long_row.json) | 1.098 / 0.568 |
| `cuda/f32/log_softmax/long_row` | normalization | 128 × 512 | dense | correct | [passed](operator_peer/cuda__f32__log_softmax__long_row.json) | 1.077 / 0.5547 | [passed](program_peer/cuda__f32__log_softmax__long_row.json) | 1.102 / 0.558 |
| `cuda/bf16/log_softmax/long_row` | normalization | 128 × 512 | dense | correct | [passed](operator_peer/cuda__bf16__log_softmax__long_row.json) | 1.085 / 0.5601 | [passed](program_peer/cuda__bf16__log_softmax__long_row.json) | 1.095 / 0.5728 |
| `cuda/f32/topk/batched/dense_axis1_b2_r16_c129_k8` | indexing | 2 × 16 × 129 | dense | correct | [passed](operator_peer/cuda__f32__topk__batched__dense_axis1_b2_r16_c129_k8.json) | 0.1098 / 0.08318 | [passed](program_peer/cuda__f32__topk__batched__dense_axis1_b2_r16_c129_k8.json) | 0.1164 / 0.08125 |
| `cuda/bf16/topk/batched/dense_axis1_b2_r16_c129_k8` | indexing | 2 × 16 × 129 | dense | correct | [passed](operator_peer/cuda__bf16__topk__batched__dense_axis1_b2_r16_c129_k8.json) | 0.1225 / 0.08217 | [passed](program_peer/cuda__bf16__topk__batched__dense_axis1_b2_r16_c129_k8.json) | 0.124 / 0.07527 |
| `cuda/f32/mean_axis/long_axis` | reduction | 128 × 768 | dense | correct | [passed](operator_peer/cuda__f32__mean_axis__long_axis.json) | 0.01974 / 0.01458 | [passed](program_peer/cuda__f32__mean_axis__long_axis.json) | 0.02471 / 0.01591 |
| `cuda/bf16/mean_axis/long_axis` | reduction | 128 × 768 | dense | correct | [passed](operator_peer/cuda__bf16__mean_axis__long_axis.json) | 0.02512 / 0.01307 | [passed](program_peer/cuda__bf16__mean_axis__long_axis.json) | 0.02729 / 0.01639 |
| `cuda/f32/sum_axis/long_axis` | reduction | 128 × 768 | dense | correct | [passed](operator_peer/cuda__f32__sum_axis__long_axis.json) | 0.0198 / 0.01486 | [passed](program_peer/cuda__f32__sum_axis__long_axis.json) | 0.02454 / 0.01613 |
| `cuda/bf16/sum_axis/long_axis` | reduction | 128 × 768 | dense | correct | [passed](operator_peer/cuda__bf16__sum_axis__long_axis.json) | 0.02507 / 0.01306 | [passed](program_peer/cuda__bf16__sum_axis__long_axis.json) | 0.02727 / 0.01602 |
| `cuda/f32/argmax_axis/long_axis/r128_c31` | reduction | 128 × 31 | dense | correct | [passed](operator_peer/cuda__f32__argmax_axis__long_axis__r128_c31.json) | 0.3778 / 0.01329 | [passed](program_peer/cuda__f32__argmax_axis__long_axis__r128_c31.json) | 0.3821 / 0.01647 |
| `cuda/bf16/argmax_axis/long_axis/r128_c31` | reduction | 128 × 31 | dense | correct | [passed](operator_peer/cuda__bf16__argmax_axis__long_axis__r128_c31.json) | 0.02816 / 0.01332 | [passed](program_peer/cuda__bf16__argmax_axis__long_axis__r128_c31.json) | 0.03292 / 0.0165 |
| `cuda/f32/argmin_axis/long_axis/r128_c31` | reduction | 128 × 31 | dense | correct | [passed](operator_peer/cuda__f32__argmin_axis__long_axis__r128_c31.json) | 0.378 / 0.01337 | [passed](program_peer/cuda__f32__argmin_axis__long_axis__r128_c31.json) | 0.3821 / 0.01668 |
| `cuda/bf16/argmin_axis/long_axis/r128_c31` | reduction | 128 × 31 | dense | correct | [passed](operator_peer/cuda__bf16__argmin_axis__long_axis__r128_c31.json) | 0.02818 / 0.01316 | [passed](program_peer/cuda__bf16__argmin_axis__long_axis__r128_c31.json) | 0.03299 / 0.01616 |
| `cuda/f32/log_softmax_nll/batched` | loss | 3 × 11 × 7 | dense | correct | [passed](operator_peer/cuda__f32__log_softmax_nll__batched.json) | 0.0434 / 0.02501 | [passed](program_peer/cuda__f32__log_softmax_nll__batched.json) | 0.0479 / 0.02863 |
| `cuda/bf16/log_softmax_nll/batched/dense_b2_c33_s17` | loss | 2 × 33 × 17 | dense | correct | [passed](operator_peer/cuda__bf16__log_softmax_nll__batched__dense_b2_c33_s17.json) | 0.05993 / 0.03513 | [passed](program_peer/cuda__bf16__log_softmax_nll__batched__dense_b2_c33_s17.json) | 0.0626 / 0.03864 |
| `cuda/f32/add/broadcast` | arithmetic | 128 × 512 | dense | correct | [passed](operator_peer/cuda__f32__add__broadcast.json) | 1.095 / 0.5492 | [passed](program_peer/cuda__f32__add__broadcast.json) | 1.109 / 0.5631 |
| `cuda/bf16/add/broadcast` | arithmetic | 128 × 512 | dense | correct | [passed](operator_peer/cuda__bf16__add__broadcast.json) | 1.079 / 0.5809 | [passed](program_peer/cuda__bf16__add__broadcast.json) | 1.101 / 0.6089 |
| `cuda/f32/div/broadcast` | arithmetic | 128 × 512 | dense | correct | [passed](operator_peer/cuda__f32__div__broadcast.json) | 1.094 / 0.5821 | [passed](program_peer/cuda__f32__div__broadcast.json) | 1.103 / 0.558 |
| `cuda/bf16/div/broadcast` | arithmetic | 128 × 512 | dense | correct | [passed](operator_peer/cuda__bf16__div__broadcast.json) | 1.073 / 0.561 | [passed](program_peer/cuda__bf16__div__broadcast.json) | 1.093 / 0.5615 |
| `cuda/f32/mul/broadcast` | arithmetic | 128 × 512 | dense | correct | [passed](operator_peer/cuda__f32__mul__broadcast.json) | 1.099 / 0.5985 | [passed](program_peer/cuda__f32__mul__broadcast.json) | 1.104 / 0.6062 |
| `cuda/bf16/mul/broadcast` | arithmetic | 128 × 512 | dense | correct | [passed](operator_peer/cuda__bf16__mul__broadcast.json) | 1.084 / 0.5565 | [passed](program_peer/cuda__bf16__mul__broadcast.json) | 1.09 / 0.6185 |
| `cuda/f32/sub/broadcast` | arithmetic | 128 × 512 | dense | correct | [passed](operator_peer/cuda__f32__sub__broadcast.json) | 1.108 / 0.5603 | [passed](program_peer/cuda__f32__sub__broadcast.json) | 1.123 / 0.5675 |
| `cuda/bf16/sub/broadcast` | arithmetic | 128 × 512 | dense | correct | [passed](operator_peer/cuda__bf16__sub__broadcast.json) | 1.085 / 0.5656 | [passed](program_peer/cuda__bf16__sub__broadcast.json) | 1.1 / 0.6195 |
| `cuda/f32/sigmoid/large_dense` | elementwise_math | 32 × 64 | dense | correct | [passed](operator_peer/cuda__f32__sigmoid__large_dense.json) | 0.06866 / 0.03464 | [passed](program_peer/cuda__f32__sigmoid__large_dense.json) | 0.073 / 0.03748 |
| `cuda/bf16/sigmoid/large_dense` | elementwise_math | 32 × 64 | dense | correct | [passed](operator_peer/cuda__bf16__sigmoid__large_dense.json) | 0.06154 / 0.02793 | [passed](program_peer/cuda__bf16__sigmoid__large_dense.json) | 0.06487 / 0.0342 |
| `cuda/f32/tanh/large_dense` | elementwise_math | 32 × 64 | dense | correct | [passed](operator_peer/cuda__f32__tanh__large_dense.json) | 0.06886 / 0.02907 | [passed](program_peer/cuda__f32__tanh__large_dense.json) | 0.07274 / 0.03695 |
| `cuda/bf16/tanh/large_dense` | elementwise_math | 32 × 64 | dense | correct | [passed](operator_peer/cuda__bf16__tanh__large_dense.json) | 0.06215 / 0.02791 | [passed](program_peer/cuda__bf16__tanh__large_dense.json) | 0.06486 / 0.03392 |
| `cuda/f32/gt/broadcast` | comparison | 8 × 16 | dense | correct | [passed](operator_peer/cuda__f32__gt__broadcast.json) | 0.02345 / 0.01636 | [passed](program_peer/cuda__f32__gt__broadcast.json) | 0.02855 / 0.0192 |
| `cuda/bf16/gt/broadcast` | comparison | 8 × 16 | dense | correct | [passed](operator_peer/cuda__bf16__gt__broadcast.json) | 0.02119 / 0.01597 | [passed](program_peer/cuda__bf16__gt__broadcast.json) | 0.02636 / 0.01947 |
| `cuda/f32/lt/broadcast` | comparison | 8 × 16 | dense | correct | [passed](operator_peer/cuda__f32__lt__broadcast.json) | 0.02345 / 0.0157 | [passed](program_peer/cuda__f32__lt__broadcast.json) | 0.02838 / 0.01981 |
| `cuda/bf16/lt/broadcast` | comparison | 8 × 16 | dense | correct | [passed](operator_peer/cuda__bf16__lt__broadcast.json) | 0.02102 / 0.01598 | [passed](program_peer/cuda__bf16__lt__broadcast.json) | 0.02599 / 0.01955 |
| `cuda/f32/eq/broadcast` | comparison | 8 × 16 | dense | correct | [passed](operator_peer/cuda__f32__eq__broadcast.json) | 0.02366 / 0.01553 | [passed](program_peer/cuda__f32__eq__broadcast.json) | 0.02826 / 0.01878 |
| `cuda/bf16/eq/broadcast` | comparison | 8 × 16 | dense | correct | [passed](operator_peer/cuda__bf16__eq__broadcast.json) | 0.02121 / 0.01569 | [passed](program_peer/cuda__bf16__eq__broadcast.json) | 0.02593 / 0.0193 |
| `cuda/f32/sin/large_dense` | elementwise_math | 32 × 64 | dense | correct | [passed](operator_peer/cuda__f32__sin__large_dense.json) | 0.06388 / 0.02848 | [passed](program_peer/cuda__f32__sin__large_dense.json) | 0.0692 / 0.03812 |
| `cuda/bf16/sin/large_dense` | elementwise_math | 32 × 64 | dense | correct | [passed](operator_peer/cuda__bf16__sin__large_dense.json) | 0.06177 / 0.03088 | [passed](program_peer/cuda__bf16__sin__large_dense.json) | 0.06457 / 0.03447 |
| `cuda/f32/cos/large_dense` | elementwise_math | 32 × 64 | dense | correct | [passed](operator_peer/cuda__f32__cos__large_dense.json) | 0.06456 / 0.02969 | [passed](program_peer/cuda__f32__cos__large_dense.json) | 0.06919 / 0.03773 |
| `cuda/bf16/cos/large_dense` | elementwise_math | 32 × 64 | dense | correct | [passed](operator_peer/cuda__bf16__cos__large_dense.json) | 0.06239 / 0.03086 | [passed](program_peer/cuda__bf16__cos__large_dense.json) | 0.06374 / 0.03384 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
