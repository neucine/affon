# Parity batch report

Target: `NVIDIA GeForce RTX 3090/7.0.0-38-generic/x86_64`  
Catalog SHA-256: `49b6a93871fe8587314da62dacc4e8895896e80ea7b68d91d2e3b6ff8ecde7f7`  
Binary SHA-256: `f3b481e37eb9d5a16e7a2208f85cf952f644d5b35102d5e989ddacefb345641b`  
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
| `cuda/f32/matmul/batched` | matrix | 4 × 8 × 16 × 12 | dense | correct | [passed](operator_peer/cuda__f32__matmul__batched.json) | 0.04221 / 0.02024 | [passed](program_peer/cuda__f32__matmul__batched.json) | 0.04904 / 0.02569 |
| `cuda/bf16/matmul/batched` | matrix | 4 × 8 × 16 × 12 | dense | correct | [passed](operator_peer/cuda__bf16__matmul__batched.json) | 0.04567 / 0.02114 | [passed](program_peer/cuda__bf16__matmul__batched.json) | 0.04924 / 0.02573 |
| `cuda/f32/index_select/batched` | indexing | 3 × 7 × 11 | dense | correct | [passed](operator_peer/cuda__f32__index_select__batched.json) | 0.02442 / 0.01306 | [passed](program_peer/cuda__f32__index_select__batched.json) | 0.0294 / 0.01658 |
| `cuda/bf16/index_select/batched` | indexing | 3 × 7 × 11 | dense | correct | [passed](operator_peer/cuda__bf16__index_select__batched.json) | 0.02866 / 0.01326 | [passed](program_peer/cuda__bf16__index_select__batched.json) | 0.03375 / 0.01614 |
| `cuda/f32/gather/batched` | indexing | 16 × 32 | dense | correct | [passed](operator_peer/cuda__f32__gather__batched.json) | 0.02344 / 0.01422 | [passed](program_peer/cuda__f32__gather__batched.json) | 0.02908 / 0.01791 |
| `cuda/bf16/gather/batched` | indexing | 16 × 32 | dense | correct | [passed](operator_peer/cuda__bf16__gather__batched.json) | 0.03008 / 0.01482 | [passed](program_peer/cuda__bf16__gather__batched.json) | 0.03544 / 0.01738 |
| `cuda/f32/layer_norm/long_row` | normalization | 128 × 512 | dense | correct | [passed](operator_peer/cuda__f32__layer_norm__long_row.json) | 1.101 / 0.5522 | [passed](program_peer/cuda__f32__layer_norm__long_row.json) | 1.116 / 0.5607 |
| `cuda/bf16/layer_norm/long_row` | normalization | 128 × 512 | dense | correct | [passed](operator_peer/cuda__bf16__layer_norm__long_row.json) | 1.095 / 0.5591 | [passed](program_peer/cuda__bf16__layer_norm__long_row.json) | 1.109 / 0.5614 |
| `cuda/f32/one_hot/batched/b1_r16_c17_k2` | indexing | 1 × 16 × 17 | dense | correct | [passed](operator_peer/cuda__f32__one_hot__batched__b1_r16_c17_k2.json) | 0.06594 / 0.03091 | [passed](program_peer/cuda__f32__one_hot__batched__b1_r16_c17_k2.json) | 0.07139 / 0.03421 |
| `cuda/f32/softmax/long_row` | normalization | 128 × 512 | dense | correct | [passed](operator_peer/cuda__f32__softmax__long_row.json) | 1.101 / 0.5524 | [passed](program_peer/cuda__f32__softmax__long_row.json) | 1.108 / 0.5548 |
| `cuda/bf16/softmax/long_row` | normalization | 128 × 512 | dense | correct | [passed](operator_peer/cuda__bf16__softmax__long_row.json) | 1.095 / 0.5487 | [passed](program_peer/cuda__bf16__softmax__long_row.json) | 1.089 / 0.6077 |
| `cuda/f32/rms_norm/long_row` | normalization | 128 × 512 | dense | correct | [passed](operator_peer/cuda__f32__rms_norm__long_row.json) | 1.107 / 0.5591 | [passed](program_peer/cuda__f32__rms_norm__long_row.json) | 1.119 / 0.6058 |
| `cuda/bf16/rms_norm/long_row` | normalization | 128 × 512 | dense | correct | [passed](operator_peer/cuda__bf16__rms_norm__long_row.json) | 1.1 / 0.5688 | [passed](program_peer/cuda__bf16__rms_norm__long_row.json) | 1.088 / 0.6164 |
| `cuda/f32/log_softmax/long_row` | normalization | 128 × 512 | dense | correct | [passed](operator_peer/cuda__f32__log_softmax__long_row.json) | 1.09 / 0.5897 | [passed](program_peer/cuda__f32__log_softmax__long_row.json) | 1.104 / 0.5548 |
| `cuda/bf16/log_softmax/long_row` | normalization | 128 × 512 | dense | correct | [passed](operator_peer/cuda__bf16__log_softmax__long_row.json) | 1.078 / 0.5619 | [passed](program_peer/cuda__bf16__log_softmax__long_row.json) | 1.089 / 0.6232 |
| `cuda/f32/topk/batched/dense_axis1_b2_r16_c129_k8` | indexing | 2 × 16 × 129 | dense | correct | [passed](operator_peer/cuda__f32__topk__batched__dense_axis1_b2_r16_c129_k8.json) | 0.1104 / 0.08441 | [passed](program_peer/cuda__f32__topk__batched__dense_axis1_b2_r16_c129_k8.json) | 0.1165 / 0.08521 |
| `cuda/bf16/topk/batched/dense_axis1_b2_r16_c129_k8` | indexing | 2 × 16 × 129 | dense | correct | [passed](operator_peer/cuda__bf16__topk__batched__dense_axis1_b2_r16_c129_k8.json) | 0.1222 / 0.08381 | [passed](program_peer/cuda__bf16__topk__batched__dense_axis1_b2_r16_c129_k8.json) | 0.1244 / 0.08032 |
| `cuda/f32/mean_axis/long_axis` | reduction | 128 × 768 | dense | correct | [passed](operator_peer/cuda__f32__mean_axis__long_axis.json) | 0.02003 / 0.01476 | [passed](program_peer/cuda__f32__mean_axis__long_axis.json) | 0.02518 / 0.01681 |
| `cuda/bf16/mean_axis/long_axis` | reduction | 128 × 768 | dense | correct | [passed](operator_peer/cuda__bf16__mean_axis__long_axis.json) | 0.02551 / 0.01347 | [passed](program_peer/cuda__bf16__mean_axis__long_axis.json) | 0.02745 / 0.0162 |
| `cuda/f32/sum_axis/long_axis` | reduction | 128 × 768 | dense | correct | [passed](operator_peer/cuda__f32__sum_axis__long_axis.json) | 0.01994 / 0.01394 | [passed](program_peer/cuda__f32__sum_axis__long_axis.json) | 0.02483 / 0.0163 |
| `cuda/bf16/sum_axis/long_axis` | reduction | 128 × 768 | dense | correct | [passed](operator_peer/cuda__bf16__sum_axis__long_axis.json) | 0.02517 / 0.01303 | [passed](program_peer/cuda__bf16__sum_axis__long_axis.json) | 0.02755 / 0.01602 |
| `cuda/f32/argmax_axis/long_axis/r128_c31` | reduction | 128 × 31 | dense | correct | [passed](operator_peer/cuda__f32__argmax_axis__long_axis__r128_c31.json) | 0.3779 / 0.01366 | [passed](program_peer/cuda__f32__argmax_axis__long_axis__r128_c31.json) | 0.3822 / 0.01583 |
| `cuda/bf16/argmax_axis/long_axis/r128_c31` | reduction | 128 × 31 | dense | correct | [passed](operator_peer/cuda__bf16__argmax_axis__long_axis__r128_c31.json) | 0.02867 / 0.01328 | [passed](program_peer/cuda__bf16__argmax_axis__long_axis__r128_c31.json) | 0.03332 / 0.01677 |
| `cuda/f32/argmin_axis/long_axis/r128_c31` | reduction | 128 × 31 | dense | correct | [passed](operator_peer/cuda__f32__argmin_axis__long_axis__r128_c31.json) | 0.3778 / 0.0125 | [passed](program_peer/cuda__f32__argmin_axis__long_axis__r128_c31.json) | 0.3825 / 0.01649 |
| `cuda/bf16/argmin_axis/long_axis/r128_c31` | reduction | 128 × 31 | dense | correct | [passed](operator_peer/cuda__bf16__argmin_axis__long_axis__r128_c31.json) | 0.02831 / 0.01343 | [passed](program_peer/cuda__bf16__argmin_axis__long_axis__r128_c31.json) | 0.0333 / 0.01657 |
| `cuda/f32/log_softmax_nll/batched` | loss | 3 × 11 × 7 | dense | correct | [passed](operator_peer/cuda__f32__log_softmax_nll__batched.json) | 0.04332 / 0.02476 | [passed](program_peer/cuda__f32__log_softmax_nll__batched.json) | 0.04852 / 0.02917 |
| `cuda/bf16/log_softmax_nll/batched/dense_b2_c33_s17` | loss | 2 × 33 × 17 | dense | correct | [passed](operator_peer/cuda__bf16__log_softmax_nll__batched__dense_b2_c33_s17.json) | 0.06034 / 0.03626 | [passed](program_peer/cuda__bf16__log_softmax_nll__batched__dense_b2_c33_s17.json) | 0.0627 / 0.03959 |
| `cuda/f32/add/broadcast` | arithmetic | 128 × 512 | dense | correct | [passed](operator_peer/cuda__f32__add__broadcast.json) | 1.092 / 0.567 | [passed](program_peer/cuda__f32__add__broadcast.json) | 1.093 / 0.5567 |
| `cuda/bf16/add/broadcast` | arithmetic | 128 × 512 | dense | correct | [passed](operator_peer/cuda__bf16__add__broadcast.json) | 1.085 / 0.5525 | [passed](program_peer/cuda__bf16__add__broadcast.json) | 1.103 / 0.5563 |
| `cuda/f32/div/broadcast` | arithmetic | 128 × 512 | dense | correct | [passed](operator_peer/cuda__f32__div__broadcast.json) | 1.102 / 0.5888 | [passed](program_peer/cuda__f32__div__broadcast.json) | 1.111 / 0.5974 |
| `cuda/bf16/div/broadcast` | arithmetic | 128 × 512 | dense | correct | [passed](operator_peer/cuda__bf16__div__broadcast.json) | 1.083 / 0.5514 | [passed](program_peer/cuda__bf16__div__broadcast.json) | 1.091 / 0.5669 |
| `cuda/f32/mul/broadcast` | arithmetic | 128 × 512 | dense | correct | [passed](operator_peer/cuda__f32__mul__broadcast.json) | 1.09 / 0.5592 | [passed](program_peer/cuda__f32__mul__broadcast.json) | 1.096 / 0.5593 |
| `cuda/bf16/mul/broadcast` | arithmetic | 128 × 512 | dense | correct | [passed](operator_peer/cuda__bf16__mul__broadcast.json) | 1.082 / 0.5882 | [passed](program_peer/cuda__bf16__mul__broadcast.json) | 1.089 / 0.5622 |
| `cuda/f32/sub/broadcast` | arithmetic | 128 × 512 | dense | correct | [passed](operator_peer/cuda__f32__sub__broadcast.json) | 1.095 / 0.6061 | [passed](program_peer/cuda__f32__sub__broadcast.json) | 1.11 / 0.5828 |
| `cuda/bf16/sub/broadcast` | arithmetic | 128 × 512 | dense | correct | [passed](operator_peer/cuda__bf16__sub__broadcast.json) | 1.101 / 0.5564 | [passed](program_peer/cuda__bf16__sub__broadcast.json) | 1.095 / 0.5632 |
| `cuda/f32/sigmoid/large_dense` | elementwise_math | 32 × 64 | dense | correct | [passed](operator_peer/cuda__f32__sigmoid__large_dense.json) | 0.0691 / 0.0289 | [passed](program_peer/cuda__f32__sigmoid__large_dense.json) | 0.0737 / 0.03766 |
| `cuda/bf16/sigmoid/large_dense` | elementwise_math | 32 × 64 | dense | correct | [passed](operator_peer/cuda__bf16__sigmoid__large_dense.json) | 0.06176 / 0.03238 | [passed](program_peer/cuda__bf16__sigmoid__large_dense.json) | 0.06443 / 0.03702 |
| `cuda/f32/tanh/large_dense` | elementwise_math | 32 × 64 | dense | correct | [passed](operator_peer/cuda__f32__tanh__large_dense.json) | 0.06858 / 0.03541 | [passed](program_peer/cuda__f32__tanh__large_dense.json) | 0.073 / 0.03846 |
| `cuda/bf16/tanh/large_dense` | elementwise_math | 32 × 64 | dense | correct | [passed](operator_peer/cuda__bf16__tanh__large_dense.json) | 0.06198 / 0.03137 | [passed](program_peer/cuda__bf16__tanh__large_dense.json) | 0.06376 / 0.03382 |
| `cuda/f32/gt/broadcast` | comparison | 8 × 16 | dense | correct | [passed](operator_peer/cuda__f32__gt__broadcast.json) | 0.02355 / 0.01578 | [passed](program_peer/cuda__f32__gt__broadcast.json) | 0.0286 / 0.01896 |
| `cuda/bf16/gt/broadcast` | comparison | 8 × 16 | dense | correct | [passed](operator_peer/cuda__bf16__gt__broadcast.json) | 0.02168 / 0.01638 | [passed](program_peer/cuda__bf16__gt__broadcast.json) | 0.0263 / 0.01969 |
| `cuda/f32/lt/broadcast` | comparison | 8 × 16 | dense | correct | [passed](operator_peer/cuda__f32__lt__broadcast.json) | 0.02386 / 0.01575 | [passed](program_peer/cuda__f32__lt__broadcast.json) | 0.02857 / 0.01957 |
| `cuda/bf16/lt/broadcast` | comparison | 8 × 16 | dense | correct | [passed](operator_peer/cuda__bf16__lt__broadcast.json) | 0.02126 / 0.01643 | [passed](program_peer/cuda__bf16__lt__broadcast.json) | 0.02629 / 0.01977 |
| `cuda/f32/eq/broadcast` | comparison | 8 × 16 | dense | correct | [passed](operator_peer/cuda__f32__eq__broadcast.json) | 0.02368 / 0.01569 | [passed](program_peer/cuda__f32__eq__broadcast.json) | 0.02849 / 0.01935 |
| `cuda/bf16/eq/broadcast` | comparison | 8 × 16 | dense | correct | [passed](operator_peer/cuda__bf16__eq__broadcast.json) | 0.02135 / 0.01663 | [passed](program_peer/cuda__bf16__eq__broadcast.json) | 0.02634 / 0.01954 |
| `cuda/f32/sin/large_dense` | elementwise_math | 32 × 64 | dense | correct | [passed](operator_peer/cuda__f32__sin__large_dense.json) | 0.06397 / 0.02887 | [passed](program_peer/cuda__f32__sin__large_dense.json) | 0.06881 / 0.03796 |
| `cuda/bf16/sin/large_dense` | elementwise_math | 32 × 64 | dense | correct | [passed](operator_peer/cuda__bf16__sin__large_dense.json) | 0.06212 / 0.03207 | [passed](program_peer/cuda__bf16__sin__large_dense.json) | 0.06381 / 0.03525 |
| `cuda/f32/cos/large_dense` | elementwise_math | 32 × 64 | dense | correct | [passed](operator_peer/cuda__f32__cos__large_dense.json) | 0.06412 / 0.03505 | [passed](program_peer/cuda__f32__cos__large_dense.json) | 0.06932 / 0.03801 |
| `cuda/bf16/cos/large_dense` | elementwise_math | 32 × 64 | dense | correct | [passed](operator_peer/cuda__bf16__cos__large_dense.json) | 0.06179 / 0.03202 | [passed](program_peer/cuda__bf16__cos__large_dense.json) | 0.06407 / 0.03403 |

Numerical results are in [capability.json](capability.json).
Timing changes are diagnostic. The saved files contain the raw checks, including failures.
