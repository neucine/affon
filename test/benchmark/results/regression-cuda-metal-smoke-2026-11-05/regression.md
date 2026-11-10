# Parity regression check

Target: `NVIDIA GeForce RTX 3090/7.0.0-38-generic/x86_64`

- Cells compared: 142
- Added cells: 0
- Removed cells: 0
- Changed case definitions: 0
- New correctness failures: 0
- Diagnostic timing suspects: 0
- Target matches: yes
- Measurement protocol matches: yes

Timing changes are diagnostic and do not fail this check. Timing is compared only when target and protocol match. No timing baseline is accepted by this comparison.

| Cell | Correctness regression | Operator relative timing | Program relative timing |
| --- | --- | ---: | ---: |
| `cuda/bf16/abs/large_dense` | none | 1.02× | 0.91× |
| `cuda/bf16/abs/offset` | none | 1.17× | 1.00× |
| `cuda/bf16/abs/strided` | none | 1.02× | 0.98× |
| `cuda/bf16/add/broadcast` | none | 0.99× | 1.00× |
| `cuda/bf16/argmax_all/long_axis` | none | 0.91× | 1.03× |
| `cuda/bf16/argmax_all/outer_axis` | none | 1.00× | 1.01× |
| `cuda/bf16/argmax_axis/long_axis` | none | 1.03× | 1.00× |
| `cuda/bf16/argmin_all/long_axis` | none | 0.98× | 0.99× |
| `cuda/bf16/argmin_axis/long_axis` | none | 1.05× | 0.97× |
| `cuda/bf16/cast/large_dense` | none | 0.83× | 1.14× |
| `cuda/bf16/cat/materialized` | none | 0.99× | 0.99× |
| `cuda/bf16/clamp/large_dense` | none | 1.00× | 1.00× |
| `cuda/bf16/contiguous/materialized` | none | 1.16× | 1.05× |
| `cuda/bf16/cos/large_dense` | none | — | — |
| `cuda/bf16/cross_entropy/batched` | none | 0.97× | 1.00× |
| `cuda/bf16/cross_entropy/batched/dense_b2_c33_s17` | none | 0.95× | 0.98× |
| `cuda/bf16/cross_entropy_indexed/batched` | none | 1.21× | 1.02× |
| `cuda/bf16/div/broadcast` | none | 0.94× | 1.00× |
| `cuda/bf16/dot/dot_or_vector` | none | 1.01× | 0.99× |
| `cuda/bf16/embedding/batched` | none | 1.00× | 0.98× |
| `cuda/bf16/eq/broadcast` | none | 1.00× | 1.02× |
| `cuda/bf16/erf/large_dense` | none | 0.98× | 0.98× |
| `cuda/bf16/exp/large_dense` | none | 1.02× | 0.99× |
| `cuda/bf16/gather/batched` | none | 0.99× | 1.01× |
| `cuda/bf16/gelu/large_dense` | none | 1.03× | 0.99× |
| `cuda/bf16/gt/broadcast` | none | 1.03× | 1.00× |
| `cuda/bf16/index_select/batched` | none | 1.00× | 0.99× |
| `cuda/bf16/layer_norm/long_row` | none | 0.99× | 0.98× |
| `cuda/bf16/log/large_dense` | none | 0.92× | 1.00× |
| `cuda/bf16/log_softmax/long_row` | none | 1.00× | 0.98× |
| `cuda/bf16/log_softmax_nll/batched` | none | 1.04× | 1.03× |
| `cuda/bf16/lt/broadcast` | none | 1.00× | 1.00× |
| `cuda/bf16/masked_fill/broadcast` | none | 1.01× | 1.01× |
| `cuda/bf16/matmul/batched` | none | 1.01× | 0.99× |
| `cuda/bf16/matmul/transposed` | none | 0.94× | 0.96× |
| `cuda/bf16/max_all/long_axis` | none | 1.06× | 1.03× |
| `cuda/bf16/max_axis/long_axis` | none | 1.01× | 0.92× |
| `cuda/bf16/mean_all/long_axis` | none | 1.03× | 0.99× |
| `cuda/bf16/mean_axis/long_axis` | none | 1.00× | 0.98× |
| `cuda/bf16/min_all/long_axis` | none | 1.07× | 1.04× |
| `cuda/bf16/min_axis/long_axis` | none | 0.99× | 0.99× |
| `cuda/bf16/mul/broadcast` | none | 1.00× | 0.95× |
| `cuda/bf16/neg/large_dense` | none | 1.09× | 0.98× |
| `cuda/bf16/permute/materialized` | none | 0.99× | 0.99× |
| `cuda/bf16/quantized_matmul/decode_m1` | none | — | — |
| `cuda/bf16/relu/large_dense` | none | 1.08× | 0.99× |
| `cuda/bf16/reshape/materialized` | none | 0.99× | 0.96× |
| `cuda/bf16/rms_norm/long_row` | none | 1.03× | 0.99× |
| `cuda/bf16/scatter_add/batched` | none | 1.37× | 1.01× |
| `cuda/bf16/sigmoid/large_dense` | none | 0.89× | 0.96× |
| `cuda/bf16/sign/large_dense` | none | 0.92× | 1.11× |
| `cuda/bf16/silu/large_dense` | none | 0.99× | 1.00× |
| `cuda/bf16/sin/large_dense` | none | — | — |
| `cuda/bf16/slice/materialized` | none | 0.98× | 0.98× |
| `cuda/bf16/softmax/long_row` | none | 1.01× | 0.98× |
| `cuda/bf16/sqrt/large_dense` | none | 0.91× | 0.92× |
| `cuda/bf16/squeeze/materialized` | none | 1.00× | 1.00× |
| `cuda/bf16/stack/materialized` | none | 1.01× | 0.96× |
| `cuda/bf16/std_all/long_axis` | none | 1.00× | 1.00× |
| `cuda/bf16/std_axis/long_axis` | none | 0.89× | 0.99× |
| `cuda/bf16/sub/broadcast` | none | 1.00× | 0.94× |
| `cuda/bf16/sum_all/long_axis` | none | 1.02× | 1.09× |
| `cuda/bf16/sum_axis/long_axis` | none | 1.02× | 1.00× |
| `cuda/bf16/tanh/large_dense` | none | 0.97× | 0.96× |
| `cuda/bf16/topk/batched` | none | 0.98× | 0.95× |
| `cuda/bf16/transpose/materialized` | none | 1.04× | 1.00× |
| `cuda/bf16/unsqueeze/materialized` | none | 0.94× | 1.06× |
| `cuda/bf16/variance_all/long_axis` | none | 0.95× | 0.99× |
| `cuda/bf16/variance_axis/long_axis` | none | 1.03× | 0.97× |
| `cuda/bf16/where/broadcast` | none | 1.00× | 0.96× |
| `cuda/f32/abs/large_dense` | none | 0.78× | 1.06× |
| `cuda/f32/abs/offset` | none | 0.98× | 1.00× |
| `cuda/f32/abs/strided` | none | 1.01× | 1.02× |
| `cuda/f32/add/broadcast` | none | 1.05× | 1.00× |
| `cuda/f32/argmax_all/long_axis` | none | 1.00× | 0.96× |
| `cuda/f32/argmax_all/outer_axis` | none | 1.00× | 1.00× |
| `cuda/f32/argmax_axis/long_axis` | none | 1.05× | 0.96× |
| `cuda/f32/argmin_all/long_axis` | none | 1.03× | 0.98× |
| `cuda/f32/argmin_axis/long_axis` | none | 0.94× | 0.99× |
| `cuda/f32/cast/large_dense` | none | 0.94× | 1.04× |
| `cuda/f32/cat/materialized` | none | 0.88× | 0.97× |
| `cuda/f32/clamp/large_dense` | none | 0.94× | 0.95× |
| `cuda/f32/contiguous/materialized` | none | 0.92× | 1.04× |
| `cuda/f32/cos/large_dense` | none | — | — |
| `cuda/f32/cross_entropy/batched` | none | 1.04× | 0.98× |
| `cuda/f32/cross_entropy/batched/dense_b2_c33_s17` | none | 0.88× | 0.98× |
| `cuda/f32/cross_entropy_indexed/batched` | none | 1.02× | 0.99× |
| `cuda/f32/div/broadcast` | none | 1.00× | 0.97× |
| `cuda/f32/dot/dot_or_vector` | none | 0.98× | 1.01× |
| `cuda/f32/embedding/batched` | none | 0.96× | 1.06× |
| `cuda/f32/eq/broadcast` | none | 1.02× | 1.04× |
| `cuda/f32/erf/large_dense` | none | 0.96× | 1.13× |
| `cuda/f32/exp/large_dense` | none | 0.82× | 1.15× |
| `cuda/f32/gather/batched` | none | 1.02× | 1.08× |
| `cuda/f32/gelu/large_dense` | none | 0.95× | 0.96× |
| `cuda/f32/gt/broadcast` | none | 1.00× | 1.04× |
| `cuda/f32/index_select/batched` | none | 0.99× | 1.01× |
| `cuda/f32/layer_norm/long_row` | none | — | — |
| `cuda/f32/log/large_dense` | none | 0.92× | 0.99× |
| `cuda/f32/log_softmax/long_row` | none | 0.92× | 1.02× |
| `cuda/f32/log_softmax_nll/batched` | none | 1.04× | 0.99× |
| `cuda/f32/lt/broadcast` | none | 1.02× | 1.02× |
| `cuda/f32/masked_fill/broadcast` | none | 0.99× | 1.04× |
| `cuda/f32/matmul/batched` | none | 1.01× | 0.99× |
| `cuda/f32/matmul/transposed` | none | 0.84× | 1.04× |
| `cuda/f32/max_all/long_axis` | none | 1.02× | 1.02× |
| `cuda/f32/max_axis/long_axis` | none | 0.97× | 0.80× |
| `cuda/f32/mean_all/long_axis` | none | 1.03× | 1.03× |
| `cuda/f32/mean_axis/long_axis` | none | 0.97× | 1.03× |
| `cuda/f32/min_all/long_axis` | none | 1.03× | 1.03× |
| `cuda/f32/min_axis/long_axis` | none | 1.01× | 0.97× |
| `cuda/f32/mul/broadcast` | none | 1.03× | 0.96× |
| `cuda/f32/neg/large_dense` | none | 0.96× | 0.94× |
| `cuda/f32/one_hot/batched` | none | 1.00× | 0.97× |
| `cuda/f32/permute/materialized` | none | 1.02× | 0.96× |
| `cuda/f32/quantized_matmul/decode_m1` | none | — | — |
| `cuda/f32/reduce_to_shape/long_axis` | none | — | — |
| `cuda/f32/relu/large_dense` | none | 0.95× | 1.05× |
| `cuda/f32/reshape/materialized` | none | 1.03× | 1.00× |
| `cuda/f32/rms_norm/long_row` | none | — | — |
| `cuda/f32/scatter_add/batched` | none | 1.03× | 1.05× |
| `cuda/f32/sigmoid/large_dense` | none | 0.80× | 0.93× |
| `cuda/f32/sign/large_dense` | none | 0.94× | 0.95× |
| `cuda/f32/silu/large_dense` | none | 0.98× | 1.11× |
| `cuda/f32/sin/large_dense` | none | — | — |
| `cuda/f32/slice/materialized` | none | 1.07× | 1.00× |
| `cuda/f32/softmax/long_row` | none | 1.02× | 1.02× |
| `cuda/f32/sqrt/large_dense` | none | 0.98× | 1.14× |
| `cuda/f32/squeeze/materialized` | none | 1.06× | 1.02× |
| `cuda/f32/stack/materialized` | none | 1.00× | 1.12× |
| `cuda/f32/std_all/long_axis` | none | 1.03× | 1.01× |
| `cuda/f32/std_axis/long_axis` | none | 1.00× | 1.01× |
| `cuda/f32/sub/broadcast` | none | 1.04× | 1.04× |
| `cuda/f32/sum_all/long_axis` | none | 1.01× | 1.04× |
| `cuda/f32/sum_axis/long_axis` | none | 1.05× | 1.02× |
| `cuda/f32/tanh/large_dense` | none | 1.11× | 0.83× |
| `cuda/f32/topk/batched` | none | 1.00× | 1.04× |
| `cuda/f32/transpose/materialized` | none | 0.98× | 1.01× |
| `cuda/f32/unsqueeze/materialized` | none | 1.04× | 1.12× |
| `cuda/f32/variance_all/long_axis` | none | 1.00× | 1.04× |
| `cuda/f32/variance_axis/long_axis` | none | 1.05× | 0.99× |
| `cuda/f32/where/broadcast` | none | 0.98× | 1.00× |
