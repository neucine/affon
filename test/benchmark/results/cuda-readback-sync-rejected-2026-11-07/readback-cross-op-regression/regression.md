# Parity regression check

Target: `NVIDIA GeForce RTX 3090/7.0.0-38-generic/x86_64`

- Cells compared: 49
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
| `cuda/bf16/add/broadcast` | none | 0.95× | 0.91× |
| `cuda/bf16/argmax_axis/long_axis/r128_c31` | none | 0.98× | 1.00× |
| `cuda/bf16/argmin_axis/long_axis/r128_c31` | none | 1.02× | 1.02× |
| `cuda/bf16/cos/large_dense` | none | 1.05× | 1.00× |
| `cuda/bf16/div/broadcast` | none | 0.97× | 1.01× |
| `cuda/bf16/eq/broadcast` | none | 1.05× | 1.00× |
| `cuda/bf16/gather/batched` | none | 1.05× | 1.01× |
| `cuda/bf16/gt/broadcast` | none | 1.00× | 1.01× |
| `cuda/bf16/index_select/batched` | none | 0.99× | 1.00× |
| `cuda/bf16/layer_norm/long_row` | none | 0.97× | 0.99× |
| `cuda/bf16/log_softmax/long_row` | none | 1.01× | 1.09× |
| `cuda/bf16/log_softmax_nll/batched/dense_b2_c33_s17` | none | 1.03× | 1.02× |
| `cuda/bf16/lt/broadcast` | none | 1.02× | 1.00× |
| `cuda/bf16/matmul/batched` | none | 0.99× | 1.03× |
| `cuda/bf16/mean_axis/long_axis` | none | 1.01× | 0.98× |
| `cuda/bf16/mul/broadcast` | none | 1.06× | 0.91× |
| `cuda/bf16/rms_norm/long_row` | none | 0.94× | 1.09× |
| `cuda/bf16/sigmoid/large_dense` | none | 1.16× | 1.09× |
| `cuda/bf16/sin/large_dense` | none | 1.03× | 1.03× |
| `cuda/bf16/softmax/long_row` | none | 1.00× | 1.10× |
| `cuda/bf16/sub/broadcast` | none | 0.97× | 0.91× |
| `cuda/bf16/sum_axis/long_axis` | none | 0.99× | 0.99× |
| `cuda/bf16/tanh/large_dense` | none | 1.13× | 1.01× |
| `cuda/bf16/topk/batched/dense_axis1_b2_r16_c129_k8` | none | 1.02× | 1.06× |
| `cuda/f32/add/broadcast` | none | 1.04× | 1.00× |
| `cuda/f32/argmax_axis/long_axis/r128_c31` | none | 1.03× | 0.96× |
| `cuda/f32/argmin_axis/long_axis/r128_c31` | none | 0.94× | 0.99× |
| `cuda/f32/cos/large_dense` | none | 1.19× | 1.01× |
| `cuda/f32/div/broadcast` | none | 1.00× | 1.06× |
| `cuda/f32/eq/broadcast` | none | 1.01× | 1.02× |
| `cuda/f32/gather/batched` | none | 1.00× | 0.99× |
| `cuda/f32/gt/broadcast` | none | 0.96× | 0.99× |
| `cuda/f32/index_select/batched` | none | 1.01× | 1.00× |
| `cuda/f32/layer_norm/long_row` | none | 0.96× | 0.97× |
| `cuda/f32/log_softmax/long_row` | none | 1.05× | 0.99× |
| `cuda/f32/log_softmax_nll/batched` | none | 0.99× | 1.01× |
| `cuda/f32/lt/broadcast` | none | 0.99× | 0.98× |
| `cuda/f32/matmul/batched` | none | 0.99× | 1.05× |
| `cuda/f32/mean_axis/long_axis` | none | 1.00× | 1.04× |
| `cuda/f32/mul/broadcast` | none | 0.94× | 0.93× |
| `cuda/f32/one_hot/batched/b1_r16_c17_k2` | none | 1.01× | 0.99× |
| `cuda/f32/rms_norm/long_row` | none | 1.00× | 1.06× |
| `cuda/f32/sigmoid/large_dense` | none | 0.83× | 1.00× |
| `cuda/f32/sin/large_dense` | none | 1.01× | 1.00× |
| `cuda/f32/softmax/long_row` | none | 0.98× | 0.99× |
| `cuda/f32/sub/broadcast` | none | 1.09× | 1.04× |
| `cuda/f32/sum_axis/long_axis` | none | 0.93× | 1.00× |
| `cuda/f32/tanh/large_dense` | none | 1.22× | 1.04× |
| `cuda/f32/topk/batched/dense_axis1_b2_r16_c129_k8` | none | 1.01× | 1.05× |
