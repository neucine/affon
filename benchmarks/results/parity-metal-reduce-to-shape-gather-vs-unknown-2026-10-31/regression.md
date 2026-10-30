# Parity regression check

Target: `Mac16,1/26.6.2/arm64`

- Cells compared: 4
- Added cells: 2
- Removed cells: 32
- Changed case definitions: 0
- New correctness failures: 0
- Diagnostic timing suspects: 0
- Target matches: yes
- Measurement protocol matches: yes

Timing changes are diagnostic and do not fail this check. Timing is compared only when target and protocol match. No timing baseline is accepted by this comparison.

| Cell | Correctness regression | Operator relative timing | Program relative timing |
| --- | --- | ---: | ---: |
| `metal/f32/reduce_to_shape/long_axis` | none | 0.81× | 1.18× |
| `metal/f32/reduce_to_shape/outer_axis` | none | 0.76× | 0.88× |
| `metal/f32/reduce_to_shape/ragged` | none | 1.03× | 1.11× |
| `metal/f32/reduce_to_shape/short_axis` | none | 0.79× | 0.83× |

## Added cells

- `metal/f32/reduce_to_shape/scalar`
- `metal/f32/reduce_to_shape/strided`

## Removed cells

- `metal/bf16/cross_entropy/large`
- `metal/bf16/cross_entropy/small`
- `metal/bf16/cross_entropy/strided`
- `metal/bf16/log_softmax_nll/large`
- `metal/bf16/log_softmax_nll/small`
- `metal/bf16/log_softmax_nll/strided`
- `metal/bf16/quantized_matmul/decode_m1`
- `metal/bf16/quantized_matmul/matrix`
- `metal/bf16/quantized_matmul/ragged`
- `metal/bf16/quantized_matmul/small_batch`
- `metal/bf16/quantized_matmul/transposed`
- `metal/bf16/scatter_add/large`
- `metal/bf16/scatter_add/small`
- `metal/bf16/scatter_add/strided`
- `metal/bf16/topk/large`
- `metal/bf16/topk/small`
- `metal/f32/cross_entropy/large`
- `metal/f32/cross_entropy/small`
- `metal/f32/cross_entropy/strided`
- `metal/f32/log_softmax_nll/large`
- `metal/f32/log_softmax_nll/small`
- `metal/f32/log_softmax_nll/strided`
- `metal/f32/quantized_matmul/decode_m1`
- `metal/f32/quantized_matmul/matrix`
- `metal/f32/quantized_matmul/ragged`
- `metal/f32/quantized_matmul/small_batch`
- `metal/f32/quantized_matmul/transposed`
- `metal/f32/scatter_add/large`
- `metal/f32/scatter_add/small`
- `metal/f32/scatter_add/strided`
- `metal/f32/topk/large`
- `metal/f32/topk/small`
