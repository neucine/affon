# Compute Kernel Matrix

This page documents forward operation coverage across CPU, Metal, and CUDA.

Last updated: 2026-09-05

## CUDA coverage

CUDA runs on Linux with an NVIDIA driver, NVRTC, and cuBLAS. The current math
and training path uses `f32`; `f64` storage support does not imply `f64` math
support. Unsupported dtype combinations report `ExecutionNotImplemented`.

| Family | CUDA support | Limits |
| --- | --- | --- |
| Allocation, transfers, contiguous views, cat, stack, slice | `f32`, `f64`, `i64` | View materialization rank <= 8; signed strides supported; slice uses the shared positive-step contract |
| Arithmetic and unary activations | `f32` | Dense and broadcast arithmetic; GELU and its derivative match the CPU tanh approximation |
| Comparison and where | `f32`, `f64`, `i64` | Dense/broadcast; comparison outputs `i64`; integer selection preserves all 64 bits |
| Masked fill | `f32` values, `i64` masks | Dense/broadcast; floating masks are normalized by the runtime |
| Cast | All pairs among `f32`, `f64`, `i64` | Invalid or out-of-range integer casts reject; index normalization also requires integral values |
| Reductions, softmax, log-softmax, layer/RMS norm | `f32` | Argmin/argmax output `i64`; log-softmax uses stable log-sum-exp; whole-tensor reductions use a parallel block |
| Dot and matmul | `f32`, cuBLAS | Batched/broadcast matmul; incompatible views are packed; zero inner dimension produces zeros |
| Embedding, gather, index-select, scatter-add | `f32` values, `i64` indices | Bounds checked; scatter-add uses atomic additions |
| One-hot and top-k | `f32` output/values, `i64` indices | Top-k requires `1 <= k <= 64`; ties preserve original index order |
| Cross-entropy | `f32` | Weighted/one-hot targets, indexed forward/backward, transposed indexed logits; row losses computed in parallel |
| Optimizer updates | `f32` | In-place muladd, axpy, subtraction, scale, Adam/AdamW; gradient clipping reduces on device |
| Graph fused dispatch | `f32` | Matmul+bias(+GELU), attention scores, add+layer-norm, unary chains, binary+unary chains use staged CUDA kernels |

CUDA graph fusion currently groups multiple kernel launches; it does not imply a
single fused GPU kernel. One CUDA ordinal is selected for the process before
initialization. Simultaneous multi-device execution is not supported.

Validation on RTX 3090 includes CPU/GPU parity for optimizer steps, losses and
backward, fused dispatch, casts, views, infinite top-k values, and empty matmul.
Run the canonical Program suites with a CUDA runtime device to validate the
public execution path.

The tables below retain the detailed CPU and Metal comparison.

Legend:

- ✅ native kernel path
- 🧱 copy-style native path (no typed math kernel)
- ➖ not applicable for this op/dtype contract
- ❌ real backend gap (`ExecutionNotImplemented`)

## Broadcast Contract

AFFON broadcast semantics follow NumPy/PyTorch for elementwise and `where`-style ops:

- right-aligned (trailing-dimension) comparison
- dimensions are compatible when equal or one side is `1`
- result dim is `max(lhs_dim, rhs_dim)` per aligned axis
- incompatible aligned dims are `ShapeMismatch`

Execution contract:

- broadcast is stride-based (broadcasted axis uses stride `0`)
- no logical broadcast tensor is materialized
- kernels/read paths index inputs through normalized broadcast strides
- this contract applies to CPU and Metal paths where broadcast coverage is marked

## Binary / Unary / Elementwise

| Op | CPU f32 | CPU f64 | CPU i64 | Metal f32 | Metal f64 | Metal i64 | Broadcast | Rank (Metal) | Key Limits (Metal) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| add | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | right-aligned, per-axis equal or one side is `1` (stride-based) | <= 8 | - |
| sub | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | right-aligned, per-axis equal or one side is `1` (stride-based) | <= 8 | - |
| mul | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | right-aligned, per-axis equal or one side is `1` (stride-based) | <= 8 | - |
| div | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | right-aligned, per-axis equal or one side is `1` (stride-based) | <= 8 | - |
| abs | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | any | - | - |
| neg | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | any | - | - |
| sign | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | any | - | - |
| relu | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | any | - | - |
| exp | ✅ | ✅ | ➖ | ✅ | ❌ | ➖ | any | - | - |
| log | ✅ | ✅ | ➖ | ✅ | ❌ | ➖ | any | - | - |
| sqrt | ✅ | ✅ | ➖ | ✅ | ❌ | ➖ | any | - | - |
| sigmoid | ✅ | ✅ | ➖ | ✅ | ❌ | ➖ | any | - | - |
| tanh | ✅ | ✅ | ➖ | ✅ | ❌ | ➖ | any | - | - |
| gelu | ✅ | ✅ | ➖ | ✅ | ❌ | ➖ | any | - | - |
| clamp | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | any | - | - |
| where | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | right-aligned, per-axis equal or one side is `1` (stride-based) | <= 8 | Metal broadcast fast path is `f32`; other broadcast cases may fall back |
| masked_fill | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | right-aligned, per-axis equal or one side is `1` | - | mask must use i64 semantics on Metal |

## Reduction / Softmax

| Op | CPU f32 | CPU f64 | CPU i64 | Metal f32 | Metal f64 | Metal i64 | Broadcast | Rank (Metal) | Key Limits (Metal) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sum_all | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | N/A | any | - |
| min_all | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | N/A | any | - |
| max_all | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | N/A | any | - |
| argmin_all | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | N/A | any | output index is i64 |
| argmax_all | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | N/A | any | output index is i64 |
| mean_all | ✅ | ✅ | ➖ | ✅ | ❌ | ➖ | N/A | any | - |
| variance_all | ✅ | ✅ | ➖ | ✅ | ❌ | ➖ | N/A | any | - |
| std_all | ✅ | ✅ | ➖ | ✅ | ❌ | ➖ | N/A | any | - |
| sum_axis | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | N/A | <= 8 | - |
| min_axis | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | N/A | <= 8 | - |
| max_axis | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | N/A | <= 8 | - |
| argmin_axis | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | N/A | <= 8 | output index is i64 |
| argmax_axis | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | N/A | <= 8 | output index is i64 |
| mean_axis | ✅ | ✅ | ➖ | ✅ | ❌ | ➖ | N/A | <= 8 | - |
| variance_axis | ✅ | ✅ | ➖ | ✅ | ❌ | ➖ | N/A | <= 8 | - |
| std_axis | ✅ | ✅ | ➖ | ✅ | ❌ | ➖ | N/A | <= 8 | - |
| softmax | ✅ | ✅ | ➖ | ✅ | ❌ | ➖ | N/A | <= 8 | - |
| log_softmax_nll | ✅ | ✅ | ➖ | ✅ | ❌ | ➖ | N/A | <= 8 | one-hot target shape parity; axis in range |
| cross_entropy | ✅ | ✅ | ➖ | ✅ | ❌ | ➖ | N/A | <= 8 | alias semantic over log_softmax_nll contract |

## Linear Algebra / Indexing / Shape

| Op | CPU f32 | CPU f64 | CPU i64 | Metal f32 | Metal f64 | Metal i64 | Broadcast | Rank (Metal) | Key Limits (Metal) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dot | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | N/A | 1D | - |
| matmul | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | batch dims broadcast (NumPy/PyTorch matmul semantics) | 1D/2D/batched (batch dims <= 8) | broadcasted batch compatibility required |
| gather | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | N/A | <= 8 | index path is i64 |
| embedding | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | N/A | <= 8 | lookup on axis 0 with i64 indices |
| index_select | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | N/A | <= 8 | index path is i64 |
| topk | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | N/A | <= 8 | k <= 64 |
| one_hot | ✅ | ➖ | ➖ | ✅ | ➖ | ➖ | N/A | any | index values must be in bounds |
| contiguous | ✅ | ➖ | ➖ | ✅ | ➖ | ✅ | N/A | any | - |
| slice | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | N/A | <= 8 | - |
| cat | ✅ | ✅ | ✅ | ✅ | 🧱 | ✅ | N/A | any | same-rank shape compatibility; concat axis varies |
| stack | ✅ | ✅ | ✅ | ✅ | 🧱 | ✅ | N/A | any | all input shapes equal; new axis inserted |

## Normalization / Composite Fused Kernels

| Op | CPU f32 | CPU f64 | CPU i64 | Metal f32 | Metal f64 | Metal i64 | Rank (Metal) | Key Limits (Metal) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| layer_norm | ✅ | ✅ | ➖ | ✅ | ❌ | ➖ | <= 8 | `eps > 0`, axis in range |
| rms_norm | ✅ | ✅ | ➖ | ✅ | ❌ | ➖ | <= 8 | `eps > 0`, axis in range |
| add_layer_norm | ✅ | ✅ | ➖ | ✅ | ❌ | ➖ | <= 8 | `eps > 0`, axis in range |

Notes:

- Metal `topk` is currently bounded by `k <= 64`.
- Many Metal ND kernels are bounded by rank `<= 8`.
- `one_hot` output is `f32`; index input is integer-like.
- `contiguous` row marks typed-data path coverage where relevant.
- Parity audit (2026-05-14): docs previously mentioned Metal `f64` matmul fallback and float index dtypes for selection; current compute contract is explicit `ExecutionNotImplemented` for Metal `f64`, and `i64` index dtype for `one_hot`/`gather`/`index_select`.

## Graph Fusion Matrix

Status terms:

- `none`: no dedicated fused kernel path
- `staged`: single fused boundary but internally chained backend-native kernels
- `monolithic`: dedicated fused kernel implementation

| Pattern | Graph pattern match ready (Graph Plan/Runner) | Fused dispatch API ready (Kernel Dispatch) | CPU dedicated fused kernel (CPU Kernel) | Metal dedicated fused kernel (Metal Kernel) | Contract tests (Graph/Execution/Kernel) | Perf validated (Bench) | Production-ready |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `binary -> unary` | yes | yes (`binaryThenUnaryChain`) | monolithic | monolithic | partial | no | no |
| `unary -> unary` | yes | yes (`unaryChain`) | monolithic | monolithic | partial | no | no |
| `matmul -> add` | yes | yes (`matmulAdd`) | monolithic | monolithic | partial | no | no |
| `matmul -> add -> gelu` | yes | yes (`matmulAddGelu`) | monolithic | monolithic (`f32`) | partial | no | no |
| `add -> layer_norm` | yes | yes (`addLayerNorm`) | monolithic | monolithic | partial | no | no |
| `matmul -> mul(scale) -> masked_fill -> softmax` | yes | yes (`attentionScores`) | monolithic | monolithic (`f32`) | partial | no | no |
| `max -> sub -> exp -> sum -> log -> sub -> mul -> sum -> neg -> mean` | yes | yes (`logSoftmaxNll` boundary) | monolithic (`log_softmax_nll`) | monolithic (`f32`) | partial | no | no |
| `gather -> max -> sub -> exp -> sum -> log -> add -> sub -> mean` | yes | yes (`logSoftmaxNll` boundary + one_hot bridge) | monolithic (`log_softmax_nll`) | monolithic (`f32`) | partial | no | no |
| `slice(target[:,1:]) -> reshape -> reshape(logits) -> gather -> logsumexp-loss` | yes | yes (`logSoftmaxNll` boundary + one_hot bridge) | monolithic (`log_softmax_nll`) | monolithic (`f32`) | partial | no | no |
