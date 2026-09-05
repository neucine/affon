# Compute Semantic Coverage

This page tracks Affon's compute correctness evidence by semantic claim.

It is different from the [kernel matrix](./kernel-matrix.md). The kernel matrix
records backend execution coverage. This ledger records whether a behavior is
specified, tested, and proven across the execution paths that matter for
trustworthy model training.

Status terms:

- `closed`: the contract is explicit and evidence covers the intended
  execution, layout, and backend class.
- `sampled`: related tests exist, but the semantic class is not fully closed.
- `open`: contract or evidence is missing.
- `unsupported`: Affon intentionally rejects the case with a specified error or
  documented limit.

Evidence levels, from strongest foundation to broadest workload signal:

1. explicit semantic contract
2. primitive hostile-layout tests
3. primitive Torch parity
4. eager / graph / compile equivalence tests
5. backend parity and backend-limit tests
6. composed motif parity
7. module-level parity
8. tiny deterministic training canaries
9. longer workload runs

Longer training runs are useful canaries, but they do not close primitive
semantic claims by themselves.

## Initial Decoder-LM Slice

This first ledger slice covers the core dependency set used by decoder-LM style
training. It should be read as an honest starting map, not a completion claim.

| Family | Public Surface | Forward Spec | Backward Spec | Layout Spec | Current Evidence | Status | Gaps |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Constructors and copy | `tensor`, `empty`, `zeros`, `ones`, `full`, `copy`, `cast`, `move` | sampled | n/a or sampled | sampled | e2e compute tests, type tests, kernel matrix | sampled | Clarify copy/view/alias behavior and device movement failure semantics in one place. |
| View and shape | `reshape`, `permute`, `transpose`, `squeeze`, `unsqueeze`, `contiguous`, `slice` | sampled | sampled | sampled | e2e shape tests, parity shape tests, hostile-layout backward regressions | sampled | Close view alias/materialization rules and compile parity for the full family. |
| Aggregation | `cat`, `stack` | sampled | sampled | sampled | e2e shape tests, parity tests, kernel matrix | sampled | Close non-contiguous input behavior, backward split routing, and graph/compile coverage. |
| Binary and broadcast | `add`, `sub`, `mul`, `div` | sampled | sampled | sampled | parity arithmetic tests, broadcast-backward tests, kernel matrix | sampled | Tie right-aligned broadcast spec to eager, graph, compile, CPU, Metal, and CUDA evidence. |
| Masking | `where`, `masked_fill` | sampled | sampled | sampled | parity selection tests, backward tests, kernel matrix | sampled | Close broadcasted mask/value combinations and unsupported dtype/device cases. |
| Reductions | `sum`, `mean`, `variance`, `std`, `min`, `max` | sampled | sampled | sampled | e2e reduction tests, parity tests, reductions-backward tests, kernel matrix | sampled | Close keepdim/axis/layout matrix and compile parity. |
| Activations | `relu`, `gelu`, `sigmoid`, `tanh`, `exp`, `log`, `sqrt` | sampled | sampled | sampled | e2e activations, parity activation tests, kernel matrix | sampled | Close hostile-layout backward and graph/compile parity by op. |
| Linear algebra | `dot`, `matmul` | sampled | sampled | sampled | e2e dot/matmul tests, parity matmul backward, matmul hint tests, kernel matrix | sampled | Close batched/broadcast matmul semantics, layout support, hints, and compile parity. |
| Indexing | `gather`, `index_select`, `scatter_add`, `embedding`, `topk`, `one_hot` | sampled | sampled or n/a | sampled | parity indexing tests, backward tests, nn embedding parity, kernel matrix | sampled | Make index dtype, bounds errors, duplicate-index gradient accumulation, and backend limits explicit. |
| Losses | `softmax`, `log_softmax_nll`, `cross_entropy` | sampled | sampled | sampled | e2e nn loss tests, parity softmax/loss tests, LM motif tests, kernel matrix | sampled | Close numerical stability contract, class axis semantics, fused/eager parity, and layout cases. |
| Normalization | `layer_norm`, `rms_norm`, `add_layer_norm` | sampled | sampled | sampled | e2e nn tests, parity layernorm tests, decoder fanout parity, kernel matrix | sampled | Close axis/normalized-shape contract, fused equivalence, and backend limitations. |
| Training updates | `sgd`, `adam`, `adamw`, `clip_grad_norm`, update kernels | sampled | n/a | sampled | e2e optim tests, parity optim tests, Metal optimizer tests, CUDA optimizer parity tests | sampled | Close dtype/device/layout behavior and update-kernel equivalence to mathematical optimizer specs. |
| Attention motif | attention score/value path | sampled | sampled | sampled | parity LM attention motif tests | sampled | Tie motif evidence back to primitive claim rows and compile/backend variants. |
| Decoder block motif | decoder block and layernorm fanout | sampled | sampled | sampled | parity decoder block tests, fanout tests, decoder training parity | sampled | Separate module parity evidence from primitive semantic closure. |
| Tiny training canary | decoder-LM tiny corpus update | sampled | sampled | sampled | app training tests and golden loss prefixes | sampled | Keep as canary only; do not count as primitive closure. |

## Closure Checklist

A row can move from `sampled` to `closed` only when the intended supported class
has evidence for:

- contiguous inputs
- non-contiguous view inputs
- offset views where accepted
- broadcast strides where applicable
- non-uniform upstream gradients for backward
- parameter gradient accumulation where applicable
- eager / graph / compile agreement where the op is intended to compile
- CPU / Metal / CUDA parity or explicit backend rejection
- expected errors for unsupported cases

## First Vertical Closure Target

The first end-to-end closure target is:

```text
broadcast add -> reshape -> permute -> matmul -> masked_fill -> softmax -> loss
```

For this chain, Affon should close:

- forward value semantics
- backward gradient routing
- non-contiguous upstream gradients
- broadcasted parameter gradients
- eager / graph / compile agreement
- CPU / Metal / CUDA parity where supported
- specified errors for unsupported cases
- diagnostics for materialization, fallback, or backend downgrade

Until that chain is closed, decoder-LM training should be treated as a valuable
canary rather than semantic proof.
