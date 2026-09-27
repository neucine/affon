# Kernel execution context

Compute's Metal `kernel.Context` is the resource/completion boundary for migrated
kernels. Dispatch selects the implementation by operation, dtype, shape, and
layout. The context owns admission and completion, not implementation selection.

## Resources and completion

Each invocation binds typed reads, writes, and scratch before encoding. Resource
leases derive from those bindings; there is no operation-name batching whitelist.
Eager mode completes the invocation synchronously. An active graph scope retains
physical buffers until its chunk completes. Legacy operations and host-access
boundaries complete pending work. MPS objects follow the same retention rule.

Scratch is part of the invocation's resource budget. Temporary metadata copied
into an encoder may be stack-owned; referenced GPU buffers may not. Invalid
encoding must poison/drain the active scope rather than publish partial outputs.

## Implemented paths

Migrated families include unary, binary, broadcast, softmax, normalization,
affine, ordinary f32 matmul and fused epilogues, cat/stack, and supported whole
and axis reductions. Dtype/layout fallbacks can still form synchronous boundaries.
A family name is not a guarantee that every variant uses the context.

Supported nonnegative-stride f32 reduction views pack and reduce in one invocation.
The source, output, and packed scratch are declared together. Integer and
negative-stride preparation retain their existing fallback behavior. Dense-row,
rank-two, and ND f32 softmax use the common finish without changing shader math.

See [scope semantics](metal-execution-scopes.md), [graph ownership](graph-memory-plan.md),
and [matmul dispatch](metal-matmul-dispatch.md). Reproduce performance with the
[operation harness](../../test/benchmarks/README.md); historical measurements are
not part of this API contract.
