# Bounded Metal execution scopes

Metal ONNX graph calls and native Llama inference forward passes accumulate
commands within a trusted synchronous scope.
Outputs are complete before return. Ordinary eager calls remain synchronous;
ONNX profiling callbacks bypass batching. No public asynchronous tensor API is implied.

The scope retains resources declared through the [kernel context](kernel-execution-context.md).
Chunks are bounded by 32 invocations and 64 MiB of leased physical capacity.
Llama enters the scope inside `no_grad`, and publishes its next KV cache only
after successful completion. CPU and CUDA retain their existing eager path.

An oversized invocation runs alone and can exceed that chunk budget. This is not
a cap on model weights, pooled buffers, or total process memory.

Unmigrated kernels and required host access complete pending commands. Resource
leases survive submission until completion, including MPS adapter objects.
Failure paths must complete or invalidate pending work before recycling storage.

[Native graph slot reuse](graph-memory-plan.md) is a separate ownership mechanism.
Cross-repository rationale is maintained in affon-arch; this page describes the
implemented runtime boundary.
