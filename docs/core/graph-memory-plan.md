# Graph memory ownership

Native reusable graphs and ONNX command batching use distinct ownership models.

## Native reusable graphs

The opt-in native planner computes value lifetimes and assigns compatible storage
slots to values whose lifetimes do not overlap. A graph execution owner retains
slots across executions. Storage cannot be recycled while a queued kernel still
references it. Scratch-using invocations may conservatively complete the current
chunk before continuing. This planner is not the public ONNX graph allocator.

## ONNX scopes

ONNX graphs use bounded physical-resource leases around their synchronous Metal
forward call. Leases retain inputs, outputs, and scratch until command completion;
they do not replace graph traversal or eliminate tensor allocations. See
[execution scopes](metal-execution-scopes.md) for chunk limits and fallback behavior.

## Implementation boundaries

Compute owns lifetime planning, storage ownership, and backend command completion.
Affon supplies a trusted graph-call boundary. Model adapters do not select which
operator names are batchable. Kernel bindings supply resource requirements through
the [execution context](kernel-execution-context.md).

Tests must cover early retirement, aliasing, repeated execution, output survival,
errors, oversized resources, and numerical equivalence. No universal performance
or memory reduction follows solely from enabling batching.
