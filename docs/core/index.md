# Core Numerics Docs

Core numerics cover the public `affon:compute` surface: tensors, parameters,
autograd, optimizers, schedules, module state, graph compilation, and backend
execution.

Read these pages in this order if you are new to the compute layer:

- [Compute Concepts](./compute.md) - compute tensors, parameters, programs, modules, compilation, and observability
- [Compute Kernel Matrix](./kernel-matrix.md) - current backend/device support by operation family
- [Optimization Checkpoint](compute-optimization-checkpoint.md) - completed work, remaining gaps, evidence, and how to resume
- [Compute Semantic Coverage](./compute-semantic-coverage.md) - correctness claims, evidence, and closure gaps
- [Error Handling](./errors.md)

## Public Boundary

Use `affon:compute` as the public numeric API. There is no separate public
`affon:tensor` or `affon:ndarray` module. Device-specific behavior, graph
lowering, and native execution details are documented only where they affect
observable behavior or support status.

## Common Tasks

- Create data with `tensor(...)`, `zeros(...)`, `ones(...)`, `rand(...)`, and `randn(...)`.
- Create trainable state with `parameter(...)`.
- Differentiate scalar losses with `grad(loss, params)`.
- Update parameters with optimizers such as `sgd(...)`, `adam(...)`, or `adamw(...)`.
- Inspect support status before relying on backend-specific execution.

Design proposals:

- [Bounded Metal execution scopes](metal-execution-scopes.md): graph-level command
  accumulation with synchronous return, ownership and validation requirements.
