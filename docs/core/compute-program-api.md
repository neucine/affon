# Compute Program API

The JavaScript compute API is converging on one execution model:

```text
Program -> Session.compile/cache -> Executable -> run -> evaluated Tensor
                          |                         |
                          +---- ExecutionState <---+
```

`program(name, p => ...)` is the only `Program` constructor. Its
`ProgramBuilder` declares role-separated `argument`, `parameter`, `state`, and
`constant` tensors. Programs are callable only for symbolic composition while
another program is being authored. `p.use(child, { as, ...bindings })` provides
the explicit named form. All ordinary tensor computation lives in
`affon:ops`. The same operation functions build nodes for `FormalTensor`
operands and execute immediate, non-differentiable computation for evaluated
`Tensor` operands. Program builders do not expose operation methods.

Immediate operations infer the owning Session from their tensor operands. All
operands must belong to that Session; mixing Sessions or mixing formal and
evaluated tensors is an error. Differentiation and optimization continue to
accept Programs only—immediate operations never create an eager tape.

The native JavaScript bridge lowers Programs into the compute core compiler.
`Session.compile(...)` caches native executables, `Session.initialize(...)`
creates seeded parameters and model state, and `Executable.run(...)` validates
named arguments and session ownership before execution. Gradient Programs use
the core differentiator. Optimize Programs evaluate the loss and gradients,
then run ordinary compiled update Programs for SGD, momentum, Adam, or AdamW;
parameter and optimizer state changes become visible only after a successful
step.

`Executable` exposes argument, parameter, state, and native input ordering.
`argument_specs` provides the complete named TensorSpec contract for tooling.
Callers pass a `ProgramArguments` record to `run`; unknown, missing, disposed,
foreign-Session, wrong-dtype, and wrong-shape values are rejected before native
execution. Parameters and state come from the session-owned `ExecutionState`.
Only authored Programs can be composed; gradient and optimizer transforms are
execution boundaries. Composition requires exact positional or named bindings.
Evaluated tensors, executables, execution state, and sessions all expose
`dispose()`. A session retains its native resources until its remaining child
objects have also been disposed or finalized.

## Public boundary

`affon:compute` is the sole compute API. Operations are imported from
`affon:ops`, optimizer descriptors from `affon:optim`, and neural-network
parameters are declared through the active Program builder. Public options use
snake_case names such as `learning_rate` and `out_features`.

The new surface intentionally contains no `Module`, stateful callable layer,
`Program.*`, hyperparameter/specialization API, `Context`, `Variables`, `Bindings`,
`Snapshot`, `Store`, tape, mutable gradient, or silent eager-fallback concept.
Parameterized Program components are declared explicitly through `p.nn`.
