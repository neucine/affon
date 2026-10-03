# ML Glossary

- **Program** — An immutable, inspectable computation authored with
  `program(name, p => output)`.
- **ProgramBuilder (`p`)** — The scoped authoring object that declares argument,
  parameter, state, and constant roles. Its `p.nn` namespace declares supported
  neural-network parameters and losses.
- **formal tensor** — A symbolic tensor value used only while authoring a
  Program. Operations on it add nodes to that Program.
- **evaluated tensor** — A Session-owned value containing computed data.
  Operations on it execute immediately and are computation-only.
- **TensorSpec** — Immutable shape, dtype, and optional semantic-axis metadata.
  Construct one with `Tensor.f32(...)`, `Tensor.f64(...)`, `Tensor.i64(...)`, or
  `Tensor.spec(...)`.
- **Session** — The owner of evaluated tensors, executable caches, and execution
  resources for one device.
- **Executable** — A Session-bound compiled Program. Run it with a named argument
  record and, when required, an `ExecutionState`.
- **ExecutionState** — Session-owned parameters, model state, optimizer state,
  and RNG state materialized for a Program.
- **Program composition** — Using an authored child Program inside a parent
  Program, either by calling it during authoring or with `p.use(...)`.
- **operation** — A function from `affon:ops`. The same spelling handles formal
  and evaluated tensors while preserving their distinct semantics.
- **parameter** — Trainable Program state declared with `p.parameter(...)` or by
  a parameterized `p.nn` helper.
- **model state** — Persistent non-parameter data declared with `p.state(...)`.
- **gradient Program** — A Program produced by `gradient(loss, names)`.
- **optimization Program** — A state-transition Program produced by
  `optimize(loss, optimizer)`.
- **optimizer descriptor** — An immutable value from `affon:optim`, such as
  `adamw({ learning_rate: 3e-4 })`.
- **logits** — Raw class scores before softmax. `p.nn.cross_entropy` consumes
  logits directly.
- **dtype** — The tensor numeric type: currently `f32`, `f64`, or `i64` in the
  canonical Program surface.
- **shape** — The ordered sizes of a tensor's axes.
- **axis** — One position in a tensor shape; it may optionally have a semantic
  name in a TensorSpec.
- **legacy API** — The older eager/autograd and callable-module model exposed
  temporarily through `affon:compute/legacy` and `affon:nn/legacy`.
