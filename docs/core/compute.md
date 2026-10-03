# Compute Programs

`affon:compute` is Affon's declarative compute API. A computation is authored
once as a `Program`, compiled for a `Session`, and run with evaluated tensors:

```text
Program -> Session.compile/cache -> Executable -> run -> Tensor
                  |                              |
                  +---- ExecutionState ----------+
```

The Program API is the only public compute model. Tensor operations live in
`affon:ops`; execution, differentiation, and optimization remain Program-level
concerns.

## Author a Program

Use `program(name, builder => output)` as the only Program constructor. The
builder declares inputs by role, and `affon:ops` supplies the computation:

```ts
import { Tensor, program } from "affon:compute"
import { linear } from "affon:nn"

const head = linear({ out_features: 10 })
const classifier = program("classifier", p => {
  const image = p.argument(
    "image",
    Tensor.f32([32, 784], { axes: ["batch", "feature"] }),
  )
  return head({ x: image }, "head")
})
```

The four declaration roles are:

- `p.argument(name, spec)` for values supplied on every run.
- `p.parameter(name, spec, { initializer })` for trainable state.
- `p.state(name, spec, { initializer })` for persistent model state.
- `p.constant(name, value, spec)` for immutable Program data.

Specs are created with `Tensor.f32`, `Tensor.f64`, `Tensor.i64`, or
`Tensor.spec`. These are the same three dtypes supported by evaluated tensors,
so every declared Program argument can be supplied through `Session.tensor()`.

Operations can be imported selectively or as a namespace:

```ts
import { add, matmul } from "affon:ops"
// or: import * as ops from "affon:ops"
```

With formal operands these functions add symbolic nodes to the owning Program.
With evaluated operands they execute immediately through the operands' shared
Session and return an evaluated `Tensor`. Immediate calls are computation-only:
they do not record gradients. Mixed representations and tensors from different
Sessions are rejected.

## Compose Programs

A Program is callable only while another Program is being authored:

```ts
import { add } from "affon:ops"

const ensemble = program("ensemble", p => {
  const image = p.argument("image", Tensor.f32([32, 784]))
  const first = classifier({ image })
  const second = classifier({ image }, "second")
  return add(first, second)
})
```

A Program call accepts named bindings and an optional instance name. Without
one, the child Program name is used. The instance name namespaces child parameters,
state, and constants with dot-separated names. Initializers and semantic metadata
survive composition. Inspection automatically records each copied node's nested
composition `path`; authors never set paths manually. The executable graph
remains flat and retains ordinary operand IDs.

Call `program.inspect()` to read immutable arguments, parameters, model state,
constants, graph nodes, outputs, and declared transitions. Calling a Program
outside authoring is an error; execution always goes through a compiled
`Executable`.

## Compile and Run

Sessions own evaluated tensors, executable caches, and execution state:

```ts
import { Session } from "affon:compute"

const session = new Session({ device: "cpu" })
const state = session.initialize(classifier, { seed: 7 })
const executable = session.compile(classifier)
const image = session.tensor(batch)
const logits = executable.run({ image }, state)
```

`Session.compile(program)` returns the cached executable when the same Program
is compiled again. `Executable.run` accepts named arguments only. Parameters
and model state are supplied from an `ExecutionState`, and values from another
Session are rejected.

Evaluated tensors expose `shape`, `ndim`, `dtype`, `device`, optional `axes`,
`item()`, `to_array()`, `toString()`, `repr()`, and `dispose()`.

For one-off computation, no explicit Program is required:

```ts
import { Tensor } from "affon:compute"
import { matmul, softmax } from "affon:ops"

const image = Tensor.from(batch)
const weight = Tensor.from(weights)
const logits = matmul(image, weight)
const probabilities = softmax(logits, 1)
```

Both results belong to the inputs' Session. Use an explicit Program when the
computation must be differentiated, optimized, inspected, or reused as a named
model.

`Tensor.from`, `Tensor.zeros`, `Tensor.ones`, `Tensor.full`, `Tensor.arange`,
`Tensor.linspace`, `Tensor.rand`, and `Tensor.randn` use one hidden,
lazily created default Session. Its device is fixed by the runtime startup
configuration: `AFFON_DEVICE` when set, otherwise Affon's normal device
detection. The canonical API provides no global setter, replacement Session,
or per-constructor device override. Create an explicit `Session` when a workload
needs isolation or a specific device.

## Losses and Differentiation

Program transforms remain declarative:

```ts
import { gradient, program, Tensor, update_parameters } from "affon:compute"
import { cross_entropy } from "affon:ops"
import { adamw } from "affon:optim"

const loss = program("classifier_loss", p => {
  const image = p.argument("image", Tensor.f32([32, 784]))
  const labels = p.argument("labels", Tensor.i64([32]))
  return cross_entropy(classifier({ image }), labels)
})
const gradients = gradient(loss, ["classifier.head.weight", "classifier.head.bias"])
const update = update_parameters(loss, gradients, adamw({ learning_rate: 3e-4 }))

const lossExecutable = session.compile(loss)
const gradientExecutable = session.compile(gradients)
const updateExecutable = session.compile(update)
```

`cross_entropy(logits, labels)` produces a single-element mean loss. Its
labels must be i64 and match the logits shape without its final class axis. `gradient`
requires a single-element output and differentiates through the compute core,
not through an eager fallback.

`update_parameters(source, gradients, optimizer)` is the low-level state
transition. The gradient Program must come from `gradient(source, names)`, and
only the selected parameter names are updated. The resulting Program preserves
the source arguments and output, so running `update` above returns the current
loss while applying the explicit gradients to its `ExecutionState`.

`affon:ops` also provides `mean_squared_error`, `mean_absolute_error`,
`binary_cross_entropy`, and `binary_cross_entropy_with_logits` for custom loss
Programs. Specialized loss callables from `affon:nn` remain the concise choice
when the standard target shape can be inferred from a model.

## Optimization

`optimize(model, loss, optimizer)` combines a reusable model, a scalar loss
Program or specialized loss callable, and an immutable optimizer descriptor into
a state-transition Program. Internally it composes the model and loss, derives
all parameter gradients, and delegates to `update_parameters`. Keep the model and loss separate when the same
model must also be compiled for evaluation or inference:

```ts
import { optimize } from "affon:compute"
import { cross_entropy } from "affon:nn"
import { adamw } from "affon:optim"

const classificationLoss = cross_entropy()

const train = optimize(classifier, classificationLoss, adamw({
  learning_rate: 3e-4,
  weight_decay: 0.01,
}))

const trainState = session.initialize(train, { seed: 7 })
const trainStep = session.compile(train)
const infer = session.compile(classifier)
const currentLoss = trainStep.run({ image, labels }, trainState)
const logits = infer.run({ image }, trainState)
```

Loss callables infer their input and target specs from the model.
`cross_entropy()` exposes an i64 `labels` training input by default. A
custom loss can instead be authored as a normal scalar Program using operations
such as `cross_entropy` from `affon:ops`. The combined training Program preserves
the model's parameter provenance, so its `ExecutionState` can be passed directly
to the separately compiled model.

`affon:nn` also provides `mean_squared_error()`,
`binary_cross_entropy()`, and
`binary_cross_entropy_with_logits()`. These infer a same-shaped
floating-point `target` input from the model output.

Available immutable descriptors are `sgd`, `adam`, and `adamw`. A successful
step atomically replaces parameters and optimizer moments, advances `$step`,
and increments the RNG counter. A failed step leaves the prior state installed.

## Resource Lifetime

Evaluated tensors, executables, execution state, and Sessions are reclaimed
automatically when they become unreachable. `dispose()` and `Symbol.dispose`
remain available for deterministic release in memory-sensitive loops; they are
not required in ordinary code. Native Session storage remains alive while a
child tensor or executable still owns it, preventing dangling native handles.
After a Session is disposed, existing tensors remain safely readable, while
executables and state cannot start new work.
