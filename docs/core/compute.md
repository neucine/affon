# Compute Programs

`affon:compute` is Affon's declarative compute API. A computation is authored
once as a `Program`, compiled for a `Session`, and run with evaluated tensors:

```text
Program -> Session.compile/cache -> Executable -> run -> Tensor
                  |                              |
                  +---- ExecutionState ----------+
```

The older eager and captured-graph API is temporarily available from
`affon:compute/legacy`. New code should not mix the two execution models.

## Author a Program

Use `program(name, builder => output)` as the only Program constructor. The
builder declares inputs by role, and `affon:ops` supplies the computation:

```ts
import { Tensor, program } from "affon:compute"

const classifier = program("classifier", p => {
  const image = p.argument(
    "image",
    Tensor.f32([32, 784], { axes: ["batch", "feature"] }),
  )
  return p.nn.linear(image, { name: "head", out_features: 10 })
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
  const first = classifier(image)
  const second = p.use(classifier, { as: "second", image })
  return add(first, second)
})
```

`p.use` is the explicit named form. Its `as` alias namespaces the child
parameters, state, and constants. Initializers and semantic metadata survive
composition.

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
import { matmul, softmax } from "affon:ops"

const logits = matmul(image, weight)
const probabilities = softmax(logits, 1)
```

Both results belong to the inputs' Session. Use an explicit Program when the
computation must be differentiated, optimized, inspected, or reused as a named
model.

## Losses and Differentiation

Program transforms remain declarative:

```ts
import { gradient, program, Tensor } from "affon:compute"

const loss = program("classifier_loss", p => {
  const image = p.argument("image", Tensor.f32([32, 784]))
  const labels = p.argument("labels", Tensor.i64([32]))
  return p.nn.cross_entropy(classifier(image), labels)
})
const gradients = gradient(loss, ["classifier_head_weight", "classifier_head_bias"])

const lossExecutable = session.compile(loss)
const gradientExecutable = session.compile(gradients)
```

`p.nn.cross_entropy(logits, labels)` produces a single-element mean loss. Its
labels must be i64 and match the logits shape without its final class axis. `gradient`
requires a single-element output and differentiates through the compute core,
not through an eager fallback.

## Optimization

`optimize` declares a state transition over a loss Program:

```ts
import { optimize } from "affon:compute"
import { adamw } from "affon:optim"

const train = optimize(loss, adamw({
  learning_rate: 3e-4,
  weight_decay: 0.01,
}))

const trainState = session.initialize(train, { seed: 7 })
const trainStep = session.compile(train)
const currentLoss = trainStep.run({ image, labels }, trainState)
```

Available immutable descriptors are `sgd`, `adam`, and `adamw`. A successful
step atomically replaces parameters and optimizer moments, advances `$step`,
and increments the RNG counter. A failed step leaves the prior state installed.

## Resource Lifetime

`Tensor`, `Executable`, `ExecutionState`, and `Session` support deterministic
`dispose()` and `Symbol.dispose`. Dispose outputs and input tensors when they
are no longer needed, then dispose execution state and the Session. Native
session storage remains alive while a child tensor or executable still owns
it, which prevents dangling native handles during cleanup.

## Legacy Migration

Existing packages that still use global constructors, global math functions,
`module`, legacy `compile(function)`, mutable gradients, or stateful optimizer
steps must import them from `affon:compute/legacy`:

```ts
import { tensor, matmul } from "affon:compute/legacy"
```

Those exports are a migration boundary, not part of the Program API.
