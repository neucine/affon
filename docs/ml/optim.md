# Optimizing Programs

Optimization is a Program transform. `affon:optim` describes an update rule;
`optimize(...)` applies it to a scalar loss Program.

```ts
import { optimize } from 'affon:compute'
import { adamw } from 'affon:optim'

const train = optimize(loss, adamw({
  learning_rate: 3e-4,
  weight_decay: 0.01,
}))
```

The public optimizer factories are:

- `sgd({ learning_rate, momentum })`
- `adam({ learning_rate, beta1, beta2, epsilon })`
- `adamw({ learning_rate, beta1, beta2, epsilon, weight_decay })`

They return immutable descriptors. They do not hold parameters, expose a
mutable learning rate, or provide a callable `step(params)` function.

## Run a training step

```ts
const state = session.initialize(train, { seed: 7 })
const step = session.compile(train)
const currentLoss = step.run({ image, labels }, state)

console.log(currentLoss.item())
currentLoss.dispose()
```

The loss and its differentiation remain inside the transformed Program. A
successful run updates the `ExecutionState` atomically: parameters and
optimizer moments are replaced, `$step` advances, and the RNG counter is
incremented. A failed run leaves the previous state installed.

`gradient(loss, names)` is available when gradients themselves are the desired
Program output. Immediate evaluated-tensor operations are intentionally not
differentiable.

## State and lifetime

Optimizer moments live in `state.optimizer_state`; model parameters live in
`state.parameters`. Dispose each output tensor after use, and dispose the
executable, execution state, and Session when training is finished.

The old mutable `.grad`, `clear_grad`, and callable optimizer-step protocol is
available only through `affon:compute/legacy` during migration.
