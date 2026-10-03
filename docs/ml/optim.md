# Optimizing Programs

Optimization combines a reusable model Program, a scalar loss Program or
specialized loss callable, and an update rule from `affon:optim`.

```ts
import { optimize } from 'affon:compute'
import { cross_entropy } from 'affon:nn'
import { adamw } from 'affon:optim'

const train = optimize(model, cross_entropy(), adamw({
  learning_rate: 3e-4,
  weight_decay: 0.01,
}))
```

The public optimizer factories are:

- `sgd({ learning_rate, momentum })`
- `adam({ learning_rate, beta1, beta2, epsilon })`
- `adamw({ learning_rate, beta1, beta2, epsilon, weight_decay })`
- `scheduled(optimizer, schedule)`
- `accumulate(optimizer, { steps })`

They return immutable descriptors. They do not hold parameters, expose a
mutable learning rate, or provide a callable `step(params)` function.

## Explicit parameter updates

Use the lower-level transforms when gradient selection must remain visible:

```ts
import { gradient, update_parameters } from 'affon:compute'

const derivatives = gradient(loss, [
  'classifier.head.weight',
  'classifier.head.bias',
])
const train = update_parameters(loss, derivatives, adamw({
  learning_rate: 3e-4,
}))
```

The gradient Program must be derived from the same source Program. Only its
selected parameters are updated. Running the transformed Program returns the
source output and atomically updates those parameters and their optimizer
state. `optimize(model, loss, optimizer)` is the high-level form that performs
the model/loss composition and gradient selection automatically.

## Gradient accumulation

Wrap an optimizer when a larger effective batch should span several runs:

```ts
import { accumulate, adamw } from 'affon:optim'

const optimizer = accumulate(adamw({ learning_rate: 3e-4 }), { steps: 4 })
const train = optimize(model, cross_entropy(), optimizer)
```

Each run computes and adds one microbatch gradient. Parameters and optimizer
moments remain unchanged until the fourth run, when the averaged gradient is
applied as one optimizer step. Accumulation buffers live in the supplied
`ExecutionState`, so separate states accumulate independently.

## Learning-rate schedules

Schedules are immutable, step-based descriptors. Wrap the base optimizer, then
wrap that result with accumulation when both are needed:

```ts
import { accumulate, adamw, scheduled, schedules } from 'affon:optim'

const optimizer = accumulate(
  scheduled(
    adamw({ learning_rate: 3e-4, weight_decay: 0.01 }),
    schedules.warmupCosine({
      start: 1e-5,
      peak: 3e-4,
      end: 3e-5,
      warmup_steps: 500,
      total_steps: 10_000,
    }),
  ),
  { steps: 4 },
)
```

Available factories are `constant`, `linear`, `cosine`, `step`,
`warmupCosine`, and `sequence`. Schedule progress uses completed optimizer
updates (`$step`), so accumulation microsteps do not advance it. Epoch-based
schedules remain training-workflow policy because epochs are not intrinsic to
a compiled training step.

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

`gradient(loss, names)` returns gradients without changing state;
`update_parameters(loss, gradients, optimizer)` applies explicitly selected
gradients. Immediate evaluated-tensor operations are intentionally not differentiable.

## State and lifetime

Optimizer moments live in `state.optimizer_state`; model parameters live in
`state.parameters`. Dispose each output tensor after use, and dispose the
executable, execution state, and Session when training is finished.

Gradients and optimizer updates are Program transforms; evaluated tensors do
not expose mutable gradient slots or stateful optimizer steps.
