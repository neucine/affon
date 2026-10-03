# Loss Programs

A loss callable accepts named `input` and `target` bindings and returns a
single-element formal tensor. A loss may also be authored as an explicit
Program when it needs independent identity or inspection.

## Cross entropy

```ts
import { Tensor, optimize, program } from 'affon:compute'
import { cross_entropy } from 'affon:nn'

const classificationLoss = cross_entropy()
const loss = program('classification_loss', p => {
  const logits = p.argument('logits', Tensor.f32([32, 10]))
  const labels = p.argument('labels', Tensor.i64([32]))
  return classificationLoss({ input: logits, target: labels }, 'objective')
})
```

`cross_entropy()` expects class logits on the final axis. Labels must use
`i64` and have the logits shape with that final axis removed. The result is a
single-element mean loss.

Keep the model and loss as separate Programs so the model remains reusable for
evaluation and inference:

```ts
const loss = program('classifier_loss', p => {
  const logits = p.argument('logits', Tensor.f32([32, 10]))
  const labels = p.argument('labels', Tensor.i64([32]))
  return classificationLoss({ input: logits, target: labels }, 'objective')
})

const train = optimize(classifier, loss, optimizer)
```

For the common case, pass the loss callable directly to `optimize`, which
infers both specs from the model:

```ts
const train = optimize(classifier, cross_entropy(), optimizer)
```

`affon:nn` currently provides these loss factories:

- `cross_entropy()` for class-index labels inferred as `i64`.
- `mean_squared_error()` for a same-shaped floating-point target.
- `binary_cross_entropy()` for same-shaped probability predictions and targets.
- `binary_cross_entropy_with_logits()` for same-shaped logits and binary targets.

Each accepts `{ target }` to replace the default training-input name (`labels`
for cross entropy, otherwise `target`). `optimize` materializes each callable
as an ordinary scalar loss Program; there is no separate execution path.

Use the specialized callables from `affon:nn`, or author an explicit loss
Program from operations in `affon:ops`.
