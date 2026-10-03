# Loss Programs

A loss is a Program with a single-element output. It can be differentiated with
`gradient(...)` or turned into a state-changing training Program with
`optimize(...)`.

## Cross entropy

```ts
import { Tensor, losses, optimize, program } from 'affon:compute'
import { cross_entropy } from 'affon:ops'

const loss = program('classification_loss', p => {
  const logits = p.argument('logits', Tensor.f32([32, 10]))
  const labels = p.argument('labels', Tensor.i64([32]))
  return cross_entropy(logits, labels)
})
```

`cross_entropy` expects class logits on the final axis. Labels must use
`i64` and have the logits shape with that final axis removed. The result is a
single-element mean loss.

Keep the model and loss as separate Programs so the model remains reusable for
evaluation and inference:

```ts
const loss = program('classifier_loss', p => {
  const logits = p.argument('logits', Tensor.f32([32, 10]))
  const labels = p.argument('labels', Tensor.i64([32]))
  return cross_entropy(logits, labels)
})

const train = optimize(classifier, loss, optimizer)
```

For the common case, `losses.cross_entropy()` is a built-in template that
infers both specs from the model when passed to `optimize`:

```ts
const train = optimize(classifier, losses.cross_entropy(), optimizer)
```

The built-in namespace currently provides:

- `losses.cross_entropy()` for class-index labels inferred as `i64`.
- `losses.mean_squared_error()` for a same-shaped floating-point target.
- `losses.binary_cross_entropy()` for same-shaped probability predictions and targets.
- `losses.binary_cross_entropy_with_logits()` for same-shaped logits and binary targets.

Each accepts `{ target }` to replace the default training-input name (`labels`
for cross entropy, otherwise `target`). All materialize ordinary scalar loss
Programs; there is no separate optimizer execution path for built-in losses.

The former callable loss modules remain available from `affon:nn/legacy` only
for migration; new Program code should use `losses` or author an explicit loss
Program from `affon:ops`.
