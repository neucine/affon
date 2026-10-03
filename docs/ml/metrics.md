# Metrics

Metrics evaluate model outputs; they do not update model state. The canonical
API exposes reporting helpers through `metrics`:

```ts
import { metrics } from 'affon:compute'

const logits = session.compile(model).run({ image }, state)
const accuracy = metrics.accuracy(logits, labels)
```

Available metrics are `accuracy`, `precision`, `recall`, `f1`,
`mean_squared_error`, `mean_absolute_error`, and `r2_score`. Classification
accuracy accepts either multiclass logits plus i64 labels or same-shaped binary
predictions and targets. Binary metrics accept an optional `threshold`.

This distinction matters:

- a **loss Program** is the scalar objective passed to `gradient(...)` or
  `optimize(...)`;
- a **metric** returns a JavaScript number for reporting and is not part of the
  differentiated graph;
- an immediate `affon:ops` call computes a value in the operands' Session and
  never creates an eager gradient tape.

Keep metrics outside an optimized loss Program. Use tensor-returning loss
operations from `affon:ops` when a value must participate in differentiation.
