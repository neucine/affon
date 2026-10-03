# NN Concepts

Neural-network authoring lives on the active Program builder under `p.nn`.
These operations declare parameters and losses directly in the Program;
execution and differentiation remain Program-level concerns.

## Topic Guides

- [Program Components](./general.md) - reusable components, parameter roles, state, and composition.
- [Basic Feed-Forward Models](./basic.md) - linear layers and simple classifiers/regressors.
- [Recurrent Models](./recurrent.md) - current Program support and the legacy boundary.
- [Activations](./activation.md) - activation functions and where they live.
- [Loss Functions](./loss.md) - loss modules and target/prediction expectations.

## Program authoring

```ts
import { program, Tensor } from 'affon:compute'
import { relu } from 'affon:ops'

const classifier = program('classifier', p => {
  const input = p.argument('input', Tensor.f32([32, 768]))
  const hidden = relu(p.nn.linear(input, {
    name: 'hidden',
    out_features: 256,
  }))
  return p.nn.linear(hidden, {
    name: 'output',
    out_features: 10,
  })
})
```

The canonical helpers currently cover linear projections, embeddings, layer
normalization, and cross entropy. Elementwise and
tensor operations remain in `affon:ops`, where the same functions accept
formal tensors during Program construction and evaluated tensors for immediate
value computation.

Reusable components are ordinary functions that receive a `ProgramBuilder`
and call its `p.nn` methods. This keeps one spelling for each NN operation.

The former callable-module/eager-autograd API is legacy and is not the basis of
new model code. It is available only from `affon:nn/legacy` and
`affon:compute/legacy` during migration.

## Related Docs

- [ML Glossary](../glossary.md)
- [Optim Concepts](../optim.md)
- [Metrics Concepts](../metrics.md)
- [Checkpoints](../checkpoints.md)
