# NN Concepts

Neural-network authoring lives on the active Program builder under `p.nn`.
These helpers declare parameterized operations directly in the Program;
losses, execution, and differentiation remain Program-level concerns.

## Topic Guides

- [Program Components](./general.md) - reusable components, parameter roles, state, and composition.
- [Basic Feed-Forward Models](./basic.md) - linear layers and simple classifiers/regressors.
- [Recurrent Models](./recurrent.md) - current Program support and limitations.
- [Activations](./activation.md) - activation functions and where they live.
- [Loss Programs](./loss.md) - built-in templates, custom losses, and
  target/prediction expectations.

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

The canonical helpers currently cover linear projections, embeddings, and
layer normalization. Elementwise, tensor, and custom loss operations remain in
`affon:ops`, where the same functions accept
formal tensors during Program construction and evaluated tensors for immediate
value computation.

Standard training losses are available as templates from the `losses`
namespace in `affon:compute` and are combined with a reusable model through
`optimize(model, loss, optimizer)`.

Reusable components are ordinary functions that receive a `ProgramBuilder`
and call its `p.nn` methods. This keeps one spelling for each NN operation.

Callable module objects are not part of the public API. Model structure,
parameters, state, differentiation, and optimization are expressed as Programs.

## Related Docs

- [ML Glossary](../glossary.md)
- [Optim Concepts](../optim.md)
- [Metrics Concepts](../metrics.md)
- [Checkpoints](../checkpoints.md)
