# NN Concepts

Neural-network authoring uses callable factories from `affon:nn`.
These factories declare parameterized operations in the Program that owns their bound tensors;
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
import { linear } from 'affon:nn'
import { relu } from 'affon:ops'

const hiddenLayer = linear({ out_features: 256 })
const outputLayer = linear({ out_features: 10 })
const classifier = program('classifier', p => {
  const input = p.argument('input', Tensor.f32([32, 768]))
  const hidden = relu(hiddenLayer({ x: input }, 'hidden'))
  return outputLayer({ x: hidden }, 'output')
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

Reusable components are ordinary functions. A factory separates fixed
hyperparameters from the named tensor bindings and optional instance name used
when it expands.

Stateful module objects are not part of the public API. Layer factories are
ordinary functions; inspectable model structure, parameters, state,
differentiation, and optimization are expressed as Programs.

## Related Docs

- [ML Glossary](../glossary.md)
- [Optim Concepts](../optim.md)
- [Metrics Concepts](../metrics.md)
- [Checkpoints](../checkpoints.md)
