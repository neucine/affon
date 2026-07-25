# NN Concepts

`affon:nn` is the model-building layer on top of `affon:compute`. Use it for
layers, losses, recurrent modules, normalization, embeddings, dropout, and
module composition.

## Topic Guides

- [General NN Concepts](./general.md) - modules, parameters, state, mode, and composition.
- [Basic Feed-Forward Models](./basic.md) - linear layers and simple classifiers/regressors.
- [Recurrent Models](./recurrent.md) - RNN and LSTM concepts, shapes, and state.
- [Activations](./activation.md) - activation functions and where they live.
- [Loss Functions](./loss.md) - loss modules and target/prediction expectations.

## Import Shape

```ts
import nn from 'affon:nn'
import { relu } from 'affon:compute'
```

Most layers and losses are constructed from `nn`. Activations are free
functions from `affon:compute`, so model code can use the same function in eager
expressions, module composition, and compiled graphs.

## Related Docs

- [ML Glossary](../glossary.md)
- [Optim Concepts](../optim.md)
- [Metrics Concepts](../metrics.md)
- [Checkpoints](../checkpoints.md)
