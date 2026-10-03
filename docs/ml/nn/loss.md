# Loss Programs

A loss is a Program with a single-element output. It can be differentiated with
`gradient(...)` or turned into a state-changing training Program with
`optimize(...)`.

## Cross entropy

```ts
import { Tensor, program } from 'affon:compute'

const loss = program('classification_loss', p => {
  const logits = p.argument('logits', Tensor.f32([32, 10]))
  const labels = p.argument('labels', Tensor.i64([32]))
  return p.nn.cross_entropy(logits, labels)
})
```

`p.nn.cross_entropy` expects class logits on the final axis. Labels must use
`i64` and have the logits shape with that final axis removed. The result is a
single-element mean loss.

When the logits come from another Program, compose it while authoring the loss:

```ts
const loss = program('classifier_loss', p => {
  const image = p.argument('image', Tensor.f32([32, 784]))
  const labels = p.argument('labels', Tensor.i64([32]))
  return p.nn.cross_entropy(classifier(image), labels)
})
```

Cross entropy is builder-bound because it is currently part of the stable NN
Program vocabulary. General tensor transformations and activations remain in
`affon:ops`.

The former callable loss modules, including MSE and binary cross entropy,
remain in `affon:nn/legacy` until equivalent canonical Program helpers are
defined.
