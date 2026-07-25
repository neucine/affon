# NN Basic

Basic feed-forward concepts for `affon:nn`.

## Linear layer

`nn.Linear(in_features, out_features, opts?)` maps:

```txt
[..., in_features] -> [..., out_features]
```

Math:

$$
y = xW + b
$$

The layer always applies across the last dimension.
Any leading dimensions are preserved.

If:

```txt
x.shape = [2, 3]
W.shape = [3, 4]
b.shape = [1, 4]
```

then:

```txt
y.shape = [2, 4]
```

And for a batched sequence:

```txt
x.shape = [5, 8, 3]
y.shape = [5, 8, 4]
```

Visual idea:

```txt
2 examples in the batch
each example has 3 input features
each example becomes 4 output features
```

## How the layer is connected

A linear layer is fully connected.

That means:
- every output feature sees every input feature
- the same weight matrix is used for every slice of the leading dimensions

If:

```txt
in_features = 3
out_features = 4
```

then the connectivity looks like:

![Fully connected neural network](https://commons.wikimedia.org/wiki/Special:FilePath/Fully_connected_neural_network.svg)

Source:
- Wikimedia Commons, "Fully connected neural network": https://commons.wikimedia.org/wiki/File:Fully_connected_neural_network.svg

So one output unit is computed from all input features, not just one.

## Parameters

For `Linear`:

- `weight` has shape `[in_features, out_features]`
- `bias` has shape `[1, out_features]`

Both are trainable parameters.

You can force the parameter dtype explicitly when needed:

```ts
const layer = nn.Linear(3, 4, { dtype: 'f32' })
```

The same `dtype` option is supported by:
- `nn.Embedding(...)`
- `nn.LayerNorm(...)`

Learnable `nn` layers default to `f32` parameters.

## Common pattern

Feed-forward models often look like:

```ts
import { relu } from 'affon:compute'
import nn from 'affon:nn'

const model = nn.Sequential(
  nn.Linear(2, 4),
  relu,
  nn.Linear(4, 1),
)
```

This means:
- apply one linear transform
- apply an activation
- apply another linear transform

Visual idea:

```txt
input -> Linear -> Activation -> Linear -> output
```
