# Linear, Convolution, Embedding, and Normalization

`affon:nn` exports factories for parameterized neural-network operations.

## Linear

```ts
const projection = linear({
  out_features: 10,
  bias: true,
})
const output = projection({ x: input }, 'output')
```

`linear` transforms the final input axis and preserves all leading axes:

```text
[..., in_features] -> [..., out_features]
```

The input feature width is inferred from the formal tensor specification. The
optional instance name determines dotted parameter names such as
`output.weight` and `output.bias`; it defaults to `linear`.

## One-dimensional convolution

```ts
const encoder = conv1d({ out_channels: 32, kernel_size: 3, stride: 2, padding: 1 })
const encoded = encoder({ x: p.argument('x', Tensor.f32([8, 16, 100])) }, 'encoder')
// encoded shape: [8, 32, 50]

const decoder = conv_transpose1d({ out_channels: 16, kernel_size: 3,
  stride: 2, padding: 1, output_padding: 1 })
const restored = decoder({ x: encoded }, 'decoder')
// restored shape: [8, 16, 100]
```

Both factories use `[batch, channels, length]` inputs. `conv1d` declares
`[out_channels, in_channels/groups, kernel_size]` weights, while
`conv_transpose1d` declares `[in_channels, out_channels/groups, kernel_size]`
weights. Both support `stride`, `padding`, `dilation`, `groups`, and `bias`;
transposed convolution also supports `output_padding`, which must be smaller
than `stride`. The factories expand into standard tensor operations and support
gradients through their weights and inputs.

## Embedding

```ts
const tokens = p.argument('tokens', Tensor.i64([32, 128]))
const tokenEmbedding = embedding({
  num_embeddings: 32_000,
  embedding_dim: 768,
})
const hidden = tokenEmbedding({ indices: tokens }, 'token_embedding')
```

Embedding inputs are integer indices. The output appends the embedding width to
the index shape.

## Layer normalization

```ts
const normalize = layer_norm({
  normalized_shape: 768,
  epsilon: 1e-5,
  affine: true,
})
const normalized = normalize({ x: hidden }, 'final_norm')
```

When `affine` is enabled, the factory declares learned scale and bias
parameters. For unparameterized immediate normalization, use
`layer_norm(x, axis, epsilon)` from `affon:ops`.
