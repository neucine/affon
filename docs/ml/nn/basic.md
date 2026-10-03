# Linear, Embedding, and Normalization

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
