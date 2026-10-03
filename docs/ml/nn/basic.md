# Linear, Embedding, and Normalization

The active Program builder exposes the parameterized neural-network operations.

## Linear

```ts
const output = p.nn.linear(input, {
  name: 'output',
  out_features: 10,
  bias: true,
})
```

`linear` transforms the final input axis and preserves all leading axes:

```text
[..., in_features] -> [..., out_features]
```

The input feature width is inferred from the formal tensor specification. The
`name` is required and determines the parameter names in the Program.

## Embedding

```ts
const tokens = p.argument('tokens', Tensor.i64([32, 128]))
const hidden = p.nn.embedding(tokens, {
  name: 'token_embedding',
  num_embeddings: 32_000,
  embedding_dim: 768,
})
```

Embedding inputs are integer indices. The output appends the embedding width to
the index shape.

## Layer normalization

```ts
const normalized = p.nn.layer_norm(hidden, {
  name: 'final_norm',
  normalized_shape: 768,
  epsilon: 1e-5,
  affine: true,
})
```

When `affine` is enabled, the builder declares learned scale and bias
parameters. For unparameterized immediate normalization, use
`layer_norm(x, axis, epsilon)` from `affon:ops`.
