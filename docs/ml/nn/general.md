# NN General

Beginner-friendly mental model for `affon:nn`.

## Core idea

A neural-network module is:
- a function: `x -> y`
- plus trainable compute tensors called parameters

In Affon:

```ts
const y = model(x)
```

Training usually looks like:

```txt
epoch
  batch
    forward
    loss
    backward
    optimizer.step()
```

One module call processes the whole input for that batch.

For modules with different inference semantics, `mode("eval")` may activate a
separate evaluation path. `compile(...)` follows that same semantic boundary
instead of asking modules for compiler-specific hooks.

Custom modules are authored with `compute.module(...)`, while built-in `nn`
layers keep the model-facing conveniences such as `parameters`, `save`, and
`load`. For module-level checkpoint files, the public namespace is
`affon:checkpoint`.

`module.metadata(path)` assigns a logical tree path to the module and propagates
derived paths to submodules from the module state tree:

```ts
const model = compute.module({
  encoder: nn.Linear(4, 8),
}, (state, x) => state.encoder(x)).metadata('model')

model.module_path         // "model"
model.encoder.module_path // "model.encoder"
```

Derived path segments use the actual state keys. Prefer `snake_case` state keys
when authoring modules so graph export, diagnostics, and inspection tooling see
the same names as the module state tree. Explicit child metadata is preserved
exactly as authored.

## Parameters

A parameter is a trainable compute tensor.

Examples:
- `Linear.weight`
- `Linear.bias`
- `RNN.weight_ih`
- `RNN.weight_hh`

You access them through:

```ts
model.parameters
model.parameters.named()
```

## Shapes

The most important rule:

- the last dimension is often the local feature width
- earlier dimensions describe structure such as batch or sequence

Examples:

```txt
[batch, features]
[seq_len, input_size]
[seq_len, batch, input_size]
```

## Sequence Helpers

`affon:nn` also carries a small set of sequence-building helpers that are
useful across transformer-style and other sequence models:

- `nn.causal_mask(length, opts?)`
  Creates a lower-triangular visibility mask for autoregressive attention.
- `nn.apply_causal_mask(scores, value?)`
  Applies a causal mask to attention scores before softmax.
- `nn.position_ids(length)`
  Creates `[0, 1, 2, ...]` position ids as a tensor.
- `nn.sinusoidal_encoding(length, dim, opts?)`
  Creates fixed sinusoidal positional encodings.

These are model-building creators, not workflow utilities. They belong with the
shared `nn` layer surface so packages like `transformers` can consume them
directly instead of re-exporting their own copies.

## Diagnostics

`nn.diagnostics` owns module-level assertion policy. Runtime compute still owns
raw numeric inspection such as `finite_summary(...)`; diagnostics decides when a
module assertion site should be checked and whether it warns or errors.

```ts
nn.diagnostics.configure({
  mode: 'error',
  include: ['finite:decoder.blocks.*'],
})
```

Assertion filters use `<kind>:<path>` keys. Empty `include` means all sites for
the selected mode; `exclude` wins over `include`.

## Terminology

- `module`
  The precise structural term in `affon:nn`.

- `model`
  A user-facing term for a composed module used for training or inference.

- `layer`
  A specific module such as `Linear`, `RNN`, `LSTM`, or `Dropout`.

For the broader vocabulary:
- [glossary.md](../glossary.md)
