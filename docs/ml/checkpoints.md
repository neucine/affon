# Checkpoints

`affon:checkpoint` owns training-oriented persistence.

Use it when you want to:

- save model or optimizer state to disk
- load named tensor state back from disk
- restore an existing module or state tree in place

## Core Shape

```ts
import checkpoint from 'affon:checkpoint'
import nn from 'affon:nn'

const model = nn.Sequential(
  nn.Linear(3, 4),
  nn.Linear(4, 2),
)

checkpoint.save(model.state(), 'model.safetensors')
const state = checkpoint.load('model.safetensors')
checkpoint.restore(model, state)
```

Loading accepts SafeTensors `F32`, `F64`, and `I64` entries. `BF16` storage is
widened exactly to f32 on CPU; BF16 execution and writing are not supported.
Loading ignores the
optional `__metadata__` string mapping used by external producers such as
Hugging Face. Unsupported dtypes and malformed tensor entries are rejected.
Reading a file does not perform model architecture or weight-name conversion.

## Selective loading

```ts
const metadata = checkpoint.inspect('model.safetensors')
// Reports original storage dtype and shape without allocating tensor payloads.
const selected = checkpoint.load('model.safetensors', {names: ['model.embed_tokens.weight']})
```

The loader seeks directly to selected tensors and streams BF16 widening through
64 KiB chunks. It validates all tensor metadata even for an empty selection;
missing or duplicate requested names are errors. JSON headers are limited to
16 MiB. The previous 500 MiB whole-file limit is removed. Loading all tensors
still requires memory for all resulting values. HF index/shard resolution lives
in `@affon/huggingface`, above this single-file API.

## Bundles

For multi-file training checkpoints, use the bundle helpers:

```ts
import checkpoint from 'affon:checkpoint'

checkpoint.saveBundle('artifacts/run-1/epoch-2', {
  state: model.state(),
  tensorGroups: {
    optimizer: optimizerState.tensors,
  },
  manifest: {
    format: 'my-training-checkpoint/v1',
    epoch: 2,
    step: 180,
  },
})

const bundle = checkpoint.loadBundle('artifacts/run-1/epoch-2')
checkpoint.restore(model, bundle.state)
```

This writes:

- `<prefix>.json`
- `<prefix>.safetensors`
- optional named tensor-group files like `<prefix>.optimizer.safetensors`

## Module Convenience

Compute-backed modules still expose:

- `model.save(path)`
- `model.load(path)`

Those are convenience methods for the module instance itself. The public
module-level persistence surface is `affon:checkpoint`, not `affon:compute` or
`affon:nn`.

## Scope

`affon:checkpoint` is for resume-oriented state persistence.

It does not currently define:

- deployment export bundles
- model publication artifacts
- hub/download packaging

Those are separate concerns from checkpointing.
