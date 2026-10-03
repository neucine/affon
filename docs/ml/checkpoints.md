# Checkpoints

`affon:checkpoint` persists named tensor values. Program definitions remain
device-neutral; save and restore the tensors owned by an `ExecutionState`.

## Core Shape

```ts
import checkpoint from 'affon:checkpoint'
import { Session } from 'affon:compute'

const session = new Session({ device: 'cpu' })
const state = session.initialize(model, { seed: 7 })

checkpoint.save(state.parameters, 'model.safetensors')
const parameters = checkpoint.load('model.safetensors')
const restored = session.initialize(model, { parameters })
```

`Session.initialize(...)` validates names, shapes, and dtypes against the
Program. It creates a new `ExecutionState`; checkpoint loading does not mutate a
Program or an existing state in place.

Loaded tensors are temporary initializer values. In a long-running process,
dispose them after `Session.initialize(...)` has copied the values into the new
state.

To restore non-parameter model state, save it separately and pass it as
`model_state`:

```ts
const restored = session.initialize(model, {
  parameters: checkpoint.load('model.safetensors'),
  model_state: checkpoint.load('model-state.safetensors'),
})
```

Loading accepts SafeTensors `F32`, `F64`, and `I64` entries. `BF16` storage is
widened exactly to f32 on CPU; BF16 execution and writing are not supported.
Loading ignores the optional `__metadata__` string mapping used by external
producers. Unsupported dtypes and malformed tensor entries are rejected.
Reading a file does not perform architecture or weight-name conversion.

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
  state: state.parameters,
  tensorGroups: {
    modelState: state.model_state,
  },
  manifest: {
    format: 'my-training-checkpoint/v1',
    epoch: 2,
    step: 180,
  },
})

const bundle = checkpoint.loadBundle('artifacts/run-1/epoch-2')
const restored = session.initialize(model, {
  parameters: bundle.state,
  model_state: bundle.tensorGroups.modelState,
})
```

This writes:

- `<prefix>.json`
- `<prefix>.safetensors`
- optional named tensor-group files like `<prefix>.optimizer.safetensors`

The manifest is application-owned. Use it to version any optimizer-step,
scheduler, or data-position policy required to resume a run.

## Scope

Checkpoint APIs are for local state persistence. Deployment exports, model hub
artifacts, processor configuration, and architecture adaptation belong to their
respective packages and applications.
