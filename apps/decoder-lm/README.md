# Decoder LM App

This is the canonical Program-native decoder-only language-model application.

`src/model.ts` defines a reusable model callable using the same vocabulary as
the rest of Affon. Linear projections, position embeddings, and normalization
come from `affon:nn`; decoder blocks and causal attention are reusable Programs.
The model definition owns no Session, evaluated tensors, parameters, or mutable
mode.

The workflow demonstrates the intended execution boundary:

```ts
import { Session, Tensor, optimize, program } from 'affon:compute'
import { cross_entropy } from 'affon:nn'
import { adamw } from 'affon:optim'

const decoder = DecoderModel(vocabulary, width, options)
const model = program('decoder_lm', p => decoder({
  token_ids: p.argument(
    'token_ids',
    Tensor.i64([batchSize, sequenceLength], { axes: ['batch', 'token'] }),
  ),
}, 'decoder'))
const train = optimize(
  model,
  cross_entropy(),
  adamw({ learning_rate: 3e-4, weight_decay: 0.01 }),
)

const session = new Session({ device: 'cpu' })
const state = session.initialize(train, { seed: 7 })
const logits = session.compile(model).run({ token_ids }, state)
const loss = session.compile(train).run({ token_ids, labels }, state)
```

The boundaries are deliberate:

- `DecoderModel(...)` returns a reusable `{ token_ids }` model callable.
- The callable infers batch and sequence dimensions from its formal binding.
- `program(...)` establishes the explicit executable and inspection boundary.
- `cross_entropy()` is the specialized `{ input, target }` loss callable.
- `optimize(model, objective, optimizer)` is the high-level training transform.

Training, evaluation, checkpointing, and generation receive the model, Session,
and ExecutionState explicitly. Inspect authored structure with
`model.inspect()`. Decoder blocks and causal attention
are reusable child Programs, so every copied node carries an automatically
generated composition path such as `decoder (decoder_model) / blocks.0
(decoder_block) / attention (decoder_attention)`. Normalization and projections
are parameterized leaf callables with dotted names such as
`decoder.blocks.0.attention.norm.weight` and
`decoder.blocks.0.attention.query.weight`. The token embedding table is intentionally
owned by the root Program because the tied language-model head shares that
exact parameter. There is no eager/captured-graph mode or Module-shaped
compatibility object.

Corpus preparation and token-window persistence live in `src/data`. Complete
workflow configuration is handled by `src/workflow.ts`.

Run the application tests from the repository root:

```sh
./zig-out/bin/affon test apps/decoder-lm/test
bun x tsc -p apps/decoder-lm/tsconfig.json --noEmit
```
