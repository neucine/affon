# Decoder LM App

This is the canonical Program-native decoder-only language-model application.

`src/model.ts` authors shape-specialized forward and scalar loss Programs from
the same callable vocabulary used by the rest of Affon. Linear projections,
position embeddings, normalization, and cross entropy come from `affon:nn`;
decoder blocks and causal attention are reusable Programs. The model definition
owns no Session, evaluated tensors, parameters, or mutable mode.

The workflow demonstrates the intended execution boundary:

```ts
import { Session, optimize } from 'affon:compute'
import { adamw } from 'affon:optim'

const model = DecoderModel(vocabulary, width, options)
const forward = model.forward(batchSize, sequenceLength)
const train = optimize(
  forward,
  model.objective,
  adamw({ learning_rate: 3e-4, weight_decay: 0.01 }),
)

const session = new Session({ device: 'cpu' })
const state = session.initialize(train, { seed: 7 })
const logits = session.compile(forward).run({ token_ids }, state)
const loss = session.compile(train).run({ token_ids, labels }, state)
```

The boundaries are deliberate:

- `model.forward(batch, length)` is the reusable inference Program.
- `model.objective` is the specialized `{ input, target }` loss callable.
- `optimize(forward, objective, optimizer)` is the high-level training transform.
- `model.loss(batch, length)` materializes the same objective as a standalone
  Program for evaluation against already-computed logits.

Training, evaluation, checkpointing, and generation receive the model, Session,
and ExecutionState explicitly. Inspect authored structure with
`model.forward(batch, length).inspect()`. Decoder blocks and causal attention
are reusable child Programs, so every copied node carries an automatically
generated composition path such as `blocks.0 (decoder_block) / attention
(decoder_attention)`. Normalization and projections are parameterized leaf
callables with dotted names such as `blocks.0.attention.norm.weight` and
`blocks.0.attention.query.weight`. The token embedding table is intentionally
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
