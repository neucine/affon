# Decoder LM App

This is the canonical Program-native decoder-only language-model application.

`src/model.ts` authors shape-specialized forward and scalar loss Programs. The
loss composes the same decoder Program used for inference, and `train(...)` is
the ordinary `optimize(model, loss, optimizer)` transform. The model definition owns
no Session, tensors, parameters, or mutable mode.

The workflow demonstrates the intended execution boundary:

```ts
const model = DecoderModel(vocabulary, width, options)
const loss = model.loss(batchSize, sequenceLength)
const session = new Session({ device: 'cpu' })
const state = session.initialize(loss, { seed: 7 })
const executable = session.compile(model.forward(batchSize, sequenceLength))
const logits = executable.run({ token_ids }, state)
```

Training, evaluation, checkpointing, and generation receive the model, Session,
and ExecutionState explicitly. Inspect authored structure with
`model.forward(batch, length).inspect()`; there is no eager/captured-graph mode
or Module-shaped compatibility object.

Corpus preparation and token-window persistence live in `src/data`. Complete
workflow configuration is handled by `src/workflow.ts`.

Run the application tests from the repository root:

```sh
./zig-out/bin/affon test apps/decoder-lm/test
bun x tsc -p apps/decoder-lm/tsconfig.json --noEmit
```
