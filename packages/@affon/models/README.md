# @affon/models

Program-first model definitions built on
`affon:compute`.

- `src/gpt2/`: tied-head GPT-2 Program authoring.
- `src/llama/`: tied-head, bias-free Llama Programs with RMSNorm, unscaled RoPE, and GQA.
- `src/bert/`: absolute-position BERT encoder, hidden states, and pooling.
- `src/vit/`: fixed-size RGB ViT classifier and hidden states.
- `src/shared/`: model-boundary validation shared by the family constructors.
- `src/index.ts`: deliberate family-level public exports.

Keep family-specific code with its family. Move components into `shared` when
actual consumers need them. The directory name does not make every helper public.
Models must not depend on apps, Hugging Face loading, or tokenizer implementations.

The configurable example `DecoderModel`, its loss, and its coupled generation
helper belong to `apps/decoder-lm/src` and are exported by that app. They are not
a model family or exports of this package. Their tests live with the app.
Corpus packing and token-cache persistence also belong to the app in `src/data`;
basic data operations remain in `affon:dataset`.

## Validation

Run `affon test packages/@affon/models/test` and type-check `tsconfig.json` in
this directory. Corpus and training workflow tests live under `apps/decoder-lm/test`.

## Construction and integration

`create_gpt2`, `create_llama`, `create_bert`, and `create_vit` accept typed, normalized config and
structured tensor parameters. They return a pure Program (or shape-specialized
Program factory) and a parameter initializer. They do not create a `Session`,
own mutable execution state, read files, interpret HF keys, or load checkpoints.
Parameters must be f32. Linear matrices use
[input, output]; ViT patch weights use [output, RGB, patch, patch]. Constructors
validate dimensions, tensor shapes, and dtype. The caller selects a device,
initializes `ExecutionState`, compiles, executes, and disposes resources.

```ts
const model = create_gpt2(config, weights)
const source = model.forward(tokenIds.length)
const session = new Session({ device: 'metal' })
const state = session.initialize(source, { parameters: model.parameters })
const outputs = session.compile(source).run({ ids, positions, mask }, state)
```

`forward(length, outputStart)` returns only the requested suffix while still
computing the declared full prefix. It is an output window, not a KV cache.
Generation and request history are application policies and do not live in this
model-definition package.

HF adapters validate supported variants, read checkpoints, map tensor names and
layouts, and call these constructors. Their returned `config` remains the HF
config; direct constructors expose the normalized model config. Image/token
processors remain in the integration layer.

## Remaining work

Strict ViT reference parity has existing gaps; extraction does not claim to fix
those or improve performance.
