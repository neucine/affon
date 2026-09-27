# @affon/models

Model definitions and their execution behavior, built on `affon:compute` and
`affon:nn`. This package replaces `@affon/lm` and `@affon/transformers`.

- `src/gpt2/`: tied-head GPT-2 forward, request-local cache sessions, and greedy generation.
- `src/bert/`: absolute-position BERT encoder, hidden states, and pooling.
- `src/vit/`: fixed-size RGB ViT classifier and hidden states.
- `src/whisper/`: prepared encoder/decoder execution, request-local caches, and greedy transcription.
- `src/shared/`: attention, decoder blocks, embeddings, feed-forward layers,
  and sequence helpers reused by model code.
- `src/index.ts`: deliberate public exports. Reusable block API names
  remain available; consumers should use this entry point rather than deep imports.

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

`create_gpt2`, `create_bert`, and `create_vit` accept typed, normalized config and
structured tensor parameters. They do not read files, interpret HF keys, or load
checkpoints. Parameters must be f32 on the selected device. Linear matrices use
[input, output]; ViT patch weights use [output, RGB, patch, patch]. Constructors
validate dimensions, tensor shapes, dtype, and device. The model retains tensors
for its lifetime; GPT-2 sessions own their request-local caches.

HF adapters validate supported variants, read checkpoints, map tensor names and
layouts, and call these constructors. Their returned `config` remains the HF
config; direct constructors expose the normalized model config. HF GPT-2 retains
its mutable EOS setting. Image/token processors remain in the integration layer.

## Remaining work

Generic ONNX classifier integration still needs its own boundary review.
Strict ViT reference parity has existing gaps; extraction does not claim to fix
those or improve performance.
