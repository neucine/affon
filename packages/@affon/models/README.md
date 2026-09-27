# @affon/models

Model definitions and their execution behavior, built on `affon:compute` and
`affon:nn`. This package replaces `@affon/lm` and `@affon/transformers`.

- `src/gpt2/`: tied-head GPT-2 forward, request-local cache sessions, and greedy generation.
- `src/bert/`: absolute-position BERT encoder, hidden states, and pooling.
- `src/vit/`: fixed-size RGB ViT classifier and hidden states.
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

ONNX model orchestration (including Whisper) still needs its own boundary review.
Strict ViT reference parity has existing gaps; extraction does not claim to fix
those or improve performance. The app test process reports 2 live Hao resources
(1,672 bytes) at shutdown; that diagnostic remains uninvestigated.

## Extraction validation (2026-09-28)

Model, decoder-app, and HF API type checks pass. The combined model/HF/decoder-app
suite passes 82 tests with 2 declared skips using the softmax-scope runtime.
Direct-construction tests exercise each family without HF files and reject
invalid parameter shapes. Existing GPT-2 cache, generation, and EOS tests pass.
BERT passes the existing Metal independent-reference audit. ViT logits and every
hidden state exactly match pre-extraction execution for both reference inputs
on CPU and Metal; the independent-reference audit still reports its known gaps.
This validation makes no performance claim.
