# First-Party Packages

First-party packages sit above the runtime and below complete apps. Boundaries
follow ownership and dependencies:

- `@affon/models`: model definitions, execution behavior, and shared model components.
- `@affon/huggingface`: HF artifacts, config/weight adaptation, loading, and processors.
- `@affon/tokenizers`: tokenizer implementations and compatibility.
- `@affon/onnx`: prepared graph import and execution.

Model families belong inside `models`; shared components belong in `models/src/shared`.
Generation belongs with the models that support it. Basic layers remain in
`affon:nn`, tensor execution in `affon:compute`, and basic data operations in
`affon:dataset`. Complete training and serving workflows belong in `apps/`.

The former `lm` and `transformers` packages have been consolidated into `models`.
Reusable blocks are exported by `@affon/models`. The configurable example
`DecoderModel`, its loss, and its generation helper belong to `apps/decoder-lm/src`
and are exported by the app entry point. Corpus helpers are available from
`apps/decoder-lm/src/data/index.ts`. There is no top-level decoder model family.

Native GPT-2, BERT, and ViT execution lives in `models/src/gpt2`, `bert`, and
`vit`. HF adapters translate config and checkpoint tensors into their constructor
contracts. Whisper execution lives in `models/src/whisper`, with prepared graph
loading in its HF adapter. Generic ONNX classifier integration remains a follow-up. See
[models](./@affon/models/README.md). Empty `cnn` and `vision` scaffolds were removed.
