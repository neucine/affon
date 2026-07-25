# First-Party Packages

First-party packages live under the `@affon/` scope, above the runtime and below complete apps.

Package ownership follows three axes:

- architecture packages own reusable model structure
- domain packages own data and task conventions
- model-family packages own stable family-level assembly and conventions

Complete runnable workloads should live under `../apps/` once they outgrow focused package examples.

## Current Packages

- `@affon/transformers/`: transformer architecture blocks and helpers.
- `@affon/tokenizers/`: tokenizer implementation and ecosystem compatibility.
- `@affon/lm/`: language-model family APIs and reusable LM corpus packing.
- `@affon/vision/`: vision-domain scaffold.
- `@affon/cnn/`: CNN architecture scaffold.
