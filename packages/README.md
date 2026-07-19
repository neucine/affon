# First-Party Packages

First-party packages sit above the runtime and below complete apps.

Package ownership follows three axes:

- architecture packages own reusable model structure
- domain packages own data and task conventions
- model-family packages own stable family-level assembly and conventions

Complete runnable workloads should live under `../apps/` once they outgrow focused package examples.

## Current Packages

- `transformers/`: transformer architecture blocks and helpers.
- `tokenizers/`: tokenizer implementation and ecosystem compatibility.
- `lm/`: language-model family APIs and reusable LM corpus packing.
- `vision/`: vision-domain scaffold.
- `cnn/`: CNN architecture scaffold.
