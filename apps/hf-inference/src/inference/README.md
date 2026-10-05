# Inference layer

Reusable application-level inference, independent of HTTP, browser assets,
environment variables, and server configuration.

- `models.ts`: `load_models(options)`, pinned model metadata, and shared types.
- `text.ts`: `generate_text(models, prompt, max_new_tokens, model_key?)`.
- `image.ts`: `classify_image(models, rgb_bytes, width, height, device)`.
- `classification.ts`: stable softmax and top-five ranking.

Callers supply validated inputs and explicit model-loading options. The HTTP
layer validates request bodies; another caller can perform its own validation.
Model adapters and processors are shared from `packages/@affon/huggingface`.

```ts
import { load_models } from './models.ts'
import { generate_text } from './text.ts'

const cache = '/absolute/path/to/repo/apps/hf-inference/artifacts/hf-cache'
const models = await load_models({
  device: 'metal',
  cache_dir: cache,
  vision_cache_dir: cache,
  local_files_only: true,
})
const result = generate_text(models, 'The future of computing is', 4)
console.log(result.text)
```

Run from the repository root:

```sh
affon test apps/hf-inference/tests/inference/classification.test.ts
```

Set `smollm2: "135M"`, `"360M"`, `"1.7B"`, or `"all"` in load options; `true`
aliases 135M. Pass an advertised model key to generation. With `text_only: true`,
only the selected SmolLM2 model or models load and omitted generation keys select
135M. Otherwise DistilGPT-2 remains the default. `device` accepts CPU, Metal, or
CUDA.
