# Inference layer

Reusable application-level inference, independent of HTTP, browser assets,
environment variables, and server configuration.

- `models.ts`: `load_models(options)`, pinned model metadata, and shared types.
- `text.ts`: `generate_text(models, prompt, max_new_tokens)`.
- `image.ts`: `classify_image(models, rgb_bytes, width, height, device)`.
- `classification.ts`: stable softmax and top-five ranking.

Callers supply validated inputs and explicit model-loading options. The HTTP
layer validates request bodies; another caller can perform its own validation.
Model adapters and processors are shared from `packages/@affon/huggingface`.

```ts
import { load_models } from './models.ts'
import { generate_text } from './text.ts'

const models = await load_models({
  device: 'metal',
  cache_dir: '/tmp/affon-hub-cache',
  vision_cache_dir: '/tmp/affon-hub-cache',
  local_files_only: true,
})
const result = generate_text(models, 'The future of computing is', 4)
console.log(result.text)
```

Run from the repository root:

```sh
affon test apps/hf-inference/tests/inference/classification.test.ts
```
