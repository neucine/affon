# Native model playground

The `serve/` app is one native Affon process. Hao's `std:http.serve()` serves
the browser UI and JSON API; Affon loads DistilGPT-2 and ViT once and performs
inference directly. Bun, Node and Python are not required for serving.
Set `AFFON_DEVICE=metal` for Apple GPU execution; the default is CPU.
This is a local demo. Optional SmolLM2 adds instruction following.

Run from the repository root, using a build that includes Hao's HTTP server:

```sh
AFFON_DEVICE=metal \
AFFON_HF_CACHE="$PWD/apps/hf-inference/artifacts/hf-cache" \
/tmp/affon-hf-release/bin/affon apps/hf-inference/src/serve/server.ts
```

Open http://127.0.0.1:8765 after both models report ready. `AFFON_SERVE_PORT`
changes the port; `AFFON_HF_OFFLINE=1` requires an existing verified cache.
Use Ctrl-C to stop the process. Model loading finishes before the listener opens.
The UI source is `apps/hf-inference/public/index.html`, loaded relative to the repository root.

```sh
curl http://127.0.0.1:8765/api/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"The future of computing is","max_new_tokens":24}'
```

`GET /api/health` reports readiness and busy state. Requests support 1–64 new
tokens and up to 256 prompt tokens (4096 characters at the API boundary).
Invalid model inputs return 400 and HTTP bodies over 16 MiB return 413.
Requests that observe active inference return 429. Synchronous inference blocks
the event loop, so later requests can wait until it completes, including health
checks. There is no worker isolation or guaranteed immediate busy response.
Generation is greedy and executes explicit full-prefix Programs with one-token
output windows. No KV cache is claimed.

### Image classification

The playground now loads both DistilGPT-2 and
`google/vit-base-patch16-224@3f49326eb077187dfe1c2a2bb15fbd74e6ab91e3` once.
Choose **Image classification**, upload a photo, and click **Classify uploaded image**.
The result shows the top five ImageNet labels and softmax scores normalized
across all 1,000 classes. Text generation remains available in the second tab.

Image files are decoded locally by the browser, with transparency composited
onto white. Photos larger than 512 pixels on their longest side are reduced
in the browser before transport. Affon then applies the model's bilinear
224×224 resize and normalization and runs ViT on the selected device. This is
browser decoding, not native image-file decoding in Hao/Affon; large-image
browser resizing is an additional preprocessing step outside the parity audit.
The known strict ViT hidden-state discrepancies remain open.

The file selector accepts browser-supported PNG, JPEG, WebP, and AVIF images
up to 20 MiB. The server accepts packed RGB8 bytes via
`POST /api/classify?width=W&height=H` with `Content-Type: application/octet-stream`.
Each side must be 1–512 pixels, and the byte count must equal `W*H*3`.
Images are not saved or sent to external services. Both tasks run on the same JS thread and serialize inference.

Use `AFFON_HF_VISION_CACHE` to select a separate vision snapshot cache; otherwise
both models use `AFFON_HF_CACHE`. Offline mode requires both snapshots. Startup
loads both models, so readiness takes longer and memory usage is higher than
in the text-only demo.

Browser validation: uploading the audit's 320×256 synthetic RGB PNG produced
class 733 (`pole`) as the top prediction, matching the independent reference.
The measured Metal preprocessing/forward/ranking time was 1.04 seconds. This
checks the upload path, not real-image accuracy. Text completion still passes
its known four-token smoke case, and malformed image dimensions/byte counts
return HTTP 400.

### Classify an image URL

Paste a direct HTTP(S) image link into **Or paste an image URL**, then click
**Download & classify**. The native Affon server downloads the image so cross-origin
browser restrictions do not block decoding. The browser shows a preview and
then submits the same RGB classification request used for file uploads.

`POST /api/image-url` accepts JSON `{ "url": "https://.../photo.jpg" }` and
returns the image bytes. It accepts PNG/JPEG/WebP/AVIF response types, follows
up to three redirects, limits the decoded download to 20 MiB, and returns a timeout error
after 30 seconds. Hao does not yet cancel in-flight client I/O; the single
download slot remains occupied until that I/O finishes. Browser cookies and authorization headers are not forwarded.
Unsupported schemes, embedded URL credentials, HTML pages, and oversized
responses fail with a visible error. Images are held in memory, not saved.
URL downloads contact the specified image host; model inference remains local.

Run `affon apps/hf-inference/tests/serve/image-url-check.ts` for download, redirect,
content-type, empty-response, and decoded byte-limit checks.

### Playground source layout

```text
apps/hf-inference/
  src/
    inference/              # Model loading, generation, preprocessing, ranking
    serve/                  # Native HTTP entry point, routes, configuration
  public/                   # HTML, CSS, browser JavaScript
  tests/                    # Inference/transport tests and fixtures
  audit/                    # Parity tools, reference generators, reports
```

The app's inference orchestration lives in the sibling `inference/` layer.
The app imports HF loaders and processors from `packages/@affon/huggingface/src/`.
Those loaders construct GPT-2, BERT, and ViT from `packages/@affon/models/src/`. `server.ts` remains
the launch entry point. Run from the repository root so static assets resolve.

```sh
affon test apps/hf-inference/tests/inference/classification.test.ts
affon apps/hf-inference/tests/serve/image-url-check.ts
```

### SmolLM2 instructions

Build the updated runtime, then enable SmolLM2 alongside DistilGPT-2:

```sh
zig build -Doptimize=ReleaseFast --prefix /tmp/affon-smollm-release install
AFFON_SMOLLM2=1 AFFON_DEVICE=metal AFFON_HF_CACHE="$PWD/apps/hf-inference/artifacts/hf-cache" \
  /tmp/affon-smollm-release/bin/affon apps/hf-inference/src/serve/server.ts
```

In **Text generation**, select **SmolLM2 135M Instruct**. Each request is a fresh
single-turn instruction; no conversation history is retained. The server applies
the pinned chat template, stops on `<|im_end|>`, and returns only assistant text.
Token-limit truncation is shown in the UI. The 256-token prompt limit includes
chat formatting. SmolLM2 is optional; without the flag, only DistilGPT-2 appears.
Set `AFFON_SMOLLM2=all` to load 135M, 360M, and 1.7B together. Their API keys are
`smollm2`, `smollm2-360m`, and `smollm2-1.7b` respectively. With offline mode
enabled, every requested snapshot must already be cached.

`POST /api/generate` accepts any advertised text-model key; `"distilgpt2"` is
the default unless text-only mode is active. Unknown or unavailable models
return 400. `GET /api/health` advertises `text_models`.

```sh
curl http://127.0.0.1:8765/api/generate -H 'Content-Type: application/json' \
  -d '{"model":"smollm2","prompt":"What is the capital of France?","max_new_tokens":24}'
```

The tiny model is experimental: factual accuracy and reasoning remain limited.

For a single larger model, choose `AFFON_SMOLLM2=360M` or `1.7B`. Loading all
sizes uses substantially more memory. On a memory-constrained machine use
text-only serving with one size:

```sh
AFFON_SMOLLM2=1.7B AFFON_TEXT_ONLY=1 AFFON_DEVICE=metal \
  AFFON_HF_CACHE="$PWD/apps/hf-inference/artifacts/hf-cache" \
  /tmp/affon-smollm-release/bin/affon apps/hf-inference/src/serve/server.ts
```

This makes SmolLM2 the default and disables image/audio features. Health includes
`default_text_model`; generation requests may omit `model`. `AFFON_DEVICE=cuda`
is accepted for a Linux CUDA build, but Mac measurements do not validate CUDA.

### Live text generation

The browser renders a new decoded text snapshot after each generated token.
Native HTTP responses remain buffered: `POST /api/generate/start` accepts the
same JSON input as `/api/generate` and returns a session `id`. Repeated
`POST /api/generate/next` requests with `{id}` each compute exactly one token,
returning the normal result fields plus `done`. The browser stops on EOS or the
token budget. Full-prefix decoding preserves byte-level Unicode boundaries.

Only one generation session may own the model at a time. `POST
/api/generate/cancel` releases it; idle sessions expire after 90 seconds.
Stop finishes the current token step, then releases the caller-owned runtime.
Closing the tab stops further token requests, and idle expiry releases that runtime.
Other inference requests receive 429 while a session is reserved. The original
buffered `/api/generate` endpoint remains available.
