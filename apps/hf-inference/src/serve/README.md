# Native model playground

The `serve/` app is one native Affon process. Hao's `std:http.serve()` serves
the browser UI and JSON API; Affon loads DistilGPT-2 and ViT once and performs
inference directly. Bun, Node and Python are not required for serving.
Set `AFFON_DEVICE=metal` for Apple GPU execution; the default is CPU.
This is a local demo, not a chat-tuned assistant or a production server.

Run from the repository root, using a build that includes Hao's HTTP server:

```sh
AFFON_DEVICE=metal \
AFFON_HF_CACHE=/tmp/affon-hub-cache \
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
Generation is greedy and uses a request-local KV cache.

### Image classification

The playground now loads both DistilGPT-2 and
`google/vit-base-patch16-224@3f49326eb077187dfe1c2a2bb15fbd74e6ab91e3` once.
Choose **Image classification**, upload a photo, and click **Classify uploaded image**.
The result shows the top five ImageNet labels and softmax scores normalized
across all 1,000 classes. Text completion remains available in the second tab.

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

The app's inference orchestration lives in the sibling `inference/` layer. Shared HF model
implementations and processors remain in `packages/@affon/huggingface/src/`;
the app imports them instead of duplicating model internals. `server.ts` remains
the launch entry point. Run from the repository root so static assets resolve.

```sh
affon test apps/hf-inference/tests/inference/classification.test.ts
affon apps/hf-inference/tests/serve/image-url-check.ts
```
