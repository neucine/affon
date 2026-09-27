# @affon/huggingface

Experimental Hugging Face integration for native Affon inference. This package
owns HF configuration interpretation, weight mapping, model/task dispatch, and
processor integration. Native GPT-2, BERT, and ViT execution is owned by
`@affon/models`. These adapters map HF configuration and checkpoint tensors into
the model constructors; processors and checkpoint validation remain here.

Native checkpoint adapters live in `src/adapters/`. The shared
`encoder-checkpoint.ts` validates, places, and converts checkpoint tensors; it
does not implement forward operations. ViT image processing lives separately
in `src/processors/vit.ts`. Public loader and processor exports are unchanged.

Family input preparation lives in `src/processors/`, with reused audio and RGB
helpers in `processors/shared/`. Hub snapshot/cache handling lives in
`src/hub/snapshot.ts`. The root `model.ts`, `processor.ts`, and `pretrained.ts`
remain dispatch/composition entry points. Whisper graph loading lives in `adapters/whisper.ts`; execution and decoding
are owned by `models/src/whisper/`. Generic ONNX classifier integration remains
separate from preprocessing.

It has no Python or PyTorch execution dependency. The audit uses Python only
to prepare artifacts and generate independent reference results.

## Load directly from the Hub

```ts
import { from_pretrained } from '@affon/huggingface'

const { model, processor } = await from_pretrained('distilbert/distilgpt2', {
  revision: '2290a62682d06624634c1f46a6ad5be0f47f38aa',
  cache_dir: '/tmp/affon-hub-cache',
  task: 'text-generation',
  device: 'cpu',
})
const ids = model.generate(processor.encode('The future of computing is'), 4)
console.log(processor.decode(ids))
```

For vision, select `task: 'image-classification'` with the supported ViT
checkpoint. Its processor exposes `process(rgb)` and the model accepts the
resulting pixels. `load_processor(directory, options)` provides the same
selection for existing local artifacts. Inputs remain domain-specific.

`snapshot_download(model_id, options)` exposes acquisition separately. It
requires an explicit 40-character commit SHA, downloads only recognized model
and processor files, and records their sizes and SHA-256 hashes. Files are
published by rename; the completion manifest is written last. Cache reuse
verifies every recorded file. Set `local_files_only: true` to prohibit Hub
access; an incomplete or altered cache fails explicitly. Integrity failures
require removing the affected snapshot and downloading it again.

The transport uses Hao native HTTP with bounded-memory file streaming and
incremental SHA-256; curl is no longer required. Cache maintenance still uses
`shasum`, `mkdir`, `mv`, and `rm` on PATH (validated on macOS). Downloads have a 500 MiB weight-file
limit and a 20 MiB limit per JSON artifact. Checksums detect subsequent cache
corruption; they are computed locally, not publisher signatures. Private/gated
repositories, branch/tag resolution, resumable downloads, timeouts/retries, and Windows portability remain future work. No tokens are read or sent.

## Local model loading

```ts
import { load_model, load_bert_processor } from '@affon/huggingface'

const directory = '/models/bert'
const processor = load_bert_processor(directory)
const model = load_model(directory, { task: 'feature-extraction', device: 'cpu' })
const batch = processor.encode_batch(['Hello world', 'A second sentence'])
const result = model.forward(batch.input_ids, batch.attention_mask, batch.token_type_ids)
```

The task is explicit and determines the TypeScript return contract. Architecture
selection reads `config.json`; unsupported model/task combinations fail before
weight loading. Inputs retain their domain-specific shapes.

| Task | Model type | Forward input | Current boundary |
| --- | --- | --- | --- |
| `text-generation` | `gpt2` | One array of token IDs | Tied head, `gelu_new`, cached greedy generation |
| `feature-extraction` | `bert` | Batched IDs, attention masks, type IDs | Absolute positions, GELU, base encoder with pooler |
| `image-classification` | `vit` | Batch-one f32 NCHW tensor | Fixed-size RGB, biased QKV, GELU, classifier |

Explicit `load_gpt2`, `load_bert`, and `load_vit` loaders are also exported.
`load_bert_processor` applies templates and padding; `process_rgb_image` handles
the supported ViT RGB8 resize/rescale/normalize pipeline. GPT-2 tokenization
uses `@affon/tokenizers`.

## Artifacts and compatibility

Requires `config.json` and one f32 `model.safetensors`, plus the appropriate
processor files, either downloaded directly or already local. Shards, pickle
conversion, reduced-precision weights, custom Python execution, and a universal
`pipeline` API are not supported. Weight names and shapes are checked strictly.
Legacy GPT-2 causal-mask buffers are accepted only after validating their full
contents; legacy ViT scalar sizes and omitted preprocessing defaults are handled.

## Ownership

- This package owns the HF-specific model implementations and compatibility
  checks, shared loading API, and domain processor integration.
- `@affon/models` owns model definitions and shared blocks;
  `@affon/tokenizers` owns tokenization algorithms and tokenizer JSON support.
- Compute kernels belong in `affon:compute`/`affon:nn`; low-level tensor
  persistence belongs in `affon:checkpoint`.
- Reference generation, comparisons, diagnostics, and reports stay in
  [the audit app](../../../apps/hf-inference/README.md).

Supporting another compatible checkpoint reuses its model-family implementation.
Acquisition and processor selection are shared across supported tasks.

## Validation

From the repository root:

```sh
./zig-out/bin/affon test packages/@affon/huggingface/test
./zig-out/bin/affon test apps/hf-inference
```

The audit app runs the shared `load_model` API against prepared real-model
fixtures. Its README documents preparing and running those larger checks.

Direct Hub smoke checks require no Python or reference artifacts:

```sh
AFFON_DEVICE=cpu /tmp/affon-hf-release/bin/affon apps/hf-inference/audit/hub-smoke.ts
AFFON_DEVICE=cpu AFFON_HF_FAMILY=vit /tmp/affon-hf-release/bin/affon apps/hf-inference/audit/hub-smoke.ts
```

Build that optimized binary as described in the audit README. Set
`AFFON_HF_CACHE` for a different cache directory, or `AFFON_HF_OFFLINE=1` to
repeat using only the verified cache. The vision smoke uses synthetic RGB data,
not an image decoder or a semantic classification benchmark.

## GPT-2 KV caching

`generate(ids, max_new_tokens)` uses a fresh request-local KV cache by default.
It evaluates the prompt once, then processes only the newly generated token.
Use `generate(ids, max_new_tokens, { use_cache: false })` to retain full-prefix
execution for diagnostics. Generation releases its cache on completion or error.

For incremental callers:

```ts
const session = model.create_session()
const prefill = session.forward(prompt_ids)
const next = session.forward([next_token_id]) // pass new tokens only
console.log(session.length)
session.reset()
```

`session.forward` returns logits and hidden states for the new chunk only.
Position IDs continue from the cached length. Multi-token chunks use an offset
causal mask. Sessions are independent; invalid steps preserve prior context.
The total processed length cannot exceed `config.n_positions`.

Caches store f32 keys/values in sequence-major layout on the model's device.
Appending currently allocates/copies growing buffers; this is not a paged or
in-place cache. Persistent K/V payload is `2 * layers * tokens * hidden_size * 4`
bytes (36 KiB per cached token for DistilGPT-2), plus temporary allocations.
Call `reset()` or release a manually managed session when it is no longer needed.

## ONNX execution backend

ONNX is a graph representation, not another architecture. The separate
`@affon/onnx` package owns conversion and execution; HF owns configuration,
labels, processors and task dispatch. Native adapters remain the default.

```ts
const model = load_model('/models/vit-hf-config', {
  task: 'image-classification',
  backend: 'onnx',
  graph_dir: '/models/vit-converted',
  device: 'metal',
})
const processor = load_processor('/models/vit-hf-config', {
  task: 'image-classification', device: 'metal',
})
const { output: logits } = model.forward(processor.process(rgb))
```

The HF directory supplies `config.json`; the prepared graph directory supplies
`graph.json` and `weights.safetensors`. No native architecture weights or adapter
are loaded for this backend. The classifier preserves HF labels/configuration
and returns `{ output }`; hidden states and generation are not promised.
Only image classification is currently bound to the graph backend. It requires
one NCHW RGB input and `[batch, classes]` output. Names are inferred when unique;
use `input_name`/`output_name` to bind explicitly. Label counts must agree.

Use configuration and processor artifacts from the same model used to export
the graph. Shape/label checks do not prove weight provenance. ViT and MobileNetV2
have passed this HF path. `load_processor` supports native MobileNetV2 RGB8
shortest-edge bilinear resize, center crop, rescaling and normalization. The
supported configuration requires these stages enabled, a crop fitting inside
the resized image, and three-channel mean/std; unsupported configurations fail. Processor selection is separate
from the backend and continues to reject unsupported configurations.

This initial API is local `load_model` integration. `from_pretrained` remains the
native-adapter Hub convenience API; it does not acquire or prepare ONNX bundles.
The [ONNX package](../onnx/README.md) documents preparation and capability limits.

### Audio classification (AST via ONNX)

```ts
import { load_model, load_processor, decode_wav } from './src/index.ts'
const model = load_model('/models/ast/source', {
  task: 'audio-classification', backend: 'onnx', graph_dir: '/models/ast', device: 'metal',
})
const processor = load_processor('/models/ast/source', {task: 'audio-classification', device: 'metal'})
const audio = decode_wav(wavBytes)
const logits = model.forward(processor.process(audio.samples, audio.sampling_rate)).output
```

The HF layer decodes bounded RIFF/WAVE PCM8/16/24/32 or IEEE float32, averages
stereo channels, and resamples 8–96 kHz input to 16 kHz with a windowed-sinc
anti-alias filter. `decode_wav` accepts 25 ms–30 s recordings. Unsupported codecs,
extensible WAV, malformed chunks and nonfinite samples fail explicitly.

The AST processor computes 400-sample Hann windows, 160-sample hops, a 512-point
FFT, Kaldi mel filters, log energy, fixed-length padding/truncation, and model
normalization. It matches the Transformers NumPy fallback at 16 kHz; resampling
is Affon's implementation, not a claim of bitwise parity with every audio library.
Processing runs in TypeScript inside Affon; only the resulting tensor and graph
inference use the selected compute device. No runtime Python or TorchAudio.

Current ONNX audio binding is AST's `[batch, frames, mel_bins]` input and
`[batch, classes]` logits. Audio has no native architecture adapter or Hub
`from_pretrained` shortcut yet. The 35-way speech-command checkpoint is a
single-label classifier; it is not ASR, a general sound classifier, or a detector
of silence/unknown words. See the app's audio report for tested coverage.

### Automatic speech recognition (Whisper)

```ts
const model = load_model('/models/whisper/source', {
  task: 'automatic-speech-recognition', backend: 'onnx',
  graph_dir: '/models/whisper', device: 'metal',
})
const processor = load_processor('/models/whisper/source', {
  task: 'automatic-speech-recognition', device: 'metal',
})
const audio = decode_wav(wavBytes)
const result = model.transcribe(processor.process(audio.samples, audio.sampling_rate))
```

This bounded integration supports the prepared `openai/whisper-tiny.en` bundle.
It uses a centered 400-point STFT, Slaney Mel filters and 30-second zero padding.
Frontend DSP runs in Affon TypeScript; encoder/decoder graphs run on the selected
device. The Whisper model orchestrates encoder output, per-request self/cross-attention
caches, suppression rules and token generation; the HF adapter supplies text decoding. The graph executor remains independent of
HF architecture names. The local-only ONNX task API is required; `from_pretrained`
still loads native architecture adapters only. See the app README for preparation
and the short-WAV/254-token limits. No timestamps or silence detection are promised.

Whisper feature extraction uses compute's native CPU `stft_power` and `filterbank`
primitives. HF owns the periodic Hann window, Slaney filters, 30-second padding,
frame selection and log normalization; the final features move to the requested
device. This requires an Affon runtime containing the spectral primitives and
removes the previous TypeScript FFT implementation. No Python or external audio
process is used at inference time.
