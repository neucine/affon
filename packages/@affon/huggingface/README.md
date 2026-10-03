# @affon/huggingface

Experimental Hugging Face integration for native Affon inference. This package
owns HF configuration interpretation, weight mapping, model/task dispatch, and
processor integration. Native GPT-2, Llama, BERT, and ViT Program definitions are
owned by `@affon/models`. These adapters map HF configuration and checkpoint tensors into
the model constructors; processors and checkpoint validation remain here.

Native checkpoint adapters live in `src/adapters/`. The shared
`encoder-checkpoint.ts` validates and converts checkpoint tensors; it
does not implement forward operations. ViT image processing lives separately
in `src/processors/vit.ts`. Public loader and processor exports are unchanged.

Family input preparation lives in `src/processors/`, with reused audio and RGB
helpers in `processors/shared/`. Hub snapshot/cache handling lives in
`src/hub/snapshot.ts`. The root `model.ts`, `processor.ts`, and `pretrained.ts`
remain dispatch/composition entry points. Whisper graph loading lives in
`adapters/whisper.ts`; it returns Programs, weights, and generation metadata.
Applications own execution state, decoding policy, and request-local caches.
Generic ONNX classifier integration remains separate from preprocessing.

It has no Python or PyTorch execution dependency. The audit uses Python only
to prepare artifacts and generate independent reference results.

## Load directly from the Hub

```ts
import { from_pretrained } from '@affon/huggingface'
import { Session } from 'affon:compute'

const { model, processor } = await from_pretrained('distilbert/distilgpt2', {
  revision: '2290a62682d06624634c1f46a6ad5be0f47f38aa',
  cache_dir: '/tmp/affon-hub-cache',
  task: 'text-generation',
  device: 'cpu',
})
const tokenIds = processor.encode('The future of computing is')
const source = model.forward(tokenIds.length)
const session = new Session({ device: 'cpu' })
const state = session.initialize(source, { parameters: model.parameters })
const ids = session.tensor(tokenIds, { dtype: 'i64' })
const positions = session.tensor(tokenIds.map((_, index) => index), { dtype: 'i64' })
const mask = session.tensor([[tokenIds.map((_, row) => tokenIds.map((_, column) => column > row ? 1 : 0))]], { dtype: 'i64' })
const [logits] = session.compile(source).run({ ids, positions, mask }, state)
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
`shasum`, `mkdir`, `mv`, and `rm` on PATH (validated on macOS). Downloads allow up to 8 GiB per weight file, 16 GiB per snapshot,
256 shards, and 20 MiB per JSON artifact. Single `model.safetensors` and indexed
`model.safetensors.index.json` snapshots are supported. Flat shard filenames,
index ownership, completeness, and duplicate tensor names are checked. Adapters
inspect metadata first and load one source tensor at a time; execution remains
f32, so BF16 storage does not halve resident model memory. Checksums detect subsequent cache
corruption; they are computed locally, not publisher signatures. Private/gated
repositories, branch/tag resolution, resumable downloads, timeouts/retries, and Windows portability remain future work. No tokens are read or sent.

## Local model loading

```ts
import { load_model, load_bert_processor } from '@affon/huggingface'

const directory = '/models/bert'
const processor = load_bert_processor(directory)
const model = load_model(directory, { task: 'feature-extraction' })
const batch = processor.encode_batch(['Hello world', 'A second sentence'])
const source = model.forward(batch.input_ids.length, batch.input_ids[0].length)
const session = new Session({ device: 'cpu' })
const state = session.initialize(source, { parameters: model.parameters })
const inputs = {
  ids: session.tensor(batch.input_ids, { dtype: 'i64' }),
  types: session.tensor(batch.token_type_ids, { dtype: 'i64' }),
  positions: session.tensor(batch.input_ids.map(row => row.map((_, index) => index)), { dtype: 'i64' }),
  valid: session.tensor(batch.attention_mask.map(row => row.map(value => [value]))),
  attention_mask: session.tensor(batch.attention_mask.map(row => [[[...row.map(value => 1 - value)]]]), { dtype: 'i64' }),
}
const outputs = session.compile(source).run(inputs, state)
```

The task is explicit and determines the TypeScript return contract. Architecture
selection reads `config.json`; unsupported model/task combinations fail before
weight loading. Inputs retain their domain-specific shapes.

| Task | Model type | Forward input | Current boundary |
| --- | --- | --- | --- |
| `text-generation` | `gpt2` | `forward(length, outputStart?)` | Tied head, `gelu_new`; Program inputs are IDs, positions, and causal mask |
| `text-generation` | `llama` | `forward(length, outputStart?)` | Tied head, SiLU, full unscaled RoPE, GQA; Program inputs are IDs and causal mask |
| `feature-extraction` | `bert` | `forward(batch, length)` | Absolute positions, GELU, base encoder with pooler |
| `image-classification` | `vit` | Fixed `forward` Program | Fixed-size RGB, biased QKV, GELU, classifier |

Explicit `load_gpt2`, `load_bert`, and `load_vit` loaders are also exported.
`load_bert_processor` applies templates and padding; `process_rgb_image` handles
the supported ViT RGB8 resize/rescale/normalize pipeline. GPT-2 tokenization
uses `@affon/tokenizers`.

## Artifacts and compatibility

Requires `config.json` and one `model.safetensors` (f32, or BF16 widened to f32), plus the appropriate
processor files, either downloaded directly or already local. Shards, pickle
conversion, reduced-precision execution, custom Python execution, and a universal
`pipeline` API are not supported. Weight names and shapes are checked strictly.
Stored GPT-2 causal-mask buffers are accepted only after validating their full
contents; ViT scalar sizes and omitted preprocessing defaults are handled.

## Ownership

- This package owns the HF-specific model implementations and compatibility
  checks, shared loading API, and domain processor integration.
- `@affon/models` owns model definitions and shared blocks;
  `@affon/tokenizers` owns tokenization algorithms and tokenizer JSON support.
- Tensor operations and parameterized layer factories belong in `affon:ops` and `affon:nn`;
  execution belongs in `affon:compute`, while low-level tensor
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

## Causal Program windows

GPT-2 and Llama expose `forward(length, outputStart = 0)`. `outputStart` narrows
the returned logits and hidden states, which is useful for decode-shaped output,
but the Program still declares and computes the complete prefix. The package
does not claim KV caching. Applications own token validation, greedy or sampled
generation, session reuse, and disposal explicitly. A future real cache must be
represented in Program inputs/state rather than hidden behind a model façade.

## ONNX execution backend

ONNX is a graph representation, not another architecture. The separate
`@affon/onnx` package owns conversion and execution; HF owns configuration,
labels, processors and task dispatch. Native adapters remain the default.

```ts
const model = load_model('/models/vit-hf-config', {
  task: 'image-classification',
  backend: 'onnx',
  graph_dir: '/models/vit-converted',
})
const processor = load_processor('/models/vit-hf-config', {
  task: 'image-classification', device: 'metal',
})
const session = new Session({ device: 'metal' })
const state = session.initialize(model.forward, { parameters: model.parameters })
const pixels = processor.process(rgb)
const logits = session.compile(model.forward).run({ [model.input_name]: pixels }, state)
```

The HF directory supplies `config.json`; the prepared graph directory supplies
`graph.json` and `weights.safetensors`. No native architecture weights or adapter
are loaded for this backend. The classifier preserves HF labels/configuration
and exposes its imported Program and named bindings; hidden states and generation are not promised.
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
  task: 'audio-classification', backend: 'onnx', graph_dir: '/models/ast',
})
const processor = load_processor('/models/ast/source', {task: 'audio-classification', device: 'metal'})
const audio = decode_wav(wavBytes)
const features = processor.process(audio.samples, audio.sampling_rate)
const session = new Session({ device: 'metal' })
const state = session.initialize(model.forward, { parameters: model.parameters })
const logits = session.compile(model.forward).run({ [model.input_name]: features }, state)
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
  graph_dir: '/models/whisper',
})
const processor = load_processor('/models/whisper/source', {
  task: 'automatic-speech-recognition', device: 'metal',
})
const audio = decode_wav(wavBytes)
const features = processor.process(audio.samples, audio.sampling_rate)
const session = new Session({ device: 'metal' })
const state = session.initialize(model.encoder.forward, { parameters: model.encoder.parameters })
const encoded = session.compile(model.encoder.forward).run({ features }, state)
```

This bounded integration supports the prepared `openai/whisper-tiny.en` bundle.
It uses a centered 400-point STFT, Slaney Mel filters and 30-second zero padding.
Frontend DSP runs in Affon TypeScript. The returned definition exposes separate
encoder, cross-attention, and decoder Programs plus embeddings, positions,
suppression metadata, and text decoding. The application creates Sessions,
owns request-local caches, and chooses token generation policy. The importer remains
independent of HF architecture names. The local-only ONNX task API is required; `from_pretrained`
still loads native architecture adapters only. See the app README for preparation
and the short-WAV/254-token limits. No timestamps or silence detection are promised.

Whisper feature extraction uses compute's native CPU `stft_power` and `filterbank`
primitives. HF owns the periodic Hann window, Slaney filters, 30-second padding,
frame selection and log normalization; the final features move to the requested
device. This requires an Affon runtime containing the spectral primitives and
removes the previous TypeScript FFT implementation. No Python or external audio
process is used at inference time.

### SmolLM2 instruction generation

`HuggingFaceTB/SmolLM2-135M-Instruct` is pinned to
`12fd25f77366fa6b3b4b768ec3050bf629380bac`. Use `from_pretrained` with
`task: 'text-generation'` and that revision, then:

```ts
if (!('encode_chat' in processor)) throw Error('Expected a chat processor')
const ids = processor.encode_chat([{role: 'user', content: 'What is the capital of France?'}])
const source = model.forward(ids.length, ids.length - 1)
// Execute the Program with a caller-owned Session/state, select a token, append,
// and author the next full-prefix Program until the application budget or EOS.
```

The native Llama adapter supports this bounded tied-head, bias-free variant with
RMSNorm, SiLU gating, grouped-query attention and full non-interleaved RoPE.
Scaled RoPE, untied heads, attention/MLP biases, and sliding windows are rejected.
The model configuration permits 8192 tokens; the playground deliberately limits
formatted prompts to 256 tokens and outputs to 64. Large-context performance is
not validated. No KV cache is claimed; current application generation executes
the full prefix with a one-token output window. No quantized execution is provided.

The original ~269 MB BF16 checkpoint downloads through the existing Hub path.
`checkpoint.load` widens BF16 values exactly to f32 on CPU before device placement;
weights therefore occupy ~538 MB in f32, excluding temporary buffers and caches.
A runtime containing BF16 checkpoint loading is required. Python is unnecessary
for downloading or inference.

The processor implements only the pinned SmolLM2 chat template, including its
default system message, and rejects other templates. It does not execute Jinja.
The tokenizer supports the model's Digits → ByteLevel pre-tokenization.
`load_llama` also accepts prepared local f32 artifacts independently of chat processing.
See the app audit README for reproducible reference checks and numerical limits.

The playground catalog also pins SmolLM2-360M-Instruct and 1.7B-Instruct; both
use this same adapter and chat processor. Their weights require roughly 1.45 GB
and 6.85 GB respectively after widening to f32, before runtime buffers. See
`apps/hf-inference/src/inference/models.ts` for exact revisions and the audit
README for backend-specific validation results.
