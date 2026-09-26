# Hugging Face inference application

```text
src/
  inference/   # Model loading, text generation, image classification
  serve/       # HTTP routes, configuration, native server entry point
  benchmark/   # Native inference measurement entry point
public/        # Browser HTML, CSS, JavaScript
tests/         # Inference/HTTP tests and reference fixtures
audit/         # Parity tooling, Python oracle scripts, reports
benchmarks/    # Benchmark runner, methodology, reports
onnx/          # Bounded static graph-import experiment, tests and reports
```

The dependency direction is `serve → inference → @affon/huggingface`.
Inference has no dependency on serving or browser code.

Run from the repository root:

```sh
AFFON_DEVICE=metal AFFON_HF_CACHE=/tmp/affon-hub-cache \
  ./zig-out/bin/affon apps/hf-inference/src/serve/server.ts
```

- [Serving and browser UI](src/serve/README.md)
- [Core inference](src/inference/README.md)
- [Audits and reference generation](audit/README.md)
- [Performance benchmarks](benchmarks/README.md)
- [ONNX graph-import experiment](onnx/README.md)

```sh
./zig-out/bin/affon test apps/hf-inference/tests
./zig-out/bin/affon apps/hf-inference/tests/serve/image-url-check.ts
```

### Image model choices

The playground always loads **ViT native**. Enable **ViT ONNX** and
**MobileNet ONNX** with optional local artifact paths:

```sh
AFFON_DEVICE=metal \
AFFON_VIT_ONNX_DIR=/models/vit-graph \
AFFON_MOBILENET_DIR=/models/mobilenet-hf \
AFFON_MOBILENET_ONNX_DIR=/models/mobilenet-graph \
affon apps/hf-inference/src/serve/server.ts
```

Keep the existing HF cache/offline settings as needed. Graph directories contain
`graph.json` and `weights.safetensors` produced by the [ONNX conversion flow](onnx/README.md).
The MobileNet HF directory contains the matching `config.json` and
`preprocessor_config.json`. Both MobileNet paths must be supplied together.
Unconfigured models are omitted from the selector; invalid configured artifacts
fail startup. All configured models are loaded once and reused.

Both uploads and downloaded image URLs use the selected model. Results show its
HF ID, native-adapter/ONNX-graph execution path, compute device, model inference
time (including output readback), and total RGB preprocessing/classification time.
Download and browser image decoding are outside that total. The HTTP API accepts
`model=vit-native|vit-onnx|mobilenet-onnx`; omitted model defaults to ViT native,
and unknown/unavailable models return 400.

MobileNet uses native shortest-edge resize to 256 followed by a 224 center crop;
ViT resizes directly to 224 square. MobileNet's 1001 output labels include the
checkpoint's background class. No Python is needed while serving or preprocessing.

### Audio classification

The **Audio classification** tab runs AST's 35 English speech-command labels.
Upload a short WAV, preview it locally, then choose **Classify audio**. The server
decodes, downmixes, resamples and computes spectrogram features in Affon. It
accepts mono/stereo PCM8/16/24/32 or float32 WAV, 8–96 kHz, 25 ms–30 s, at most
16 MiB. It classifies the first 1.295 seconds and reports truncation. Other codecs
and extensible WAV are not supported yet. Say a single word near the start;
there is no speech/silence rejection or transcription.

Prepare the pinned model once (Python is only preparation/reference tooling):

```sh
python apps/hf-inference/onnx/export-audio-model.py \
  --revision 315b0b847a3ca207e68b718503ad72066612eacd \
  --output /tmp/affon-onnx-ast \
  --wav apps/hf-inference/tests/fixtures/command-yes.wav
python packages/@affon/onnx/tools/convert.py \
  /tmp/affon-onnx-ast/model.onnx /tmp/affon-onnx-ast
AFFON_DEVICE=metal AFFON_AST_ONNX_DIR=/tmp/affon-onnx-ast \
  affon apps/hf-inference/onnx/check-audio.ts
```

Add `AFFON_AST_DIR=/tmp/affon-onnx-ast/source` and
`AFFON_AST_ONNX_DIR=/tmp/affon-onnx-ast` to the existing server launch environment.
The model is optional, and both paths must be supplied together. Without them,
the audio tab explains that it is unavailable. `POST /api/classify-audio` accepts
raw `audio/wav` bytes and returns top predictions, source audio metadata,
truncation status, device, and inference/total timing.

The [spoken “yes” fixture](tests/fixtures/command-yes.wav) was synthesized locally
with macOS Samantha; it contains no user recording. See the
[audio audit](onnx/reports/audio/summary.md) for reference results and remaining gaps.

### Speech to text (Whisper)

The **Speech to text** tab runs `openai/whisper-tiny.en` using a native frontend
and cached ONNX decoding. Upload an English WAV up to 30 seconds and 16 MiB.
The model has a 254-token generation budget; the UI marks transcripts that reach
that limit. No timestamps, long-form segmentation or silence detection yet.

Prepare the pinned graphs and independent reference once:

```sh
python apps/hf-inference/onnx/export-whisper.py \
  --revision 87c7102498dcde7456f24cfd30239ca606ed9063 \
  --output /tmp/affon-onnx-whisper \
  --wav apps/hf-inference/tests/fixtures/whisper-sentence.wav
python apps/hf-inference/onnx/export-whisper-cache.py /tmp/affon-onnx-whisper
for part in encoder decoder cross step; do
  python packages/@affon/onnx/tools/convert.py \
    /tmp/affon-onnx-whisper/$part/model.onnx /tmp/affon-onnx-whisper/$part
done
AFFON_DEVICE=metal AFFON_WHISPER_DIR=/tmp/affon-onnx-whisper \
  affon apps/hf-inference/onnx/check-whisper.ts
```

Add `AFFON_WHISPER_DIR=/tmp/affon-onnx-whisper` to the existing server environment.
This bundle includes source configuration/tokenizer, embeddings/positions,
generation settings and converted encoder/cross/step directories. The full-prefix
`decoder` and reference tensors are audit artifacts, not required for serving.
The optional model loads once; each transcription owns its decoder caches.
`POST /api/transcribe` takes `audio/wav` bytes and returns text, token IDs,
encoder/inference/total timings and truncation status. The UI is kept in
`public/transcription.js`, separate from inference and serving code.

See the [Whisper audit and next decision](onnx/reports/whisper/summary.md).
