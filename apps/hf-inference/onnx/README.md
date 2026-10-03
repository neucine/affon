# Bounded ONNX graph-import experiment

This experiment asks whether an exported computation graph can replace a
handwritten HF architecture adapter while still running on Affon's compute core.
It uses the same local f32 ViT checkpoint and preprocessed reference inputs as
the existing image-classification audit. It does not read HF configuration or
recognize ViT layer names when converting or executing a graph.

The deployment path is:

```text
Preparation: local HF model → ONNX → graph.json + weights.safetensors
Execution:   prepared pixels → generic graph executor → affon:compute → logits
```

Python, PyTorch, ONNX and ONNX Runtime are preparation/reference dependencies.
The deployed executor uses Affon only. This is an **offline ONNX conversion
prototype**, not a native `.onnx` protobuf loader or a general ONNX backend.
Image decoding, resizing and normalization remain processor responsibilities;
this experiment feeds saved processor outputs to isolate graph correctness.

## Reproduce

Use the audit environment from `../audit/README.md`, then install the additional
versions in `requirements.txt`. Commands run from the repository root:

```sh
python apps/hf-inference/onnx/export-vit.py \
  --model-dir /tmp/affon-hf-vit --output /tmp/affon-onnx-vit
python apps/hf-inference/onnx/convert.py \
  /tmp/affon-onnx-vit/model.onnx /tmp/affon-onnx-vit
python apps/hf-inference/onnx/run.py \
  --affon /tmp/affon-hf-decode/bin/affon \
  --model-dir /tmp/affon-hf-vit --graph-dir /tmp/affon-onnx-vit \
  --output /tmp/affon-onnx-results
```

The Affon binary needs the native `layer_norm` API. The existing prepared
checkpoint must include `reference.json` and `reference.safetensors`.
No model weights or large exported models are checked into this directory.

For native execution alone:

```ts
import { load_graph } from '@affon/onnx'
import { Session } from 'affon:compute'
const model = load_graph('/path/to/converted')
const session = new Session({ device: 'metal' })
const state = session.initialize(model.forward, { parameters: model.parameters })
const logits = session.compile(model.forward).run({ pixels }, state)
```

The runner starts separate graph/adapter processes on CPU and Metal sequentially.
Each performs one warmup plus three timed forwards for each of two reference
inputs. Timing includes logit readback but excludes preprocessing and file
loading. Loading and live tensor bytes are recorded separately; file caches are
not flushed. Background activity is uncontrolled. Three samples are not a tail
latency or long-term memory-stability measurement.

## Second architecture: MobileNetV2

```sh
python apps/hf-inference/onnx/export-image-model.py \
  --model google/mobilenet_v2_1.0_224 \
  --revision 75e607b00aeae1297cc89d026a118bce012f5c5a \
  --output /tmp/affon-onnx-mobilenet
python apps/hf-inference/onnx/convert.py \
  /tmp/affon-onnx-mobilenet/model.onnx /tmp/affon-onnx-mobilenet
python apps/hf-inference/onnx/run.py \
  --affon /tmp/affon-hf-decode/bin/affon --routes graph \
  --model-dir /tmp/affon-onnx-mobilenet/source \
  --graph-dir /tmp/affon-onnx-mobilenet --output /tmp/mobilenet-results \
  --build-profile ReleaseFast
```

This checks inference on saved HF-processor outputs; it does not add a native
MobileNet processor or change the playground's model selection. The exporter
checks ONNX Runtime against PyTorch before native execution. Its wrapper remains
in evaluation mode so exporting cannot change subsequent BatchNorm references.

## Supported subset

Only default-domain **opset 17**, static positive dimensions and runtime **f32**
tensors are accepted. No dynamic batch/sequence length, control flow, local
functions, custom domains, sparse or overridable initializers. Unsupported
operations and attributes are rejected during conversion.

- Add, Mul, Div, rank-two-or-higher MatMul, Gemm, Identity and f32 identity Cast.
- Transpose, Concat, constant-shape Reshape (`allowzero=0`), Softmax and constant
  scalar-i64 Gather. ONNX scalar Gather removes its indexed axis.
- LayerNormalization: final axis only, f32 accumulation, affine scale and bias,
  one output. Normalization is authored as canonical Program operations.
- Conv: NCHW/OIHW 2D, constant weights and optional constant bias, explicit
  nonnegative padding, positive strides/dilations and compatible groups.
  Nonoverlapping group-one windows keep the original patch/matmul path; other
  windows use prepared spatial gathers plus grouped matmul or depthwise
  multiplication/reduction. Implicit `auto_pad` is rejected.
- Pad: nonnegative NCHW spatial padding with constant zero fill only.
- Clip: finite constant scalar lower/upper bounds. GlobalAveragePool: NCHW.
  Flatten: static axis lowered to a reshape.
- Erf: explicit A&S 7.1.26 approximation, independently checked by the fixture
  and model logits. A native exact-erf operator remains a gap.
- Constant-only subgraphs are evaluated during preparation. Static Shape and
  subsequent shape expressions are folded; shape/Gather/Reshape controls are
  not silently cast into runtime f32 data.

The prepared manifest is an internal experiment format. Its executor validates
input/constant/output shapes, dataflow order and operation names. It is intended
for manifests produced by this converter, not arbitrary untrusted manifests.
Intermediate tensors are released after their final consumer; constants remain
owned by the loaded model.

Operator semantics are based on the [ONNX operator specifications](https://onnx.ai/onnx/operators/),
including [LayerNormalization](https://onnx.ai/onnx/operators/onnx__LayerNormalization.html).

## Tests and evidence

```sh
python packages/@affon/onnx/test/convert_test.py
AFFON_DEVICE=cpu affon test packages/@affon/onnx/test/runtime.test.ts
AFFON_DEVICE=metal affon test packages/@affon/onnx/test/runtime.test.ts
```

The preparation tests regenerate a small synthetic ONNX model and independent
ONNX Runtime expected values. It exercises convolution, layout changes,
normalization, Erf, projection, Softmax, negative scalar Gather and affine Gemm,
with multiple graph outputs. Repeated execution and malformed input/dataflow
checks run natively. Converter rejection tests cover unsupported opsets,
dynamic inputs, implicit padding, nonzero Pad fill, dynamic convolution bias
and unsupported runtime operations. Additional independent spatial fixtures test
batches, groups, depthwise kernels, overlap, dilation and asymmetric padding.
