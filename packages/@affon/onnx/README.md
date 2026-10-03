# @affon/onnx

Experimental graph-format infrastructure, independent of Hugging Face and model
architecture. Loads an offline-converted static-f32 ONNX graph as a canonical
Affon `Program`. No Python or ONNX Runtime is used during import or execution.
This is not a native `.onnx` protobuf reader.

```ts
import { load_graph, capabilities } from '@affon/onnx'
import { Session } from 'affon:compute'

const model = load_graph('/models/converted')
const session = new Session({ device: 'metal' })
const state = session.initialize(model.forward, { parameters: model.parameters })
const outputs = session.compile(model.forward).run({ pixels }, state)
console.log(capabilities.opset, capabilities.operators)
```

`load_graph(directory)` reads `graph.json` and `weights.safetensors`.
The returned `graph` describes named inputs, outputs and static shapes;
`forward` is a device-neutral Program and `parameters` is its initializer map.
`output_names` preserves the manifest's output binding order. Import checks
input/constant/output shapes, graph dependencies and operation names. Unsupported
operations fail explicitly. This internal artifact format is versioned as `affon-onnx-static/v1`;
the package remains experimental and expects trusted converter-produced bundles.

## Prepare artifacts

```sh
python packages/@affon/onnx/tools/convert.py model.onnx /models/converted
```

The preparation environment needs `onnx`, `numpy` and `safetensors`. Independent
fixture generation also needs `onnxruntime`. Tested versions are recorded in the
[app audit environment](../../../apps/hf-inference/onnx/requirements.txt).
Exporting a framework model is a separate preparation step. The converter folds
constants and static shape expressions, checks the supported subset, and writes
a manifest containing the source ONNX SHA-256. Model files remain local; package
loading does not download, export or execute Python.

## Boundary

Only default-domain opset 17, static positive dimensions, and runtime f32 tensors.
See exported `capabilities` for operation-specific restrictions. The subset
includes explicit-pad grouped/depthwise NCHW convolution, constant-zero spatial
padding, pooling, normalization, matrix operations and shape transforms.
Erf uses the A&S 7.1.26 approximation through native compute on CPU/Metal.
CUDA and older runtimes retain the equivalent tensor-expression fallback. Dynamic shapes, control flow, custom domains,
implicit auto-padding, general Pad modes, and lower-precision execution are not
supported. CUDA is not validated by these experiments.

The Program importer and spatial lowerings are package-owned. HF model configuration,
labels, processors, Hub downloads and task APIs belong to `@affon/huggingface`.
Model export scripts, independent framework oracles, benchmarks and reports remain
in the [application experiment](../../../apps/hf-inference/onnx/README.md).

## Verification

```sh
python packages/@affon/onnx/test/convert_test.py
AFFON_DEVICE=cpu affon test packages/@affon/onnx/test
AFFON_DEVICE=metal affon test packages/@affon/onnx/test
```

Fixtures are small independent ONNX graphs. ViT and MobileNetV2 are validated
on CPU and Metal; this establishes those configurations, not broad ONNX support.

Static `Unsqueeze` with constant one-dimensional i64 axes is lowered to a
validated `Reshape` during conversion. No AST-specific graph implementation is
required. AST speech-command classification now also exercises the executor;
its waveform preprocessing belongs to the HF package.

Whisper additionally exercises NCW 1D convolution (lifted to the existing NCHW
spatial implementation) and constant-bound `Slice` with unit positive steps.
Negative indices/axes are normalized during conversion; empty slices, nonunit
steps and runtime control tensors are rejected. Cached decoding uses ordinary
multiple graph inputs and outputs, without HF-specific graph operations.
