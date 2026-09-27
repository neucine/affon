# @affon/onnx

Experimental graph-format infrastructure, independent of Hugging Face and model
architecture. Loads an offline-converted static-f32 ONNX graph and executes it
using native Affon tensor operations. No Python or ONNX Runtime is used during
execution. This is not a native `.onnx` protobuf reader.

```ts
import { load_graph, capabilities } from '@affon/onnx'

const model = load_graph('/models/converted', 'metal') // CPU by default
const outputs = model.forward({ pixels })
console.log(outputs.logits)
console.log(capabilities.opset, capabilities.operators)
```

`load_graph(directory, device?)` reads `graph.json` and `weights.safetensors`.
The returned `graph` describes named inputs, outputs and static shapes;
`forward(namedInputs)` returns named tensors. It checks input/constant/output
shapes, graph dependencies and operation names. Unsupported operations fail
explicitly. This internal artifact format is versioned as `affon-onnx-static/v1`;
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

The runtime and spatial lowerings are package-owned. HF model configuration,
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

Metal dense last-axis layer normalization computes statistics once per row in
the compute core, preserving the existing reduction order. Other axes keep the
general kernel. On Metal, the executor automatically compiles the normalization
scale/bias expression, allowing compute to fuse eligible dense suffix-broadcast
f32 multiply/add operations with separate-operation rounding. CPU and unsupported
layouts retain ordinary operations. No affine-fusion configuration is required;
the former `AFFON_ONNX_AFFINE_FUSION` switch has been removed. Fresh Whisper
comparisons show a repeatable roughly 5% full-request improvement with the current
runtime; see the [command audit](../../../apps/hf-inference/benchmarks/reports/whisper-command-audit/README.md).


Metal graph calls automatically batch compatible kernels within bounded execution
scopes. Each `forward` call completes GPU work before returning its outputs.
Profiling callbacks use synchronous per-operation execution so their timing and
callback behavior stay consistent. This requires no caller option; CPU execution
and the `load_graph` return shape are unchanged.
