# Model package experiment

A local experiment for the idea: an immutable package describes a model's
artifacts and execution contract; a separate runtime loads and executes it.
This lives in an app until the shared contract has evidence for a package API.

Two existing synthetic fixtures exercise different representations and calls:

| Model | Definition | Weights | Call |
| --- | --- | --- | --- |
| Convolutional tensor graph | Prepared ONNX `graph.json` | Safetensors | Named tensors → named tensors |
| Tiny Llama | HF `config.json`, architecture implemented by the adapter | Safetensors | Token IDs and generation budget → token IDs |

Both execute in Affon through different adapters. This does **not** demonstrate
portability across engines, arbitrary ONNX support, distributed execution,
automatic placement, or scaling. The graph is Affon's existing prepared ONNX
format, not a raw ONNX protobuf. The Llama config is not a complete compute graph.
No model downloads, Python ML libraries, or new tensor format are required.
Python handles local packaging and verification; inference executes in Affon.

## Try it

Run from the repository root, with Python 3.11+ and a current Affon on PATH:

```sh
# Build the current checkout, then expose its executable and local packages:
zig build -Doptimize=ReleaseFast
export PATH="$PWD/zig-out/bin:$PATH"
export RUNTIME_PACKAGE_PATH="$PWD/packages"

python3 apps/model-package/package.py pack \
  packages/@affon/onnx/test/fixtures /tmp/conv-package \
  --runtime affon/prepared-onnx/v1 --files graph.json weights.safetensors

python3 apps/model-package/package.py inspect /tmp/conv-package

python3 apps/model-package/package.py pack \
  packages/@affon/huggingface/test/fixtures/llama /tmp/llama-package \
  --runtime affon/llama-tokens/v1 --files config.json model.safetensors

printf '%s\n' '{"input_ids":[1,7,12,3,29],"max_new_tokens":4}' > /tmp/llama-input.json
python3 apps/model-package/package.py run /tmp/llama-package --input /tmp/llama-input.json
```

Expected token IDs: `[1,7,12,3,29,29,29,29,29]`. These are synthetic fixture
weights; this is an execution check, not a useful language model. Existing
package destinations are rejected. Use another destination for repeated runs.

Graph requests have `{"inputs":{"x": <nested numeric array>}}`; the graph
fixture's `reference.json` contains the required input under `input`. Responses
include each output's shape and data. Token generation uses the adapter contract
shown above; a tokenizer is intentionally unnecessary for this experiment.
`--device metal` selects Metal without changing the package. CPU is the default.

## The proposed v0 envelope

Illustrative digest abbreviated below; the pack command writes full hashes:

```json
{
  "schema": "affon-model-package/v0",
  "runtime": "affon/prepared-onnx/v1",
  "artifacts": [
    {"path": "graph.json", "size": 123, "digest": "sha256:..."},
    {"path": "weights.safetensors", "size": 456, "digest": "sha256:..."}
  ]
}
```

These are the only shared fields needed by the two examples:

- `schema` versions the envelope.
- `runtime` names a versioned adapter contract. The caller installs the adapter;
  the package cannot name an arbitrary command to execute.
- `artifacts` lists flat local filenames, byte sizes, and content digests. The
  original file formats and filenames survive packaging.

The packer can transport an unknown adapter's artifacts. Execution requires an
installed adapter that understands the package. Each adapter defines required
files, supported model variants, request/response semantics, and loading rules.
The core does not infer a model family from weight names or invent a common
generation/tensor API. There is no extensions bag or capabilities vocabulary yet.

Custom support currently means registering a contract in `package.py` and adding
a handler in `src/run.ts`. Extra configuration can be an ordinary hashed artifact
interpreted by that adapter. This is a small local registry, not a plugin ABI.
Unknown schema versions, duplicate or invalid paths, and incomplete adapter
layouts fail explicitly.

`runtime` pins a contract version, not the Affon binary or kernel build. Exact
runtime provenance, numeric tolerances, and cross-engine compatibility remain
open design work. A future runtime image reference must preserve this distinction.

## Package lifecycle and limits

Packing stages a complete directory before publication and refuses existing
destinations. Inspection verifies every listed size and digest. Running copies
only declared files into a private temporary directory and verifies those bytes
before invoking the adapter. Unlisted files cannot influence the loader. Packages
are immutable by convention; hashes detect changes against the current manifest,
not publisher identity. This prototype assumes trusted local artifacts.

The private copy makes ownership simple but adds disk space and startup cost.
It is not the eventual loading strategy for large weights. Distribution, caching,
weight shard acquisition, OCI transport, runtime images, and immutable references
to the manifest itself are separate next experiments.

Hardware selection is invocation state. Replicas and service objectives will
belong to a deployment request once a scheduler exists. Artifact sizes do not
prove resident memory usage or reveal a valid sharding strategy.

## Verification and what it establishes

```sh
python3 -m unittest discover -s apps/model-package/test -p '*_test.py' -v
```

Tests package and execute both models on CPU against existing independent ONNX
Runtime and PyTorch references. They also check relocation, reproducible
manifests, same-size corruption, missing files, unsupported adapters, invalid
paths, adapter file contracts, and failed publication.

Validated on 2026-09-29: all eight tests passed on macOS arm64 / CPU with a fresh
ReleaseFast build of this checkout. The installed `affon` and preexisting
`zig-out/bin/affon` were too old for current model APIs. The verification build
was installed at `/tmp/affon-model-package-build/bin/affon`; no global installation
was changed. Metal execution is exposed but was not validated in this experiment.

The experiment tests shared packaging and explicit adapter boundaries. Its main
remaining question is how much compatibility metadata a second engine needs.
Try the same underlying model with another engine before promoting any runtime
requirements or scheduling vocabulary into the shared envelope.
