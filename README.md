<p align="center">
  <img src="assets/affon-lockup-vertical.svg" alt="affon" width="220"/>
</p>

<p align="center">
  <code>aff·on</code>&nbsp;&nbsp;<span style="color: #c5bc10; font-size: 15px; font-family: monospace;">/ əfˈɒn /</span>
</p>

<p align="center">
  A TypeScript runtime for scientific computing and machine learning
</p>

---

Affon is a TypeScript runtime for scientific computing and machine learning.

It provides declarative compute Programs, explicit differentiation and
optimization transforms, Session-bound execution, datasets, checkpointing,
first-party ML packages, notebooks, and runnable reference workloads in one
runtime-oriented package.

Affon is experimental and deliberately transparent. Its working direction is to
make computational models explainable as semantic programs: not only numeric
operators, but the roles of values and dimensions, mutable state,
differentiation, transformations, and the path to backend execution. TypeScript
is the current host interface rather than the project's central distinction.
CPU and Metal are the primary validation targets; CUDA has less complete
testing. Full PyTorch API compatibility and broad model-count coverage are not
goals. See the [roadmap](docs/roadmap.md) for the research direction and its
evidence gates.

## Quick Start

Install the latest release:

```bash
curl -fsSL https://affon.ai/install.sh | bash
```

This installs `affon` into `~/.affon/bin` and adds that directory to your shell
`PATH`.

Run a script:

```bash
affon hello.ts
```

```typescript
import { Session, Tensor, program } from "affon:compute";
import { linear } from "affon:nn";

const output = linear({ out_features: 1 });
const model = program("regression", p => {
  const x = p.argument("x", Tensor.f32([3, 1]));
  return output({ x }, "output");
});

const session = new Session({ device: "cpu" });
const state = session.initialize(model, { seed: 7 });
const executable = session.compile(model);
const x = session.tensor([[1], [2], [3]]);
const prediction = executable.run({ x }, state);

console.log(prediction.to_array());
```

For immediate computation, evaluated tensors can use the lazy default Session:

```typescript
import { Tensor } from "affon:compute";
import { add } from "affon:ops";

const result = add(Tensor.from([1, 2, 3]), Tensor.ones([3]));
console.log(result.to_array());
```

The default device is selected at runtime startup from `AFFON_DEVICE` (or
Affon's normal automatic detection when it is unset). It cannot be replaced or
changed through the canonical API.

## Public Modules

- `affon:compute` is the declarative Program, Session, differentiation,
  built-in loss-template, and Program-transform surface.
- `affon:nn` contains callable factories for parameterized neural-network layers.
- `affon:ops` is the shared operation vocabulary for formal and evaluated tensors.
- `affon:optim` contains immutable optimizer descriptors.
- `affon:dataset` is the ingest, preprocessing, batching, text-record, and tokenizer surface.
- `affon:checkpoint` is the training-state persistence and restore surface.

Runtime/system modules such as filesystem, process, and telemetry are provided
by the underlying runtime under `std:*` specifiers.

## Features

- **Declarative compute Programs** with typed tensor roles, explicit state, differentiation, optimization, and compilation
- **CPU, Metal, and CUDA device placement** with kernel-capability-aware lowering
- **Callable neural-network factories** for linear, embedding, and layer normalization
- **Built-in loss templates** under `losses`, combined with reusable models by `optimize(model, loss, optimizer)`
- **Immutable optimizer descriptors** for SGD, Adam, and AdamW Program transforms
- **Dataset pipelines** for tabular and text workflows, including token windows and tokenizer adapters
- **Checkpoint persistence** for named Program state and application-owned bundles
- **Runtime diagnostics** with `std:telemetry.metrics()`, traces, and memory signals
- **First-party packages** for models, Hugging Face integration, tokenizers, and ONNX execution
- **Runnable apps** including the decoder language-model reference workload

## Stability

Affon focuses on a documented scientific-computing and ML surface rather than
general Node compatibility.

- module loading supports a focused ESM-oriented subset rather than general Node compatibility
- Program compilation lowers authored structure to a Session-bound executable
- unsupported canonical Program lowering surfaces an explicit error; it never silently changes execution models
- Metal acceleration is partial and operation-dependent
- CUDA supports the documented Program path for selected `f32` computation, indexing, losses, and optimizer updates, with cuBLAS matmul; verify the exact Program and shape on the target device
- package APIs evolve through the first-party package and app workflow

Use the linked docs as the source of truth for exact supported behavior and edge
cases.

## Platform Notes

- macOS uses Accelerate and includes Metal-backed paths where supported
- Linux supports CPU execution and NVIDIA CUDA acceleration; CUDA requires a
  working driver plus CUDA 12 NVRTC and cuBLAS runtime libraries
- select the default backend with `AFFON_DEVICE=cpu`, `metal`, or `cuda`; select
  a CUDA ordinal with `AFFON_CUDA_DEVICE=N` before the first CUDA allocation
- release automation should verify the platform assets attached to a given release

## Docs

- Roadmap: [Legible computational Programs](docs/roadmap.md)
- Getting started: [Python to AFFON](docs/getting-started/python-to-affon.md)
- Core numerics: [Compute Concepts](docs/core/compute.md), [Backend Support](docs/core/backend-support.md), [Error Handling](docs/core/errors.md)
- Machine learning: [NN Concepts](docs/ml/nn/index.md), [Metrics Concepts](docs/ml/metrics.md), [Optim Concepts](docs/ml/optim.md), [Checkpoints](docs/ml/checkpoints.md), [Text Datasets](docs/ml/text-datasets.md), [ML Glossary](docs/ml/glossary.md)
- Runtime: [Install](docs/runtime/install.md), [Configuration](docs/runtime/configuration.md), [Module Loader](docs/runtime/module-loader.md), [Memory Debugging](docs/runtime/memory-debugging.md)
- Apps: [apps/](apps/), [Decoder LM](apps/decoder-lm/README.md)
- Packages: [packages/](packages/), [Models](packages/@affon/models/README.md), [Hugging Face](packages/@affon/huggingface/README.md), [Tokenizers](packages/@affon/tokenizers/README.md)
- Editor: [VS Code TypeScript Notebook](https://github.com/neucine/vscode-typescript-notebook)

## License

MPL-2.0

Bundled datasets and tokenizer assets retain their respective third-party
terms. See [Third-Party Notices](THIRD_PARTY_NOTICES.md).
