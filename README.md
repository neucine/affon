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

It provides compute tensors, autograd, neural-network layers, datasets,
checkpointing, first-party ML packages, notebooks, and runnable reference
workloads in one runtime-oriented package.

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
import { tensor, clear_grad, grad, sgd, relu } from "affon:compute";
import nn from "affon:nn";

const model = nn.Sequential(nn.Linear(1, 4), relu, nn.Linear(4, 1));
const params = model.parameters;
const step = sgd({ lr: 0.01 });
const criterion = nn.MSELoss();

const x = tensor([[1], [2], [3]]);
const target = tensor([[2], [4], [6]]);

clear_grad(params);
const pred = model(x);
const loss = criterion(pred, target);
grad(loss, params);
step(params);

console.log(loss.item());
```

## Public Modules

- `affon:compute` is the tensor, autograd, optimizer, schedule, module, and graph-compilation substrate.
- `affon:nn` is the model-building layer on top of `affon:compute`.
- `affon:dataset` is the ingest, preprocessing, batching, text-record, and tokenizer surface.
- `affon:checkpoint` is the training-state persistence and restore surface.

Runtime/system modules such as filesystem, process, and telemetry are provided
by the underlying runtime under `std:*` specifiers.

## Examples

- [Getting Started](examples/getting-started.ipynb)
- [NN Basics: Feed-forward + Recurrent](examples/nn-basics.ipynb)
- [RNNs](examples/rnn.ipynb)
- [Embeddings](examples/embedding.ipynb)
- [Illustrated Tensors](examples/illustrated/tensor-illustrated.ipynb)
- [Tensor Ops Illustrated](examples/illustrated/tensor-ops-illustrated.ipynb)
- [Illustrated NN](examples/illustrated/nn-illustrated.ipynb)
- [Illustrated Losses And Metrics](examples/illustrated/losses-metrics-illustrated.ipynb)
- [Illustrated Training And Optimizers](examples/illustrated/training-illustrated.ipynb)

## Features

- **Compute-first numerics** with typed tensors, parameters, modules, gradients, and graph compilation
- **CPU and Metal execution** with explicit device placement and kernel-capability-aware lowering
- **Neural-network layers** including linear, recurrent, normalization, embedding, dropout, and batchnorm modules
- **Optimizers and schedules** including SGD, Adam, AdamW, gradient clipping, and scheduled training loops
- **Dataset pipelines** for tabular and text workflows, including token windows and tokenizer adapters
- **Checkpoint persistence** for model and optimizer state
- **Runtime diagnostics** with `std:telemetry.metrics()`, traces, and memory signals
- **First-party packages** for transformers, language-model workflows, tokenizers, CNN, and vision work
- **Runnable apps** including the decoder language-model reference workload

## Stability

Affon focuses on a documented scientific-computing and ML surface rather than
general Node compatibility.

- module loading supports a focused ESM-oriented subset rather than general Node compatibility
- graph compilation prefers native graph-backed execution when capture and lowering succeed
- incompatible captures or unsupported lowering may fall back to eager execution or surface explicit errors
- Metal acceleration is partial and operation-dependent
- package APIs evolve through the first-party package and app workflow

Use the linked docs as the source of truth for exact supported behavior and edge
cases.

## Platform Notes

- macOS uses Accelerate and includes Metal-backed paths where supported
- Linux uses the configured CPU linear algebra path
- release automation should verify the platform assets attached to a given release

## Docs

- Getting started: [Python to AFFON](docs/getting-started/python-to-affon.md)
- Core numerics: [Compute Concepts](docs/core/compute.md), [Compute Kernel Matrix](docs/core/kernel-matrix.md), [Error Handling](docs/core/errors.md)
- Machine learning: [NN Concepts](docs/ml/nn/index.md), [Metrics Concepts](docs/ml/metrics.md), [Optim Concepts](docs/ml/optim.md), [Checkpoints](docs/ml/checkpoints.md), [Text Datasets](docs/ml/text-datasets.md), [ML Glossary](docs/ml/glossary.md)
- Runtime: [Install](docs/runtime/install.md), [Releasing](docs/runtime/releasing.md), [Configuration](docs/runtime/configuration.md), [Module Loader](docs/runtime/module-loader.md), [Memory Debugging](docs/runtime/memory-debugging.md)
- Apps: [apps/](apps/), [Decoder LM](apps/decoder-lm/README.md)
- Packages: [packages/](packages/), [Transformers](packages/transformers/README.md), [LM](packages/lm/README.md), [Tokenizers](packages/tokenizers/README.md)
- Editor: [Affon for VS Code](https://github.com/neucine/affon-vscode)

## License

MPL-2.0
