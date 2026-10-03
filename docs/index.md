# Affon Docs

This folder contains the public user documentation for Affon. The docs are
organized around the runtime surfaces users import from code, then the packages
and apps that build on those surfaces.

## Start Here

- New to Affon from Python or PyTorch: read [Python to AFFON](./getting-started/python-to-affon.md).
- Installing the runtime: read [Install](./runtime/install.md).
- Building tensor or autograd code: read [Compute Concepts](./core/compute.md).
- Building models: read [NN Concepts](./ml/nn/index.md).
- Loading text or tabular data: read [Text Datasets](./ml/text-datasets.md).
- Saving training state: read [Checkpoints](./ml/checkpoints.md).

## Public Modules

Affon keeps the public runtime surface intentionally small:

- `affon:compute` provides tensors, Programs, Sessions, differentiation,
  built-in loss templates, and Program transforms.
- `affon:ops` provides operations shared by formal and evaluated tensors.
- `p.nn` provides builder-bound parameterized neural-network declarations.
- `affon:optim` provides optimizer descriptors.
- `affon:dataset` provides ingest, preprocessing, batching, text records, and tensor export.
- `affon:checkpoint` provides training-state persistence and restoration.
- `std:*` runtime modules provide filesystem, process, telemetry, and other runtime services.

First-party packages under `packages/` provide reusable higher-level building
blocks, such as tokenizers, transformer blocks, and language-model helpers.
The [Hugging Face package](../packages/@affon/huggingface/README.md) owns
native HF model loading and processor integration across domains.
Complete runnable workflows live under `apps/`.

## Getting Started

- [Python to AFFON](./getting-started/python-to-affon.md)

## Core Numerics

- [Compute Programs](./core/compute.md) - Programs, formal and evaluated tensors, Sessions, differentiation, optimization, and inspection
- [Compute Kernel Matrix](./core/kernel-matrix.md) - current backend/device support by operation family
- [Error Handling](./core/errors.md)

## Machine Learning

- [ML Overview](./ml/index.md)
- [ML Glossary](./ml/glossary.md)
- [Metrics Concepts](./ml/metrics.md)
- [Optim Concepts](./ml/optim.md)
- [NN Concepts](./ml/nn/index.md)
- [Checkpoints](./ml/checkpoints.md)
- [Text Datasets](./ml/text-datasets.md)

## Runtime & Interop

- [Install](./runtime/install.md)
- [Configuration](./runtime/configuration.md) - environment variables and startup/runtime mutability
- [Module Loader](./runtime/module-loader.md) - supported ESM/package subset and current compatibility boundary
- [Memory Debugging](./runtime/memory-debugging.md) - metrics, trace collection, and cache trimming
- [FFI](./runtime/ffi.md) - native interop boundary and ownership notes

## Documentation Conventions

The public docs focus on stable imports, commands, examples, and documented
behavior. When implementation details are mentioned, they are labeled as
implementation boundaries rather than alternative public APIs.
