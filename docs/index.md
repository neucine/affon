# Affon Docs

This folder contains the public user documentation for Affon.

Current public module taxonomy:
- `affon:compute` for tensors, autograd, and training control
- `affon:nn` for model-building on top of compute
- `affon:dataset` for ingest, preprocessing, and batching
- `affon:checkpoint` for training-state persistence and restoration
- `affon:embed` for embedding-model inference
- `affon:plot` for plotting
- `affon:ffi` for raw C ABI calls and the higher-level `c.decl(...)` C binding layer
- `affon:process`, `affon:fs`, and `affon:http` for focused runtime/system interop
- `affon:test` and `affon:util` for built-in test and utility helpers

## Getting Started

- [Python to AFFON](./getting-started/python-to-affon.md)

## Core Numerics

- [Compute Concepts](./core/compute.md) - compute tensors, parameters, modules, gradients, compilation, and graph inspection
- [Compute Kernel Matrix](./core/kernel-matrix.md) - current backend/device support by operation family
- [Error Handling](./core/errors.md)

## Machine Learning

- [Illustrated Learning Coverage](./learn/illustrated/index.md)
- [ML Glossary](./ml/glossary.md)
- [Metrics Concepts](./ml/metrics.md)
- [Optim Concepts](./ml/optim.md)
- [NN Concepts](./ml/nn/index.md)
- [Checkpoints](./ml/checkpoints.md)
- [Text Datasets](./ml/text-datasets.md)

## Runtime & Interop

- [Install](./runtime/install.md)
- [FFI](./runtime/ffi.md) - raw `affon:ffi` plus its higher-level `c` C interop export
- [Module Loader](./runtime/module-loader.md) - supported ESM/package subset and current compatibility boundary
- [Memory Debugging](./runtime/memory-debugging.md) - metrics, trace collection, observer endpoints, and cache trimming
