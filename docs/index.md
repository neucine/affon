# Affon Docs

This folder contains the public user documentation for Affon.

Current public module taxonomy:
- `affon:compute` for tensors, autograd, and training control
- `affon:nn` for model-building on top of compute
- `affon:dataset` for ingest, preprocessing, and batching
- `affon:checkpoint` for training-state persistence and restoration
- `affon:test` for built-in test helpers
- `std:*` runtime modules for filesystem, process, telemetry, and other runtime services

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
- [Configuration](./runtime/configuration.md) - environment variables, startup/runtime mutability, and legacy aliases
- [Module Loader](./runtime/module-loader.md) - supported ESM/package subset and current compatibility boundary
- [Memory Debugging](./runtime/memory-debugging.md) - metrics, trace collection, and cache trimming
