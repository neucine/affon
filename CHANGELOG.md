# Changelog

All notable user-facing changes to Affon will be documented in this file.

The format is based on Keep a Changelog and the project follows Semantic Versioning.

## [Unreleased]

## [0.3.0] - 2026-09-05

### Added

- added a Linux CUDA backend with runtime device selection, CUDA ordinal
  selection, runtime-loaded NVRTC and cuBLAS, and device memory telemetry
- added CUDA `f32` training coverage for tensor math, autograd, Adam/AdamW,
  gradient clipping, losses, indexing, normalization, graph execution, and
  cuBLAS-backed matmul
- added CUDA regression suites and synchronized benchmark tooling, including
  CPU/GPU parity checks and decoder-LM training validation on an RTX 3090

### Changed

- made decoder-LM training device-selectable through `AFFON_TRAIN_DEVICE` and
  added neutral CPU/Metal/CUDA progress metrics
- separated compiled decoder training from eager validation so validation
  always observes the current parameters
- updated release and CI dependency pins for the CUDA-capable compute stack

### Fixed

- fixed stale validation results after compiled training updated parameters
- fixed CUDA loss, view, reduction, indexing, cast, and empty-matmul edge cases
  found by backend parity testing

## [0.2.0] - 2026-07-21

### Added

- added first-party package and app coverage to CI, including runtime e2e suites
  and TypeScript contract checks
- added public TypeScript declarations for `std:test`, `std:fs`,
  `std:process`, and `std:telemetry`
- added `affon --version` and `affon version`

### Changed

- moved the next Affon package boundary onto the Hao runtime and shared `zig-libs`
  helpers
- made `affon <file.ts|file.js>` the default script execution form while
  keeping `affon run <file.ts|file.js>` available
- moved filesystem and process usage to the underlying `std:fs` and
  `std:process` runtime modules instead of Affon-specific aliases
- refreshed runtime diagnostics docs around `std:telemetry.metrics()` and
  `scope`-based metric snapshots
- documented schema-backed runtime configuration environment variables
- enriched public docs indexes and install guidance
- updated dataset and LM tests to match the current tokenizer package and graph
  plan metadata surfaces

## [0.1.0] - 2026-05-20

### Added

- initial public release of the Affon runtime
- built-in TypeScript runtime modules for compute tensors, neural networks,
  datasets, checkpoints, process/fs interop, and testing
- eager autograd tensor execution and experimental graph capture
- CPU execution plus partial Metal acceleration for supported paths
- notebook-oriented workflows, examples, rich repr output, and SVG plotting
- public runtime docs under `docs/` and declaration docs under `types/`

### Notes

- `0.1.0` is the first public release and the supported surface is intentionally narrower than Node, NumPy, or PyTorch.
- Prefer the docs for the exact current behavior and limitations of module loading, FFI, Metal coverage, and graph capture.
