# Changelog

All notable user-facing changes to Affon will be documented in this file.

The format is based on Keep a Changelog and the project follows Semantic Versioning.

## [Unreleased]

### Changed

- moved the next Affon package boundary onto the Hao runtime and shared `zig-libs`
  helpers
- refreshed runtime diagnostics docs around `std:telemetry.metrics()` and
  `scope`-based metric snapshots
- documented schema-backed runtime configuration environment variables

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
