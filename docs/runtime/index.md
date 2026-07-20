# Runtime & Interop Docs

Runtime docs cover installing Affon, configuring process-level behavior, loading
modules, inspecting memory/telemetry, and understanding the native interop
boundary.

- [Install](./install.md)
- [Configuration](./configuration.md) - environment variables, startup/runtime mutability, and legacy aliases
- [Module Loader](./module-loader.md) - built-in `affon:*` modules, ESM package resolution, and compatibility limits
- [Memory Debugging](./memory-debugging.md) - `std:telemetry.metrics()`, traces, and memory signals
- [FFI](./ffi.md) - native interop boundary and ownership notes

## Runtime Scope

Affon is a focused TypeScript runtime for scientific computing and ML workloads.
It supports an ESM-oriented module loader and a documented set of built-in
`affon:*` and `std:*` modules. It is not intended to be a drop-in replacement
for the full Node.js runtime surface.

## Operational Notes

- Set startup-only environment variables before launching `affon`.
- Prefer `std:telemetry` for public diagnostics.
- Check the module-loader page before assuming package or Node compatibility.
- Treat FFI ownership and lifetime rules as part of the public safety contract.
