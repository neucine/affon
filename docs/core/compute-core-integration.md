# Compute Core integration

Affon imports the sibling Compute Core and Hao packages directly. Build the
runtime from the repository root:

```sh
cyna build . --build default --output build/affon
```

Hao hosts the TypeScript modules in QuickJS. Affon registers its native compute
and checkpoint modules with that runtime. Compute Core owns compilation,
execution, tensor storage, and selected-plan evidence in the same process.
Compiled Programs remain owned by their Affon executable and are reused across
runs.

Sessions execute through Compute Core on CPU or CUDA. Unsupported devices and
unavailable CUDA runtimes fail explicitly. There is no alternate runtime routing
or Zig fallback. Runtime evidence (schema 11) names the device as `cpu` or `cuda`;
compilation explanations (schema 3) identify `compute_core` as the execution
and plan backend. Selected semantic steps, physical nodes, residual decisions,
selection comparisons, and search evidence use implementation-independent field
names.

Telemetry uses `computeTiming: "none"`, `"invocation"`, or `"kernel"`.
Evidence capture and timing observe the same executable used for computation.
