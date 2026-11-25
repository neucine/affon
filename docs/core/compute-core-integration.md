# Compute Core integration

Affon imports the sibling Compute Core and Hao packages directly. Build the
runtime from the repository root:

```sh
cyna build . --build default --output build/affon
```

Hao hosts the TypeScript modules in its embedded QuickJS runtime. Affon
registers its native compute and checkpoint modules with that runtime, and
TypeScript calls those registered functions directly. Compute Core owns
compilation, execution, tensor storage, and selected-plan evidence in the same
process. Compiled Programs remain owned by their Affon executable and are reused
across runs.

Sessions execute through Compute Core on CPU, Metal, or CUDA. A Session keeps
its native context and backend resident for its lifetime. Execution borrows
input storage, uploads it to the selected device, runs the cached Executable,
and reads outputs back into owned tensors. Unsupported devices and unavailable
Metal or CUDA runtimes fail explicitly. There is no alternate runtime routing or
Zig fallback.

Runtime evidence (schema 11) names the device as `cpu`, `metal`, or `cuda`.
Compilation explanations (schema 3) identify `compute_core` as the execution and
plan backend, while selected physical-node evidence names the device-specific
implementation such as `metal_matmul` or `cuda_matmul`. Selected semantic steps,
physical nodes, residual decisions, selection comparisons, and search evidence
use implementation-independent field names.

Telemetry uses `computeTiming: "none"`, `"invocation"`, or `"kernel"`.
Evidence capture and timing observe the same executable used for computation.
