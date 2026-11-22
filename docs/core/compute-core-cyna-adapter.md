# Cyna compute core integration

Affon has a Cyna-native integration gate at the repository root. Its manifest
imports sibling `compute-core` and Hao packages directly. The executable builds
Compute Programs, executes them through the Cyna CPU or CUDA backend, and calls
Hao's native package surface without linking either project through an exported
C ABI. Build it with:

```sh
cyna build .
```

This executable is Affon's normal entry path. `compute-core` exposes the
generic native entry through `src/lib.cy`, and Hao exposes
`src/native_api.cy`. Affon's Cyna host combines those package APIs with Hao's
embedded QuickJS runtime and registers the TypeScript-facing native module
directly.

The public TypeScript execution path selects the in-process Cyna graph for
covered Programs, including telemetry capture and structural output views. The
compiled graph stays owned by its Affon executable and is reused across runs.
Compute Core supplies tensor specifications, execution results, selected plan
evidence, and cleanup through the native integration surface.

Affon's public `affon:core` Session selects Cyna for covered Programs on CPU
and CUDA. The routing record names the configured device, transport, category,
reason, operation, and option. Evidence reports `cyna_cpu` or `cyna_cuda` from
the Session that owns the executable. Unsupported devices and unavailable CUDA
runtimes fail with `backend unavailable` instead of silently changing
backends. Evidence capture and compute timing use the same selected Cyna
execution.
