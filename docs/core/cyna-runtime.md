# Affon Cyna runtime

This executable imports `compute-core` and Hao as Cyna path dependencies. It
does not link either project through its C ABI. Cyna compiles the bridge,
compute core, and the native Hao surface into one program with one Cyna
runtime.

Build and run it from the Affon repository root:

```sh
CYNA_BIN="${CYNA_BIN:-cyna}"
"$CYNA_BIN" build . --build default --output build/affon
AFFON_ROOT="$PWD" build/affon examples/first-program.ts
```

On Windows, use the `windows` build config so Cyna selects the installed GNU
target and its system libraries:

```sh
cyna build . --build windows --output build/affon
```

On Linux, use the `linux` config. It omits Apple-only Metal sources while
keeping the CUDA and CPU backends available:

```sh
cyna build . --build linux --output build/affon
```

The executable installs Affon's TypeScript modules in Hao's embedded QuickJS
runtime and registers `affon:core/native` directly from Cyna. QuickJS is Hao's
internal JavaScript host, so Affon does not run or coordinate a separate Cyna
process. Programs are compiled and run through Compute Core's Cyna API in the
same process.

Each TypeScript `Session` owns one persistent native context for its lifetime.
CPU Sessions use a CPU context. `metal` Sessions use a Metal context and
PreparedBackend. `cuda` and `cuda:N` Sessions use the CUDA PreparedBackend and
retain one CUDA context until the Session is disposed. TypeScript calls the
registered native API directly; input tensors are borrowed for each invocation,
device buffers are released after execution, and compiled Executables remain
resident so repeated runs do not recompile the Program.

There is no subprocess protocol, exported C ABI between these projects, or Zig
execution path.

TypeScript readback goes through `Tensor.to_bytes()` and a typed JavaScript
decoder. This keeps the native bridge to one buffer transfer and avoids one
Hao-to-QuickJS call per tensor element when constructing `to_array()` results.

Run focused tests with:

```sh
AFFON_ROOT="$PWD" build/affon \
  test test/e2e/program/execution test/e2e/dataset test/e2e/checkpoint
```
