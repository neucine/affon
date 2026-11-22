# Affon Cyna runtime

This executable imports `compute-core` and Hao as Cyna path dependencies. It
does not link either project through its C ABI. Cyna compiles the bridge,
compute core, and the native Hao surface into one program with one Cyna
runtime.

Build and run it from the Affon repository root:

```sh
cyna build .
AFFON_ROOT="$PWD" build/affon examples/first-program.ts
```

The executable installs Affon's TypeScript modules in Hao's embedded QuickJS
runtime and registers `affon:core/native` directly from Cyna. Programs are
compiled and run through Compute Core's Cyna API in the same process. CPU
Sessions use a CPU backend context; `cuda` and `cuda:N` Sessions use the CUDA
PreparedBackend and keep one CUDA context for the Session lifetime. There is no
subprocess protocol, exported C ABI between these projects, or Zig execution
path.

Run focused tests with:

```sh
AFFON_ROOT="$PWD" build/affon \
  test test/e2e/program/execution test/e2e/dataset test/e2e/checkpoint
```
