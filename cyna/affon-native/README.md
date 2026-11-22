# Affon Cyna runtime

This executable imports `compute-core` and Hao as Cyna path dependencies. It
does not link either project through its C ABI. Cyna compiles the bridge,
compute core, and the native Hao surface into one program with one Cyna
runtime.

Build and run it from the Affon repository root:

```sh
cyna build cyna/affon-native
AFFON_ROOT="$PWD" cyna/affon-native/build/affon-cyna-native examples/first-program.ts
```

The executable installs Affon's TypeScript modules in Hao's embedded QuickJS
runtime and registers `affon:core/native` directly from Cyna. Programs are
compiled and run through Compute Core's Cyna API in the same process. There is
no subprocess protocol or Zig execution path.

Run focused tests with:

```sh
AFFON_ROOT="$PWD" cyna/affon-native/build/affon-cyna-native \
  test test/e2e/program/execution test/e2e/dataset test/e2e/checkpoint
```
