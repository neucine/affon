# Affon native Cyna bridge

This executable imports `compute-core` and Hao as Cyna path dependencies. It
does not link either project through its C ABI. Cyna compiles the bridge,
compute core, and the native Hao surface into one program with one Cyna
runtime.

Run the smoke gate from the Affon repository root:

```sh
zig build test-cyna-native
```

The first slice builds a real `compute-core` add Program, executes it on the
Cyna CPU backend, and verifies that both native package APIs are loaded. The
same package is the boundary for replacing Affon's existing in-process C ABI
adapter with process-level native calls.
