# Affon

Affon is a package built on the Hao runtime for compute and data applications.

This repository is the next Affon package boundary. Hao owns the embedded
JavaScript runtime, module loading, async execution, and addon interface.
Affon owns its domain modules, native implementation, and TypeScript API.

## Development

This repository currently uses the sibling Hao checkout for local development:

```bash
zig build test
zig build install
./zig-out/bin/affon test test/e2e/compute/autograd.test.ts
```

The dependency will move to a released Hao source package when the first Hao
embedding release is available.

## Native compute library

The compute engine is also available as an independent Zig module:

```zig
const compute = @import("compute");

const engine = compute.Engine.init(allocator, .{});
const value = try engine.fromF32(&.{2}, &.{1, 2});
defer value.deinit();
const result = try engine.relu(value);
defer result.deinit();
```
