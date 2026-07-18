# Affon

Affon is a package built on the Hao runtime for compute and data applications.

This repository is the next Affon package boundary. Hao owns the embedded
JavaScript runtime, module loading, async execution, and addon interface.
Affon owns its domain modules, native implementation, and TypeScript API.

## Development

This repository currently uses the sibling Hao checkout for local development:

```bash
zig build test
```

The dependency will move to a released Hao source package when the first Hao
embedding release is available.

## Native compute library

The compute engine is also available as an independent Zig module:

```zig
const compute = @import("compute");

const value = try compute.tensor.Value.fromSliceF32(allocator, &.{2}, &.{1, 2});
const op = try compute.operation.Op.init(.relu, &.{value}, .{ .none = {} });
const result = try compute.eager.execute(allocator, op);
```
