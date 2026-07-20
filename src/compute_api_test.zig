const std = @import("std");
const compute = @import("compute");

test "standalone client surface" {
    const engine = compute.Engine.init(std.testing.allocator, .{});
    const lhs = try engine.fromF32(&.{2}, &.{ 1, 2 });
    defer lhs.deinit();
    const rhs = try engine.fromF32(&.{2}, &.{ 3, 4 });
    defer rhs.deinit();

    const result = try engine.add(lhs, rhs);
    defer result.deinit();

    var bytes: [2 * @sizeOf(f32)]u8 = undefined;
    try engine.copyToHost(result, &bytes);
    var values: [2]f32 = undefined;
    @memcpy(std.mem.asBytes(&values), &bytes);
    try std.testing.expectEqualSlices(f32, &.{ 4, 6 }, &values);
}

test "compute is usable as an independent native module" {
    const engine = compute.Engine.init(std.testing.allocator, .{});
    const lhs = try engine.fromF32(&.{2}, &.{ 1, 2 });
    defer lhs.deinit();
    const rhs = try engine.fromF32(&.{2}, &.{ 10, 20 });
    defer rhs.deinit();

    const result = try engine.add(lhs, rhs);
    defer result.deinit();

    var bytes: [2 * @sizeOf(f32)]u8 = undefined;
    try engine.copyToHost(result, &bytes);
    var values: [2]f32 = undefined;
    @memcpy(std.mem.asBytes(&values), &bytes);
    try std.testing.expectEqualSlices(f32, &.{ 11, 22 }, &values);
}

test "shared types and sema validate without execution" {
    var shape = try compute.shared.types.tensor.Shape.initCopy(std.testing.allocator, &.{ 2, 3 });
    defer shape.deinit();
    var layout = try compute.shared.types.tensor.Layout.initContiguous(std.testing.allocator, shape);
    defer layout.deinit();
    const spec = compute.shared.types.tensor.TensorSpec{
        .shape = shape,
        .dtype = .f32,
        .layout = layout,
        .device = .cpu,
    };

    var inferred = try compute.shared.sema.inferFromSpecs(std.testing.allocator, .add, &.{ spec, spec }, .{ .none = {} });
    defer inferred.deinit();

    try std.testing.expectEqualSlices(usize, &.{ 2, 3 }, inferred.shape.dims);
    try std.testing.expectEqual(compute.shared.types.tensor.DType.f32, inferred.dtype);
}
