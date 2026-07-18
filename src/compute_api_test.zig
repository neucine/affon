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
