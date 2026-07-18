const std = @import("std");
const compute = @import("compute");

test "compute is usable as an independent native module" {
    const Value = compute.tensor.Value;
    const lhs = try Value.fromSliceF32(std.testing.allocator, &.{2}, &.{ 1, 2 });
    defer lhs.deinit();
    const rhs = try Value.fromSliceF32(std.testing.allocator, &.{2}, &.{ 10, 20 });
    defer rhs.deinit();

    const inputs = [_]*Value{ lhs, rhs };
    const op = try compute.operation.Op.init(.add, &inputs, .{ .none = {} });
    const result = try compute.eager.execute(std.testing.allocator, op);
    defer result.deinit();

    const bytes = try result.storage.?.readableBytes();
    try std.testing.expectEqualSlices(f32, &.{ 11, 22 }, std.mem.bytesAsSlice(f32, bytes));
}
