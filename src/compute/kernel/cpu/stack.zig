const std = @import("std");
const DType = @import("../../tensor/dtype.zig").DType;
const Storage = @import("../../tensor/storage.zig").Storage;

pub fn run(dtype: DType, inputs: []const *const Storage, out: *Storage, in_shape: []const usize, axis: usize) !void {
    switch (dtype) {
        .f32 => try runTyped(f32, inputs, out, in_shape, axis),
        .f64 => try runTyped(f64, inputs, out, in_shape, axis),
        .i64 => try runTyped(i64, inputs, out, in_shape, axis),
    }
}

fn runTyped(comptime T: type, inputs: []const *const Storage, out: *Storage, in_shape: []const usize, axis: usize) !void {
    const dst = std.mem.bytesAsSlice(T, try out.writableBytes());
    var outer: usize = 1;
    var inner: usize = 1;
    for (in_shape[0..axis]) |d| outer *= d;
    for (in_shape[axis..]) |d| inner *= d;

    for (inputs, 0..) |input_storage, s| {
        const src = std.mem.bytesAsSlice(T, try input_storage.readableBytes());
        for (0..outer) |o| {
            for (0..inner) |i| {
                const dst_idx = o * (inputs.len * inner) + s * inner + i;
                const src_idx = o * inner + i;
                dst[dst_idx] = src[src_idx];
            }
        }
    }
}
