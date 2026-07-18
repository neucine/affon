const std = @import("std");
const DType = @import("../../types/tensor/dtype.zig").DType;
const Storage = @import("../../types/tensor/storage.zig").Storage;

pub fn run(dtype: DType, input: *const Storage, index: *const Storage, out: *Storage, shape: []const usize, axis: usize, index_len: usize) !void {
    const idx = std.mem.bytesAsSlice(i64, try index.readableBytes());
    if (idx.len != index_len) return error.ShapeMismatch;
    switch (dtype) {
        .f32 => try runTyped(f32, input, idx, out, shape, axis),
        .f64 => try runTyped(f64, input, idx, out, shape, axis),
        .i64 => try runTyped(i64, input, idx, out, shape, axis),
    }
}

fn runTyped(comptime T: type, input: *const Storage, idx: []const i64, out: *Storage, shape: []const usize, axis: usize) !void {
    const src = std.mem.bytesAsSlice(T, try input.readableBytes());
    const dst = std.mem.bytesAsSlice(T, try out.writableBytes());

    var outer: usize = 1;
    var inner: usize = 1;
    const axis_len = shape[axis];
    for (shape[0..axis]) |d| outer *= d;
    for (shape[axis + 1 ..]) |d| inner *= d;

    for (0..outer) |o| {
        for (0..idx.len) |a| {
            const pick = idx[a];
            if (pick < 0 or pick >= axis_len) return error.IndexOutOfBounds;
            const pi: usize = @intCast(pick);
            for (0..inner) |i| {
                const out_index = o * idx.len * inner + a * inner + i;
                const in_index = o * axis_len * inner + pi * inner + i;
                dst[out_index] = src[in_index];
            }
        }
    }
}
