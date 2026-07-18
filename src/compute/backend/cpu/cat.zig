const std = @import("std");
const DType = @import("../../types/tensor/dtype.zig").DType;
const Storage = @import("../../types/tensor/storage.zig").Storage;

pub fn run(dtype: DType, inputs: []const *const Storage, out: *Storage, in_shape: []const usize, axis: usize) !void {
    switch (dtype) {
        .f32 => try runTyped(f32, inputs, out, in_shape, axis),
        .f64 => try runTyped(f64, inputs, out, in_shape, axis),
        .i64 => try runTyped(i64, inputs, out, in_shape, axis),
    }
}

fn runTyped(comptime T: type, inputs: []const *const Storage, out: *Storage, in_shape: []const usize, axis: usize) !void {
    const out_buf = std.mem.bytesAsSlice(T, try out.writableBytes());

    var outer: usize = 1;
    var inner: usize = 1;
    for (in_shape[0..axis]) |d| outer *= d;
    for (in_shape[axis + 1 ..]) |d| inner *= d;

    var axis_offset: usize = 0;
    for (inputs) |input_storage| {
        const src = std.mem.bytesAsSlice(T, try input_storage.readableBytes());
        const axis_len = src.len / (outer * inner);
        for (0..outer) |o| {
            const src_base = o * axis_len * inner;
            const dst_base = o * (out_buf.len / outer) + axis_offset * inner;
            const span = axis_len * inner;
            @memcpy(out_buf[dst_base .. dst_base + span], src[src_base .. src_base + span]);
        }
        axis_offset += axis_len;
    }
}
