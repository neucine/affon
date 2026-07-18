const std = @import("std");
const DType = @import("../../types/tensor/dtype.zig").DType;
const Storage = @import("../../types/tensor/storage.zig").Storage;

pub fn run(dtype: DType, input: *const Storage, out: *Storage, shape: []const usize, strides: []const isize, offset: usize) !void {
    switch (dtype) {
        .f32 => try runTyped(f32, input, out, shape, strides, offset),
        .f64 => try runTyped(f64, input, out, shape, strides, offset),
        .i64 => try runTyped(i64, input, out, shape, strides, offset),
    }
}

fn runTyped(comptime T: type, input: *const Storage, out: *Storage, shape: []const usize, strides: []const isize, offset: usize) !void {
    if (shape.len > 8 or shape.len != strides.len) return error.ExecutionNotImplemented;
    const src = std.mem.bytesAsSlice(T, try input.readableBytes());
    const dst = std.mem.bytesAsSlice(T, try out.writableBytes());
    const len = dst.len;
    var idx: [8]usize = [_]usize{0} ** 8;
    for (0..len) |flat| {
        var src_index: isize = @intCast(offset);
        for (0..shape.len) |d| src_index += @as(isize, @intCast(idx[d])) * strides[d];
        dst[flat] = src[@intCast(src_index)];
        var dim = shape.len;
        while (dim > 0) {
            dim -= 1;
            idx[dim] += 1;
            if (idx[dim] < shape[dim]) break;
            idx[dim] = 0;
            if (dim == 0) break;
        }
    }
}
