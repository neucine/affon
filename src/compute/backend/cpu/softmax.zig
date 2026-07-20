const std = @import("std");
const DType = @import("../../shared/types/tensor/dtype.zig").DType;
const Storage = @import("../../shared/types/tensor/storage.zig").Storage;

pub fn run(dtype: DType, input: *const Storage, out: *Storage, shape: []const usize, axis: usize) !void {
    if (shape.len == 0 or axis >= shape.len) return error.InvalidAxis;
    switch (dtype) {
        .f32 => try runTyped(f32, input, out, shape, axis),
        .f64 => try runTyped(f64, input, out, shape, axis),
        .i64 => return error.ExecutionNotImplemented,
    }
}

fn runTyped(comptime T: type, input: *const Storage, out: *Storage, shape: []const usize, axis: usize) !void {
    const src = std.mem.bytesAsSlice(T, try input.readableBytes());
    const dst = std.mem.bytesAsSlice(T, try out.writableBytes());

    var outer: usize = 1;
    var inner: usize = 1;
    const axis_len = shape[axis];

    var i: usize = 0;
    while (i < axis) : (i += 1) outer *= shape[i];
    i = axis + 1;
    while (i < shape.len) : (i += 1) inner *= shape[i];

    var oi: usize = 0;
    while (oi < outer) : (oi += 1) {
        var ii: usize = 0;
        while (ii < inner) : (ii += 1) {
            const base = oi * axis_len * inner + ii;

            var max_v = src[base];
            var a: usize = 1;
            while (a < axis_len) : (a += 1) {
                const v = src[base + a * inner];
                if (v > max_v) max_v = v;
            }

            var sum_exp: T = 0;
            a = 0;
            while (a < axis_len) : (a += 1) {
                const e = @exp(src[base + a * inner] - max_v);
                dst[base + a * inner] = e;
                sum_exp += e;
            }

            a = 0;
            while (a < axis_len) : (a += 1) {
                dst[base + a * inner] /= sum_exp;
            }
        }
    }
}
