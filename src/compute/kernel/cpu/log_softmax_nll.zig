const std = @import("std");
const DType = @import("../../tensor/dtype.zig").DType;
const Storage = @import("../../tensor/storage.zig").Storage;

pub fn run(dtype: DType, logits: *const Storage, targets: *const Storage, out: *Storage, shape: []const usize, axis: usize) !void {
    if (shape.len < 2 or shape.len > 8) return error.ShapeMismatch;
    if (axis >= shape.len) return error.InvalidAxis;
    switch (dtype) {
        .f32 => try runTyped(f32, logits, targets, out, shape, axis),
        .f64 => try runTyped(f64, logits, targets, out, shape, axis),
        .i64 => return error.ExecutionNotImplemented,
    }
}

fn runTyped(comptime T: type, logits: *const Storage, targets: *const Storage, out: *Storage, shape: []const usize, axis: usize) !void {
    const x = std.mem.bytesAsSlice(T, try logits.readableBytes());
    const t = std.mem.bytesAsSlice(T, try targets.readableBytes());
    const y = std.mem.bytesAsSlice(T, try out.writableBytes());

    var outer: usize = 1;
    var inner: usize = 1;
    const axis_len = shape[axis];
    var i: usize = 0;
    while (i < axis) : (i += 1) outer *= shape[i];
    i = axis + 1;
    while (i < shape.len) : (i += 1) inner *= shape[i];

    var acc: T = 0;
    const groups = outer * inner;
    var g: usize = 0;
    while (g < groups) : (g += 1) {
        const oi = g / inner;
        const ii = g % inner;
        const base = oi * axis_len * inner + ii;

        var max_v = x[base];
        var a: usize = 1;
        while (a < axis_len) : (a += 1) {
            const v = x[base + a * inner];
            if (v > max_v) max_v = v;
        }

        var sum_exp: T = 0;
        a = 0;
        while (a < axis_len) : (a += 1) {
            sum_exp += @exp(x[base + a * inner] - max_v);
        }
        const log_sum_exp = @log(sum_exp);

        var row_nll: T = 0;
        a = 0;
        while (a < axis_len) : (a += 1) {
            const log_softmax = (x[base + a * inner] - max_v) - log_sum_exp;
            row_nll += -t[base + a * inner] * log_softmax;
        }
        acc += row_nll;
    }

    y[0] = acc / @as(T, @floatFromInt(groups));
}
