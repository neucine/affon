const std = @import("std");
const DType = @import("../../types/tensor/dtype.zig").DType;
const Storage = @import("../../types/tensor/storage.zig").Storage;
const OpTag = @import("../../types/operation/tag.zig").OpTag;

pub fn run(tag: OpTag, input_dtype: DType, input: *const Storage, out: *Storage, shape: []const usize, axis: usize, keepdim: bool) !void {
    if (shape.len == 0) return error.ShapeMismatch;
    if (axis >= shape.len) return error.InvalidAxis;

    switch (tag) {
        .sum_axis => try numericReduce(tag, input_dtype, input, out, shape, axis, keepdim),
        .mean_axis => try numericReduce(tag, input_dtype, input, out, shape, axis, keepdim),
        .min_axis => try numericReduce(tag, input_dtype, input, out, shape, axis, keepdim),
        .max_axis => try numericReduce(tag, input_dtype, input, out, shape, axis, keepdim),
        .variance_axis => try numericReduce(tag, input_dtype, input, out, shape, axis, keepdim),
        .std_axis => try numericReduce(tag, input_dtype, input, out, shape, axis, keepdim),
        .argmin_axis => try indexReduce(tag, input_dtype, input, out, shape, axis, keepdim),
        .argmax_axis => try indexReduce(tag, input_dtype, input, out, shape, axis, keepdim),
        else => return error.ExecutionNotImplemented,
    }
}

fn numericReduce(tag: OpTag, input_dtype: DType, input: *const Storage, out: *Storage, shape: []const usize, axis: usize, keepdim: bool) !void {
    switch (input_dtype) {
        .f32 => try numericReduceTyped(f32, tag, input, out, shape, axis, keepdim),
        .f64 => try numericReduceTyped(f64, tag, input, out, shape, axis, keepdim),
        .i64 => switch (tag) {
            .sum_axis, .min_axis, .max_axis => try numericReduceTyped(i64, tag, input, out, shape, axis, keepdim),
            else => return error.ExecutionNotImplemented,
        },
    }
}

fn numericReduceTyped(comptime T: type, tag: OpTag, input: *const Storage, out: *Storage, shape: []const usize, axis: usize, keepdim: bool) !void {
    const src = std.mem.bytesAsSlice(T, try input.readableBytes());
    const dst = std.mem.bytesAsSlice(T, try out.writableBytes());

    const rank = shape.len;
    var outer: usize = 1;
    var inner: usize = 1;
    const axis_len = shape[axis];

    var i: usize = 0;
    while (i < axis) : (i += 1) outer *= shape[i];
    i = axis + 1;
    while (i < rank) : (i += 1) inner *= shape[i];

    const out_count = outer * inner;
    if (!keepdim and rank > 1 and dst.len != out_count) return error.ShapeMismatch;

    var oi: usize = 0;
    while (oi < outer) : (oi += 1) {
        var ii: usize = 0;
        while (ii < inner) : (ii += 1) {
            const base = oi * axis_len * inner + ii;
            var acc = src[base];
            var a: usize = 1;
            while (a < axis_len) : (a += 1) {
                const v = src[base + a * inner];
                switch (tag) {
                    .sum_axis, .mean_axis, .variance_axis, .std_axis => acc += v,
                    .min_axis => {
                        if (v < acc) acc = v;
                    },
                    .max_axis => {
                        if (v > acc) acc = v;
                    },
                    else => return error.ExecutionNotImplemented,
                }
            }

            const out_idx = oi * inner + ii;
            switch (tag) {
                .sum_axis => dst[out_idx] = acc,
                .mean_axis => {
                    if (T == i64) return error.ExecutionNotImplemented;
                    dst[out_idx] = acc / @as(T, @floatFromInt(axis_len));
                },
                .min_axis => dst[out_idx] = acc,
                .max_axis => dst[out_idx] = acc,
                .variance_axis, .std_axis => {
                    if (T == i64) return error.ExecutionNotImplemented;
                    const mean = acc / @as(T, @floatFromInt(axis_len));
                    var var_acc: T = 0;
                    var b: usize = 0;
                    while (b < axis_len) : (b += 1) {
                        const v = src[base + b * inner];
                        const d = v - mean;
                        var_acc += d * d;
                    }
                    const variance = var_acc / @as(T, @floatFromInt(axis_len));
                    dst[out_idx] = if (tag == .std_axis) @sqrt(variance) else variance;
                },
                else => return error.ExecutionNotImplemented,
            }
        }
    }
}

fn indexReduce(tag: OpTag, input_dtype: DType, input: *const Storage, out: *Storage, shape: []const usize, axis: usize, keepdim: bool) !void {
    _ = keepdim;
    const dst = std.mem.bytesAsSlice(i64, try out.writableBytes());

    switch (input_dtype) {
        .f32 => {
            const src = std.mem.bytesAsSlice(f32, try input.readableBytes());
            try indexReduceTyped(f32, tag, src, dst, shape, axis);
        },
        .f64 => {
            const src = std.mem.bytesAsSlice(f64, try input.readableBytes());
            try indexReduceTyped(f64, tag, src, dst, shape, axis);
        },
        .i64 => {
            const src = std.mem.bytesAsSlice(i64, try input.readableBytes());
            try indexReduceTyped(i64, tag, src, dst, shape, axis);
        },
    }
}

fn indexReduceTyped(comptime T: type, tag: OpTag, src: []const T, dst: []i64, shape: []const usize, axis: usize) !void {
    const rank = shape.len;
    var outer: usize = 1;
    var inner: usize = 1;
    const axis_len = shape[axis];

    var i: usize = 0;
    while (i < axis) : (i += 1) outer *= shape[i];
    i = axis + 1;
    while (i < rank) : (i += 1) inner *= shape[i];

    const out_count = outer * inner;
    if (dst.len != out_count) return error.ShapeMismatch;

    var oi: usize = 0;
    while (oi < outer) : (oi += 1) {
        var ii: usize = 0;
        while (ii < inner) : (ii += 1) {
            const base = oi * axis_len * inner + ii;
            var best_idx: usize = 0;
            var best = src[base];

            var a: usize = 1;
            while (a < axis_len) : (a += 1) {
                const v = src[base + a * inner];
                const better = switch (tag) {
                    .argmin_axis => v < best,
                    .argmax_axis => v > best,
                    else => return error.ExecutionNotImplemented,
                };
                if (better) {
                    best = v;
                    best_idx = a;
                }
            }

            dst[oi * inner + ii] = @intCast(best_idx);
        }
    }
}
