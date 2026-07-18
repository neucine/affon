const std = @import("std");
const DType = @import("../../tensor/dtype.zig").DType;
const Storage = @import("../../tensor/storage.zig").Storage;
const SliceRange = @import("../../op/options.zig").SliceRange;

pub fn run(dtype: DType, input: *const Storage, out: *Storage, in_shape: []const usize, ranges: []const SliceRange) !void {
    switch (dtype) {
        .f32 => try runTyped(f32, input, out, in_shape, ranges),
        .f64 => try runTyped(f64, input, out, in_shape, ranges),
        .i64 => try runTyped(i64, input, out, in_shape, ranges),
    }
}

fn runTyped(comptime T: type, input: *const Storage, out: *Storage, in_shape: []const usize, ranges: []const SliceRange) !void {
    const src = std.mem.bytesAsSlice(T, try input.readableBytes());
    const dst = std.mem.bytesAsSlice(T, try out.writableBytes());
    const rank = in_shape.len;

    var starts: [8]usize = [_]usize{0} ** 8;
    var stops: [8]usize = [_]usize{0} ** 8;
    var steps: [8]usize = [_]usize{1} ** 8;
    var out_dims: [8]usize = [_]usize{1} ** 8;
    var in_strides: [8]usize = [_]usize{1} ** 8;

    if (rank > 8) return error.ExecutionNotImplemented;

    var i: usize = 0;
    while (i < rank) : (i += 1) {
        const r = if (i < ranges.len) ranges[i] else SliceRange{ .start = 0, .stop = in_shape[i], .step = 1 };
        starts[i] = r.start;
        stops[i] = r.stop;
        steps[i] = @intCast(r.step);
        if (steps[i] == 0) return error.InvalidSliceStep;
        out_dims[i] = if (starts[i] >= stops[i]) 0 else ((stops[i] - starts[i] - 1) / steps[i]) + 1;
    }

    if (rank > 0) {
        in_strides[rank - 1] = 1;
        var d: usize = rank - 1;
        while (d > 0) {
            d -= 1;
            in_strides[d] = in_strides[d + 1] * in_shape[d + 1];
        }
    }

    var out_index: usize = 0;
    var idx: [8]usize = [_]usize{0} ** 8;
    while (true) {
        var in_flat: usize = 0;
        i = 0;
        while (i < rank) : (i += 1) {
            in_flat += (starts[i] + idx[i] * steps[i]) * in_strides[i];
        }
        dst[out_index] = src[in_flat];
        out_index += 1;

        var dim = rank;
        while (dim > 0) {
            dim -= 1;
            idx[dim] += 1;
            if (idx[dim] < out_dims[dim]) break;
            idx[dim] = 0;
            if (dim == 0) return;
        }
    }
}
