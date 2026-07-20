const std = @import("std");
const DType = @import("../../shared/types/tensor/dtype.zig").DType;
const Storage = @import("../../shared/types/tensor/storage.zig").Storage;

pub fn run(dtype: DType, input: *const Storage, out: *Storage, shape: []const usize, axis: usize, eps: f64) !void {
    if (axis >= shape.len) return error.InvalidAxis;
    if (!(eps > 0.0)) return error.InvalidEpsilon;

    var outer: usize = 1;
    var inner: usize = 1;
    for (shape[0..axis]) |d| outer *= d;
    for (shape[axis + 1 ..]) |d| inner *= d;
    const axis_size = shape[axis];
    if (axis_size == 0) return error.InvalidShape;

    switch (dtype) {
        .f32 => {
            const src = std.mem.bytesAsSlice(f32, try input.readableBytes());
            const dst = std.mem.bytesAsSlice(f32, try out.writableBytes());
            for (0..outer) |o| {
                for (0..inner) |i| {
                    const base = o * axis_size * inner + i;
                    var mean_sq: f64 = 0.0;
                    for (0..axis_size) |k| {
                        const x = @as(f64, src[base + k * inner]);
                        mean_sq += x * x;
                    }
                    mean_sq /= @as(f64, @floatFromInt(axis_size));
                    const inv_rms = 1.0 / @sqrt(mean_sq + eps);
                    for (0..axis_size) |k| {
                        const x = @as(f64, src[base + k * inner]);
                        dst[base + k * inner] = @floatCast(x * inv_rms);
                    }
                }
            }
        },
        .f64 => {
            const src = std.mem.bytesAsSlice(f64, try input.readableBytes());
            const dst = std.mem.bytesAsSlice(f64, try out.writableBytes());
            for (0..outer) |o| {
                for (0..inner) |i| {
                    const base = o * axis_size * inner + i;
                    var mean_sq: f64 = 0.0;
                    for (0..axis_size) |k| {
                        const x = src[base + k * inner];
                        mean_sq += x * x;
                    }
                    mean_sq /= @as(f64, @floatFromInt(axis_size));
                    const inv_rms = 1.0 / @sqrt(mean_sq + eps);
                    for (0..axis_size) |k| dst[base + k * inner] = src[base + k * inner] * inv_rms;
                }
            }
        },
        .i64 => return error.ExecutionNotImplemented,
    }
}
