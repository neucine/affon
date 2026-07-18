const std = @import("std");
const DType = @import("../../tensor/dtype.zig").DType;
const Storage = @import("../../tensor/storage.zig").Storage;

pub fn run(dtype: DType, a: *const Storage, b: *const Storage, out: *Storage, shape: []const usize, axis: usize, eps: f64) !void {
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
            const av = std.mem.bytesAsSlice(f32, try a.readableBytes());
            const bv = std.mem.bytesAsSlice(f32, try b.readableBytes());
            const ov = std.mem.bytesAsSlice(f32, try out.writableBytes());
            for (0..outer) |o| {
                for (0..inner) |i| {
                    const base = o * axis_size * inner + i;
                    var mean: f64 = 0.0;
                    for (0..axis_size) |k| mean += @as(f64, av[base + k * inner] + bv[base + k * inner]);
                    mean /= @as(f64, @floatFromInt(axis_size));
                    var variance: f64 = 0.0;
                    for (0..axis_size) |k| {
                        const x = @as(f64, av[base + k * inner] + bv[base + k * inner]);
                        const d = x - mean;
                        variance += d * d;
                    }
                    variance /= @as(f64, @floatFromInt(axis_size));
                    const inv_std = 1.0 / @sqrt(variance + eps);
                    for (0..axis_size) |k| {
                        const x = @as(f64, av[base + k * inner] + bv[base + k * inner]);
                        ov[base + k * inner] = @floatCast((x - mean) * inv_std);
                    }
                }
            }
        },
        .f64 => {
            const av = std.mem.bytesAsSlice(f64, try a.readableBytes());
            const bv = std.mem.bytesAsSlice(f64, try b.readableBytes());
            const ov = std.mem.bytesAsSlice(f64, try out.writableBytes());
            for (0..outer) |o| {
                for (0..inner) |i| {
                    const base = o * axis_size * inner + i;
                    var mean: f64 = 0.0;
                    for (0..axis_size) |k| mean += av[base + k * inner] + bv[base + k * inner];
                    mean /= @as(f64, @floatFromInt(axis_size));
                    var variance: f64 = 0.0;
                    for (0..axis_size) |k| {
                        const x = av[base + k * inner] + bv[base + k * inner];
                        const d = x - mean;
                        variance += d * d;
                    }
                    variance /= @as(f64, @floatFromInt(axis_size));
                    const inv_std = 1.0 / @sqrt(variance + eps);
                    for (0..axis_size) |k| {
                        const x = av[base + k * inner] + bv[base + k * inner];
                        ov[base + k * inner] = (x - mean) * inv_std;
                    }
                }
            }
        },
        .i64 => return error.ExecutionNotImplemented,
    }
}
