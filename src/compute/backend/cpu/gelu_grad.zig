const std = @import("std");
const DType = @import("../../shared/types/tensor/dtype.zig").DType;
const Storage = @import("../../shared/types/tensor/storage.zig").Storage;

fn geluGradApprox(comptime T: type, x: T) T {
    const c1: T = 0.044715;
    const c2: T = 0.7978845608028654; // sqrt(2/pi)
    const u = c2 * (x + c1 * x * x * x);
    const t = std.math.tanh(u);
    if (t >= 1 - 1e-12) return 1;
    if (t <= -1 + 1e-12) return 0;
    const sech2 = 1 - t * t;
    const du_dx = c2 * (1 + 3 * c1 * x * x);
    return 0.5 * (1 + t) + 0.5 * x * sech2 * du_dx;
}

pub fn run(dtype: DType, input: *const Storage, out: *Storage) !void {
    switch (dtype) {
        .f32 => {
            const src = std.mem.bytesAsSlice(f32, try input.readableBytes());
            const dst = std.mem.bytesAsSlice(f32, try out.writableBytes());
            for (dst, src) |*d, s| d.* = geluGradApprox(f32, s);
        },
        .f64 => {
            const src = std.mem.bytesAsSlice(f64, try input.readableBytes());
            const dst = std.mem.bytesAsSlice(f64, try out.writableBytes());
            for (dst, src) |*d, s| d.* = geluGradApprox(f64, s);
        },
        .i64 => return error.ExecutionNotImplemented,
    }
}
