const std = @import("std");
const DType = @import("../../tensor/dtype.zig").DType;
const Storage = @import("../../tensor/storage.zig").Storage;

fn geluApprox(comptime T: type, x: T) T {
    const c0: T = 0.5;
    const c1: T = 0.044715;
    const c2: T = 0.7978845608028654; // sqrt(2/pi)
    return c0 * x * (1 + std.math.tanh(c2 * (x + c1 * x * x * x)));
}

pub fn run(dtype: DType, input: *const Storage, out: *Storage) !void {
    switch (dtype) {
        .f32 => {
            const src = std.mem.bytesAsSlice(f32, try input.readableBytes());
            const dst = std.mem.bytesAsSlice(f32, try out.writableBytes());
            for (dst, src) |*d, s| d.* = geluApprox(f32, s);
        },
        .f64 => {
            const src = std.mem.bytesAsSlice(f64, try input.readableBytes());
            const dst = std.mem.bytesAsSlice(f64, try out.writableBytes());
            for (dst, src) |*d, s| d.* = geluApprox(f64, s);
        },
        .i64 => return error.ExecutionNotImplemented,
    }
}
