const std = @import("std");
const DType = @import("../../types/tensor/dtype.zig").DType;
const Storage = @import("../../types/tensor/storage.zig").Storage;

fn siluTyped(comptime T: type, x: T) T {
    const sig = 1.0 / (1.0 + @exp(-x));
    return x * sig;
}

pub fn run(dtype: DType, input: *const Storage, out: *Storage) !void {
    switch (dtype) {
        .f32 => {
            const src = std.mem.bytesAsSlice(f32, try input.readableBytes());
            const dst = std.mem.bytesAsSlice(f32, try out.writableBytes());
            for (dst, src) |*d, s| d.* = siluTyped(f32, s);
        },
        .f64 => {
            const src = std.mem.bytesAsSlice(f64, try input.readableBytes());
            const dst = std.mem.bytesAsSlice(f64, try out.writableBytes());
            for (dst, src) |*d, s| d.* = siluTyped(f64, s);
        },
        .i64 => return error.ExecutionNotImplemented,
    }
}
