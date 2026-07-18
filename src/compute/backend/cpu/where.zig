const std = @import("std");
const DType = @import("../../types/tensor/dtype.zig").DType;
const Storage = @import("../../types/tensor/storage.zig").Storage;

pub fn run(cond_dtype: DType, value_dtype: DType, cond: *const Storage, a: *const Storage, b: *const Storage, out: *Storage) !void {
    const cond_bytes = try cond.readableBytes();
    switch (value_dtype) {
        .f32 => {
            const av = std.mem.bytesAsSlice(f32, try a.readableBytes());
            const bv = std.mem.bytesAsSlice(f32, try b.readableBytes());
            const dst = std.mem.bytesAsSlice(f32, try out.writableBytes());
            for (dst, 0..) |*d, i| d.* = if (condAt(cond_dtype, cond_bytes, i)) av[i] else bv[i];
        },
        .f64 => {
            const av = std.mem.bytesAsSlice(f64, try a.readableBytes());
            const bv = std.mem.bytesAsSlice(f64, try b.readableBytes());
            const dst = std.mem.bytesAsSlice(f64, try out.writableBytes());
            for (dst, 0..) |*d, i| d.* = if (condAt(cond_dtype, cond_bytes, i)) av[i] else bv[i];
        },
        .i64 => {
            const av = std.mem.bytesAsSlice(i64, try a.readableBytes());
            const bv = std.mem.bytesAsSlice(i64, try b.readableBytes());
            const dst = std.mem.bytesAsSlice(i64, try out.writableBytes());
            for (dst, 0..) |*d, i| d.* = if (condAt(cond_dtype, cond_bytes, i)) av[i] else bv[i];
        },
    }
}

fn condAt(dtype: DType, cond_bytes: []align(8) const u8, i: usize) bool {
    return switch (dtype) {
        .f32 => std.mem.bytesAsSlice(f32, cond_bytes)[i] != 0,
        .f64 => std.mem.bytesAsSlice(f64, cond_bytes)[i] != 0,
        .i64 => std.mem.bytesAsSlice(i64, cond_bytes)[i] != 0,
    };
}
