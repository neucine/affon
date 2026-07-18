const std = @import("std");
const DType = @import("../../tensor/dtype.zig").DType;
const Storage = @import("../../tensor/storage.zig").Storage;

pub fn run(dtype: DType, a: *const Storage, b: *const Storage, out: *Storage) !void {
    switch (dtype) {
        .f32 => {
            const av = std.mem.bytesAsSlice(f32, try a.readableBytes());
            const bv = std.mem.bytesAsSlice(f32, try b.readableBytes());
            const dst = std.mem.bytesAsSlice(f32, try out.writableBytes());
            var acc: f32 = 0;
            for (av, bv) |x, y| acc += x * y;
            dst[0] = acc;
        },
        .f64 => {
            const av = std.mem.bytesAsSlice(f64, try a.readableBytes());
            const bv = std.mem.bytesAsSlice(f64, try b.readableBytes());
            const dst = std.mem.bytesAsSlice(f64, try out.writableBytes());
            var acc: f64 = 0;
            for (av, bv) |x, y| acc += x * y;
            dst[0] = acc;
        },
        .i64 => {
            const av = std.mem.bytesAsSlice(i64, try a.readableBytes());
            const bv = std.mem.bytesAsSlice(i64, try b.readableBytes());
            const dst = std.mem.bytesAsSlice(i64, try out.writableBytes());
            var acc: i64 = 0;
            for (av, bv) |x, y| acc += x * y;
            dst[0] = acc;
        },
    }
}
