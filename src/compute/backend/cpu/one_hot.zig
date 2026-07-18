const std = @import("std");
const Storage = @import("../../types/tensor/storage.zig").Storage;

pub fn run(index: *const Storage, out: *Storage, num_classes: usize) !void {
    const src = std.mem.bytesAsSlice(i64, try index.readableBytes());
    const dst = std.mem.bytesAsSlice(f32, try out.writableBytes());
    @memset(dst, 0);
    for (src, 0..) |cls, i| {
        if (cls < 0 or cls >= num_classes) return error.IndexOutOfBounds;
        const c: usize = @intCast(cls);
        dst[i * num_classes + c] = 1;
    }
}
