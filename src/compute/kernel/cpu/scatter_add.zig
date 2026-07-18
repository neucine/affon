const std = @import("std");
const DType = @import("../../tensor/dtype.zig").DType;
const Storage = @import("../../tensor/storage.zig").Storage;

pub fn run(dtype: DType, base: *const Storage, index: *const Storage, updates: *const Storage, out: *Storage, shape: []const usize, axis: usize) !void {
    const idx = std.mem.bytesAsSlice(i64, try index.readableBytes());
    switch (dtype) {
        .f32 => try runTyped(f32, base, idx, updates, out, shape, axis),
        .f64 => try runTyped(f64, base, idx, updates, out, shape, axis),
        .i64 => try runTyped(i64, base, idx, updates, out, shape, axis),
    }
}

fn runTyped(comptime T: type, base: *const Storage, idx: []const i64, updates: *const Storage, out: *Storage, shape: []const usize, axis: usize) !void {
    const base_src = std.mem.bytesAsSlice(T, try base.readableBytes());
    const upd = std.mem.bytesAsSlice(T, try updates.readableBytes());
    const dst = std.mem.bytesAsSlice(T, try out.writableBytes());
    if (base_src.len != dst.len) return error.ShapeMismatch;
    if (upd.len != idx.len) return error.ShapeMismatch;

    @memcpy(dst, base_src);

    var outer: usize = 1;
    var inner: usize = 1;
    const axis_len = shape[axis];
    for (shape[0..axis]) |d| outer *= d;
    for (shape[axis + 1 ..]) |d| inner *= d;

    if (outer == 0 or inner == 0) return;
    const select = idx.len / (outer * inner);
    if (select * outer * inner != idx.len) return error.ShapeMismatch;

    for (0..outer) |o| {
        for (0..select) |a| {
            for (0..inner) |i| {
                const u = o * select * inner + a * inner + i;
                const picked = idx[u];
                if (picked < 0 or picked >= axis_len) return error.IndexOutOfBounds;
                const di: usize = @intCast(picked);
                const out_index = o * axis_len * inner + di * inner + i;
                dst[out_index] += upd[u];
            }
        }
    }
}
