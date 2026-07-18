const std = @import("std");
const DType = @import("../../types/tensor/dtype.zig").DType;
const Storage = @import("../../types/tensor/storage.zig").Storage;

pub fn run(dtype: DType, table: *const Storage, index: *const Storage, out: *Storage, table_shape: []const usize, index_shape: []const usize) !void {
    if (table_shape.len < 2) return error.ShapeMismatch;
    const idx = std.mem.bytesAsSlice(i64, try index.readableBytes());
    const vocab = table_shape[0];
    var emb_dim: usize = 1;
    for (table_shape[1..]) |d| emb_dim *= d;
    const index_count: usize = if (index_shape.len == 0) 1 else blk: {
        var n: usize = 1;
        for (index_shape) |d| n *= d;
        break :blk n;
    };
    if (idx.len != index_count) return error.ShapeMismatch;

    switch (dtype) {
        .f32 => try runTyped(f32, table, idx, out, vocab, emb_dim),
        .f64 => try runTyped(f64, table, idx, out, vocab, emb_dim),
        .i64 => try runTyped(i64, table, idx, out, vocab, emb_dim),
    }
}

fn runTyped(comptime T: type, table: *const Storage, idx: []const i64, out: *Storage, vocab: usize, emb_dim: usize) !void {
    const src = std.mem.bytesAsSlice(T, try table.readableBytes());
    const dst = std.mem.bytesAsSlice(T, try out.writableBytes());
    for (idx, 0..) |token, i| {
        if (token < 0 or token >= vocab) return error.IndexOutOfBounds;
        const t: usize = @intCast(token);
        const src_base = t * emb_dim;
        const dst_base = i * emb_dim;
        @memcpy(dst[dst_base .. dst_base + emb_dim], src[src_base .. src_base + emb_dim]);
    }
}
