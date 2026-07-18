const DType = @import("../../types/tensor/dtype.zig").DType;
const Storage = @import("../../types/tensor/storage.zig").Storage;
const common = @import("common.zig");

extern fn affon_metal_embedding_i64_f32(table_handle: *anyopaque, index_handle: *anyopaque, out_handle: *anyopaque, vocab: usize, emb_dim: usize, index_count: usize) c_int;
extern fn affon_metal_embedding_i64_i64(table_handle: *anyopaque, index_handle: *anyopaque, out_handle: *anyopaque, vocab: usize, emb_dim: usize, index_count: usize) c_int;

pub fn run(dtype: DType, table: *const Storage, index: *const Storage, out: *Storage, table_shape: []const usize, index_shape: []const usize) !void {
    if (!common.isAvailable()) return error.MetalUnavailable;
    if (table_shape.len < 2) return error.ShapeMismatch;
    var emb_dim: usize = 1;
    for (table_shape[1..]) |d| emb_dim *= d;
    const index_count: usize = if (index_shape.len == 0) 1 else blk: {
        var n: usize = 1;
        for (index_shape) |d| n *= d;
        break :blk n;
    };
    const rc = switch (dtype) {
        .f32 => affon_metal_embedding_i64_f32(
            try table.metalHandle(),
            try index.metalHandle(),
            try out.metalHandle(),
            table_shape[0],
            emb_dim,
            index_count,
        ),
        .i64 => affon_metal_embedding_i64_i64(
            try table.metalHandle(),
            try index.metalHandle(),
            try out.metalHandle(),
            table_shape[0],
            emb_dim,
            index_count,
        ),
        else => return error.ExecutionNotImplemented,
    };
    if (rc != 0) return error.MetalKernelLaunchFailed;
}
