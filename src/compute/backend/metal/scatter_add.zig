const DType = @import("../../shared/types/tensor/dtype.zig").DType;
const Storage = @import("../../shared/types/tensor/storage.zig").Storage;
const common = @import("common.zig");

extern fn affon_metal_scatter_add_f32(
    base_handle: *anyopaque,
    index_handle: *anyopaque,
    src_handle: *anyopaque,
    out_handle: *anyopaque,
    ndim: usize,
    dst_shape: [*]const u32,
    src_shape: [*]const u32,
    axis: usize,
    src_len: usize,
    dst_len: usize,
) c_int;
extern fn affon_metal_scatter_add_i64_f32(
    base_handle: *anyopaque,
    index_handle: *anyopaque,
    src_handle: *anyopaque,
    out_handle: *anyopaque,
    ndim: usize,
    dst_shape: [*]const u32,
    src_shape: [*]const u32,
    axis: usize,
    src_len: usize,
    dst_len: usize,
) c_int;

pub fn run(dtype: DType, index_dtype: DType, base: *const Storage, index: *const Storage, src: *const Storage, out: *Storage, dst_shape: []const usize, axis: usize) !void {
    if (!common.isAvailable()) return error.MetalUnavailable;
    if (dtype != .f32) return error.ExecutionNotImplemented;
    if (dst_shape.len == 0 or dst_shape.len > 8) return error.ExecutionNotImplemented;
    if (axis >= dst_shape.len) return error.InvalidAxis;

    var dst_shape_u32: [8]u32 = [_]u32{1} ** 8;
    var src_shape_u32: [8]u32 = [_]u32{1} ** 8;
    for (dst_shape, 0..) |d, i| dst_shape_u32[i] = @intCast(d);
    for (dst_shape, 0..) |d, i| src_shape_u32[i] = @intCast(d);
    var outer: usize = 1;
    var inner: usize = 1;
    const axis_len = dst_shape[axis];
    for (dst_shape[0..axis]) |d| outer *= d;
    for (dst_shape[axis + 1 ..]) |d| inner *= d;
    const src_len = src.bytes / @sizeOf(f32);
    if (outer == 0 or inner == 0) return;
    const select = src_len / (outer * inner);
    if (select * outer * inner != src_len) return error.ShapeMismatch;
    _ = axis_len;
    src_shape_u32[axis] = @intCast(select);

    const base_handle = try base.metalHandle();
    const index_handle = try index.metalHandle();
    const src_handle = try src.metalHandle();
    const out_handle = try out.metalHandle();
    const rc = switch (index_dtype) {
        .f32 => affon_metal_scatter_add_f32(
            base_handle,
            index_handle,
            src_handle,
            out_handle,
            dst_shape.len,
            &dst_shape_u32,
            &src_shape_u32,
            axis,
            src_len,
            out.bytes / @sizeOf(f32),
        ),
        .i64 => affon_metal_scatter_add_i64_f32(
            base_handle,
            index_handle,
            src_handle,
            out_handle,
            dst_shape.len,
            &dst_shape_u32,
            &src_shape_u32,
            axis,
            src_len,
            out.bytes / @sizeOf(f32),
        ),
        else => return error.ExecutionNotImplemented,
    };
    if (rc != 0) return error.MetalKernelLaunchFailed;
}
