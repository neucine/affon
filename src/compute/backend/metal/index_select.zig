const DType = @import("../../shared/types/tensor/dtype.zig").DType;
const Storage = @import("../../shared/types/tensor/storage.zig").Storage;
const common = @import("common.zig");

extern fn affon_metal_index_select_i64_f32(
    input_handle: *anyopaque,
    index_handle: *anyopaque,
    out_handle: *anyopaque,
    ndim: usize,
    out_shape: [*]const u32,
    input_strides: [*]const u32,
    axis: usize,
    index_len: usize,
    axis_size: usize,
    len: usize,
) c_int;
extern fn affon_metal_index_select_i64_i64(
    input_handle: *anyopaque,
    index_handle: *anyopaque,
    out_handle: *anyopaque,
    ndim: usize,
    out_shape: [*]const u32,
    input_strides: [*]const u32,
    axis: usize,
    index_len: usize,
    axis_size: usize,
    len: usize,
) c_int;

pub fn run(dtype: DType, input: *const Storage, index: *const Storage, out: *Storage, shape: []const usize, axis: usize, index_len: usize) !void {
    if (!common.isAvailable()) return error.MetalUnavailable;
    if (shape.len == 0 or shape.len > 8) return error.ExecutionNotImplemented;
    if (axis >= shape.len) return error.InvalidAxis;

    var out_shape_u32: [8]u32 = [_]u32{1} ** 8;
    var strides_u32: [8]u32 = [_]u32{1} ** 8;
    for (shape, 0..) |d, i| {
        out_shape_u32[i] = @intCast(d);
    }
    out_shape_u32[axis] = @intCast(index_len);

    var stride: usize = 1;
    var i = shape.len;
    while (i > 0) {
        i -= 1;
        strides_u32[i] = @intCast(stride);
        stride *= shape[i];
    }

    const input_handle = try input.metalHandle();
    const index_handle = try index.metalHandle();
    const out_handle = try out.metalHandle();
    const rc = switch (dtype) {
        .f32 => affon_metal_index_select_i64_f32(
            input_handle,
            index_handle,
            out_handle,
            shape.len,
            &out_shape_u32,
            &strides_u32,
            axis,
            index_len,
            shape[axis],
            out.bytes / @sizeOf(f32),
        ),
        .i64 => affon_metal_index_select_i64_i64(
            input_handle,
            index_handle,
            out_handle,
            shape.len,
            &out_shape_u32,
            &strides_u32,
            axis,
            index_len,
            shape[axis],
            out.bytes / @sizeOf(i64),
        ),
        else => return error.ExecutionNotImplemented,
    };
    if (rc != 0) return error.MetalKernelLaunchFailed;
}
