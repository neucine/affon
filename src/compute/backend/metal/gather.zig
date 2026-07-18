const DType = @import("../../types/tensor/dtype.zig").DType;
const Storage = @import("../../types/tensor/storage.zig").Storage;
const common = @import("common.zig");

extern fn affon_metal_gather_i64_f32(input_handle: *anyopaque, index_handle: *anyopaque, out_handle: *anyopaque, ndim: usize, shape: [*]const u32, input_strides: [*]const u32, axis: usize, axis_size: usize, len: usize) c_int;
extern fn affon_metal_gather_i64_i64(input_handle: *anyopaque, index_handle: *anyopaque, out_handle: *anyopaque, ndim: usize, shape: [*]const u32, input_strides: [*]const u32, axis: usize, axis_size: usize, len: usize) c_int;

pub fn run(
    dtype: DType,
    input: *const Storage,
    index: *const Storage,
    out: *Storage,
    out_shape: []const usize,
    axis: usize,
    input_axis_size: usize,
) !void {
    if (!common.isAvailable()) return error.MetalUnavailable;
    if (out_shape.len == 0 or out_shape.len > 8) return error.ExecutionNotImplemented;
    if (axis >= out_shape.len) return error.InvalidAxis;

    var shape_u32: [8]u32 = [_]u32{1} ** 8;
    var input_dims: [8]usize = [_]usize{1} ** 8;
    var strides_u32: [8]u32 = [_]u32{1} ** 8;
    for (out_shape, 0..) |d, i| shape_u32[i] = @intCast(d);
    for (out_shape, 0..) |d, i| input_dims[i] = d;
    input_dims[axis] = input_axis_size;

    var stride: usize = 1;
    var i = out_shape.len;
    while (i > 0) {
        i -= 1;
        strides_u32[i] = @intCast(stride);
        stride *= input_dims[i];
    }

    const input_handle = try input.metalHandle();
    const index_handle = try index.metalHandle();
    const out_handle = try out.metalHandle();
    const rc = switch (dtype) {
        .f32 => affon_metal_gather_i64_f32(
            input_handle,
            index_handle,
            out_handle,
            out_shape.len,
            &shape_u32,
            &strides_u32,
            axis,
            input_axis_size,
            out.bytes / @sizeOf(f32),
        ),
        .i64 => affon_metal_gather_i64_i64(
            input_handle,
            index_handle,
            out_handle,
            out_shape.len,
            &shape_u32,
            &strides_u32,
            axis,
            input_axis_size,
            out.bytes / @sizeOf(i64),
        ),
        else => return error.ExecutionNotImplemented,
    };
    if (rc != 0) return error.MetalKernelLaunchFailed;
}
