const DType = @import("../../shared/types/tensor/dtype.zig").DType;
const Storage = @import("../../shared/types/tensor/storage.zig").Storage;
const common = @import("common.zig");

extern fn affon_metal_topk_dir_i64_f32(
    input_handle: *anyopaque,
    values_handle: *anyopaque,
    indices_handle: *anyopaque,
    rows: usize,
    cols: usize,
    axis: usize,
    k: usize,
    largest: usize,
) c_int;
extern fn affon_metal_topk_nd_i64_f32(
    input_handle: *anyopaque,
    values_handle: *anyopaque,
    indices_handle: *anyopaque,
    ndim: usize,
    out_shape: [*]const u32,
    out_strides: [*]const u32,
    input_strides: [*]const u32,
    axis: usize,
    axis_size: usize,
    k: usize,
    largest: usize,
    out_len: usize,
) c_int;
extern fn affon_metal_topk_dir_i64_i64(
    input_handle: *anyopaque,
    values_handle: *anyopaque,
    indices_handle: *anyopaque,
    rows: usize,
    cols: usize,
    axis: usize,
    k: usize,
    largest: usize,
) c_int;
extern fn affon_metal_topk_nd_i64_i64(
    input_handle: *anyopaque,
    values_handle: *anyopaque,
    indices_handle: *anyopaque,
    ndim: usize,
    out_shape: [*]const u32,
    out_strides: [*]const u32,
    input_strides: [*]const u32,
    axis: usize,
    axis_size: usize,
    k: usize,
    largest: usize,
    out_len: usize,
) c_int;

pub fn run(
    dtype: DType,
    input: *const Storage,
    values_out: *Storage,
    indices_out: *Storage,
    shape: []const usize,
    axis: usize,
    k: usize,
    largest: bool,
) !void {
    if (!common.isAvailable()) return error.MetalUnavailable;
    if (k == 0 or k > 64) return error.ExecutionNotImplemented;

    var rows: usize = undefined;
    var cols: usize = undefined;
    var kernel_axis: usize = undefined;
    if (shape.len == 2) {
        if (axis > 1) return error.ExecutionNotImplemented;
        rows = shape[0];
        cols = shape[1];
        kernel_axis = axis;
    } else if (shape.len == 1) {
        if (axis != 0) return error.ExecutionNotImplemented;
        rows = 1;
        cols = shape[0];
        kernel_axis = 1;
    } else {
        if (shape.len > 8) return error.ExecutionNotImplemented;
        if (axis >= shape.len) return error.InvalidAxis;

        var out_shape_u32: [8]u32 = [_]u32{1} ** 8;
        var out_strides_u32: [8]u32 = [_]u32{1} ** 8;
        var input_strides_u32: [8]u32 = [_]u32{1} ** 8;
        for (shape, 0..) |d, i| out_shape_u32[i] = @intCast(d);
        out_shape_u32[axis] = @intCast(k);

        var stride: usize = 1;
        var i = shape.len;
        while (i > 0) {
            i -= 1;
            input_strides_u32[i] = @intCast(stride);
            stride *= shape[i];
        }

        stride = 1;
        i = shape.len;
        while (i > 0) {
            i -= 1;
            out_strides_u32[i] = @intCast(stride);
            stride *= @as(usize, out_shape_u32[i]);
        }

        const input_handle = try input.metalHandle();
        const values_handle = try values_out.metalHandle();
        const indices_handle = try indices_out.metalHandle();
        const rc_nd = switch (dtype) {
            .f32 => affon_metal_topk_nd_i64_f32(
                input_handle,
                values_handle,
                indices_handle,
                shape.len,
                &out_shape_u32,
                &out_strides_u32,
                &input_strides_u32,
                axis,
                shape[axis],
                k,
                if (largest) 1 else 0,
                values_out.bytes / @sizeOf(f32),
            ),
            .i64 => affon_metal_topk_nd_i64_i64(
                input_handle,
                values_handle,
                indices_handle,
                shape.len,
                &out_shape_u32,
                &out_strides_u32,
                &input_strides_u32,
                axis,
                shape[axis],
                k,
                if (largest) 1 else 0,
                values_out.bytes / @sizeOf(i64),
            ),
            else => return error.ExecutionNotImplemented,
        };
        if (rc_nd != 0) return error.MetalKernelLaunchFailed;
        return;
    }

    const input_handle = try input.metalHandle();
    const values_handle = try values_out.metalHandle();
    const indices_handle = try indices_out.metalHandle();
    const rc = switch (dtype) {
        .f32 => affon_metal_topk_dir_i64_f32(
            input_handle,
            values_handle,
            indices_handle,
            rows,
            cols,
            kernel_axis,
            k,
            if (largest) 1 else 0,
        ),
        .i64 => affon_metal_topk_dir_i64_i64(
            input_handle,
            values_handle,
            indices_handle,
            rows,
            cols,
            kernel_axis,
            k,
            if (largest) 1 else 0,
        ),
        else => return error.ExecutionNotImplemented,
    };
    if (rc != 0) return error.MetalKernelLaunchFailed;
}
