const DType = @import("../../shared/types/tensor/dtype.zig").DType;
const Storage = @import("../../shared/types/tensor/storage.zig").Storage;
const common = @import("common.zig");

extern fn affon_metal_contiguous_f32(
    input_handle: *anyopaque,
    out_handle: *anyopaque,
    ndim: usize,
    shape: [*]const u32,
    strides: [*]const u32,
    offset: usize,
    len: usize,
) c_int;
extern fn affon_metal_contiguous_i64(
    input_handle: *anyopaque,
    out_handle: *anyopaque,
    ndim: usize,
    shape: [*]const u32,
    strides: [*]const u32,
    offset: usize,
    len: usize,
) c_int;
extern fn affon_metal_contiguous_signed_f32(
    input_handle: *anyopaque,
    out_handle: *anyopaque,
    ndim: usize,
    shape: [*]const u32,
    strides: [*]const i32,
    offset: usize,
    len: usize,
) c_int;
extern fn affon_metal_contiguous_signed_i64(
    input_handle: *anyopaque,
    out_handle: *anyopaque,
    ndim: usize,
    shape: [*]const u32,
    strides: [*]const i32,
    offset: usize,
    len: usize,
) c_int;

pub fn run(dtype: DType, input: *const Storage, out: *Storage, shape: []const usize, strides: []const isize, offset: usize) !void {
    if (!common.isAvailable()) return error.MetalUnavailable;
    if (shape.len > 8 or shape.len != strides.len) return error.ExecutionNotImplemented;
    var shape_u32: [8]u32 = [_]u32{1} ** 8;
    var strides_u32: [8]u32 = [_]u32{1} ** 8;
    var strides_i32: [8]i32 = [_]i32{1} ** 8;
    var elems: usize = 1;
    var has_negative_stride = false;
    for (shape, 0..) |dim, i| {
        shape_u32[i] = @intCast(dim);
        elems *= dim;
        if (strides[i] < 0) has_negative_stride = true;
        if (strides[i] >= 0) {
            strides_u32[i] = @intCast(strides[i]);
        }
        strides_i32[i] = @intCast(strides[i]);
    }
    const rc = switch (dtype) {
        .f32 => if (has_negative_stride)
            affon_metal_contiguous_signed_f32(
                try input.metalHandle(),
                try out.metalHandle(),
                shape.len,
                &shape_u32,
                &strides_i32,
                offset,
                elems,
            )
        else
            affon_metal_contiguous_f32(
                try input.metalHandle(),
                try out.metalHandle(),
                shape.len,
                &shape_u32,
                &strides_u32,
                offset,
                elems,
            ),
        .i64 => if (has_negative_stride)
            affon_metal_contiguous_signed_i64(
                try input.metalHandle(),
                try out.metalHandle(),
                shape.len,
                &shape_u32,
                &strides_i32,
                offset,
                elems,
            )
        else
            affon_metal_contiguous_i64(
                try input.metalHandle(),
                try out.metalHandle(),
                shape.len,
                &shape_u32,
                &strides_u32,
                offset,
                elems,
            ),
        else => return error.ExecutionNotImplemented,
    };
    if (rc != 0) return error.MetalKernelLaunchFailed;
}
