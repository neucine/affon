const DType = @import("../../shared/types/tensor/dtype.zig").DType;
const Storage = @import("../../shared/types/tensor/storage.zig").Storage;
const common = @import("common.zig");

extern fn affon_metal_add_layer_norm_nd_f32(a_handle: *anyopaque, b_handle: *anyopaque, out_handle: *anyopaque, ndim: usize, shape: [*]const u32, input_strides: [*]const u32, axis: usize, axis_size: usize, len: usize, eps: f32) c_int;

pub fn run(dtype: DType, a: *const Storage, b: *const Storage, out: *Storage, shape: []const usize, axis: usize, eps: f64) !void {
    if (!common.isAvailable()) return error.MetalUnavailable;
    if (dtype != .f32) return error.ExecutionNotImplemented;
    if (shape.len == 0 or shape.len > 8) return error.ExecutionNotImplemented;
    if (axis >= shape.len) return error.InvalidAxis;
    if (!(eps > 0.0)) return error.InvalidEpsilon;

    var shape_u32: [8]u32 = [_]u32{1} ** 8;
    var strides_u32: [8]u32 = [_]u32{1} ** 8;
    for (shape, 0..) |d, i| shape_u32[i] = @intCast(d);

    var stride: usize = 1;
    var i = shape.len;
    while (i > 0) {
        i -= 1;
        strides_u32[i] = @intCast(stride);
        stride *= shape[i];
    }

    const len = out.bytes / @sizeOf(f32);
    const rc = affon_metal_add_layer_norm_nd_f32(
        try a.metalHandle(),
        try b.metalHandle(),
        try out.metalHandle(),
        shape.len,
        &shape_u32,
        &strides_u32,
        axis,
        shape[axis],
        len,
        @floatCast(eps),
    );
    if (rc != 0) return error.MetalKernelLaunchFailed;
}
