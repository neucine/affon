const DType = @import("../../tensor/dtype.zig").DType;
const Storage = @import("../../tensor/storage.zig").Storage;
const common = @import("common.zig");

extern fn affon_metal_log_softmax_nll_nd_f32(
    logits_handle: *anyopaque,
    targets_handle: *anyopaque,
    out_handle: *anyopaque,
    ndim: usize,
    shape: [*]const u32,
    input_strides: [*]const u32,
    axis: usize,
    axis_size: usize,
    groups: usize,
) c_int;

pub fn run(dtype: DType, logits: *const Storage, targets: *const Storage, out: *Storage, shape: []const usize, axis: usize) !void {
    if (!common.isAvailable()) return error.MetalUnavailable;
    if (dtype != .f32) return error.ExecutionNotImplemented;
    if (shape.len < 2 or shape.len > 8) return error.ExecutionNotImplemented;
    if (axis >= shape.len) return error.InvalidAxis;

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

    const axis_size = shape[axis];
    if (axis_size == 0) return error.InvalidAxis;
    const groups = stride / axis_size;

    const rc = affon_metal_log_softmax_nll_nd_f32(
        try logits.metalHandle(),
        try targets.metalHandle(),
        try out.metalHandle(),
        shape.len,
        &shape_u32,
        &strides_u32,
        axis,
        axis_size,
        groups,
    );
    if (rc != 0) return error.MetalKernelLaunchFailed;
}
