const DType = @import("../../shared/types/tensor/dtype.zig").DType;
const Layout = @import("../../shared/types/tensor/layout.zig").Layout;
const Storage = @import("../../shared/types/tensor/storage.zig").Storage;
const common = @import("common.zig");

extern fn affon_metal_softmax_f32(a_handle: *anyopaque, out_handle: *anyopaque, rows: usize, cols: usize, axis: usize) c_int;
extern fn affon_metal_softmax_nd_f32(a_handle: *anyopaque, out_handle: *anyopaque, ndim: usize, shape: [*]const u32, input_strides: [*]const u32, axis: usize, axis_size: usize, len: usize) c_int;

pub fn run(dtype: DType, input: *const Storage, out: *Storage, shape: []const usize, layout: Layout, axis: usize) !void {
    if (!common.isAvailable()) return error.MetalUnavailable;
    if (dtype != .f32) return error.ExecutionNotImplemented;
    if (layout.offset != 0) return error.ExecutionNotImplemented;
    if (shape.len == 2 and isDenseLayout(shape, layout)) {
        if (axis > 1) return error.InvalidAxis;
        const input_handle = try input.metalHandle();
        const out_handle = try out.metalHandle();
        const rc = affon_metal_softmax_f32(input_handle, out_handle, shape[0], shape[1], axis);
        if (rc != 0) return error.MetalKernelLaunchFailed;
        return;
    }
    if (shape.len == 1 and isDenseLayout(shape, layout)) {
        if (axis != 0) return error.InvalidAxis;
        const input_handle = try input.metalHandle();
        const out_handle = try out.metalHandle();
        // Lift vector to 1xN and reduce along cols.
        const rc = affon_metal_softmax_f32(input_handle, out_handle, 1, shape[0], 1);
        if (rc != 0) return error.MetalKernelLaunchFailed;
        return;
    }
    if (shape.len > 8) return error.ExecutionNotImplemented;
    if (axis >= shape.len) return error.InvalidAxis;

    var shape_u32: [8]u32 = [_]u32{1} ** 8;
    var strides_u32: [8]u32 = [_]u32{1} ** 8;
    for (shape, 0..) |d, i| {
        shape_u32[i] = @intCast(d);
        if (layout.strides[i] < 0) return error.ExecutionNotImplemented;
        strides_u32[i] = @intCast(layout.strides[i]);
    }

    const len = out.bytes / @sizeOf(f32);
    const rc = affon_metal_softmax_nd_f32(
        try input.metalHandle(),
        try out.metalHandle(),
        shape.len,
        &shape_u32,
        &strides_u32,
        axis,
        shape[axis],
        len,
    );
    if (rc != 0) return error.MetalKernelLaunchFailed;
}

fn isDenseLayout(shape: []const usize, layout: Layout) bool {
    if (layout.offset != 0 or layout.strides.len != shape.len) return false;
    var expected: isize = 1;
    var i = shape.len;
    while (i > 0) {
        i -= 1;
        if (layout.strides[i] != expected) return false;
        expected *= @as(isize, @intCast(shape[i]));
    }
    return true;
}
