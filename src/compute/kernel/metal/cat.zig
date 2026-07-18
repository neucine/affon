const DType = @import("../../tensor/dtype.zig").DType;
const Storage = @import("../../tensor/storage.zig").Storage;
const common = @import("common.zig");

extern fn affon_metal_buffer_copy(
    dst_handle: *anyopaque,
    dst_offset: usize,
    src_handle: *anyopaque,
    src_offset: usize,
    byte_len: usize,
) c_int;

pub fn run(dtype: DType, inputs: []const *const Storage, out: *Storage, in_shape: []const usize, axis: usize) !void {
    if (!common.isAvailable()) return error.MetalUnavailable;
    if (inputs.len == 0) return error.InvalidInputCount;
    if (axis >= in_shape.len) return error.InvalidAxis;

    const elem_bytes = dtypeBytes(dtype);
    if (elem_bytes == 0) return error.ExecutionNotImplemented;

    var outer: usize = 1;
    var inner: usize = 1;
    for (in_shape[0..axis]) |d| outer *= d;
    for (in_shape[axis + 1 ..]) |d| inner *= d;
    if (outer == 0 or inner == 0) return error.ShapeMismatch;

    if (out.bytes % (elem_bytes * outer * inner) != 0) return error.ShapeMismatch;
    const out_axis_len = out.bytes / (elem_bytes * outer * inner);
    var axis_offset: usize = 0;

    const out_handle = try out.metalHandle();
    for (inputs) |input| {
        if (input.bytes % (elem_bytes * outer * inner) != 0) return error.ShapeMismatch;
        const axis_len = input.bytes / (elem_bytes * outer * inner);
        const input_handle = try input.metalHandle();
        for (0..outer) |o| {
            const src_offset = o * axis_len * inner * elem_bytes;
            const dst_offset = (o * out_axis_len * inner + axis_offset * inner) * elem_bytes;
            const byte_len = axis_len * inner * elem_bytes;
            const rc = affon_metal_buffer_copy(out_handle, dst_offset, input_handle, src_offset, byte_len);
            if (rc != 0) return error.MetalKernelLaunchFailed;
        }
        axis_offset += axis_len;
    }
    if (axis_offset != out_axis_len) return error.ShapeMismatch;
}

fn dtypeBytes(dtype: DType) usize {
    return switch (dtype) {
        .f32 => @sizeOf(f32),
        .f64 => @sizeOf(f64),
        .i64 => @sizeOf(i64),
    };
}
