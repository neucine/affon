const DType = @import("../../types/tensor/dtype.zig").DType;
const Storage = @import("../../types/tensor/storage.zig").Storage;
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
    if (axis > in_shape.len) return error.InvalidAxis;

    const elem_bytes = dtypeBytes(dtype);
    if (elem_bytes == 0) return error.ExecutionNotImplemented;

    var outer: usize = 1;
    var inner: usize = 1;
    for (in_shape[0..axis]) |d| outer *= d;
    for (in_shape[axis..]) |d| inner *= d;
    if (outer == 0 or inner == 0) return error.ShapeMismatch;

    const expected_out_elems = outer * inputs.len * inner;
    if (out.bytes != expected_out_elems * elem_bytes) return error.ShapeMismatch;

    const out_handle = try out.metalHandle();
    for (inputs, 0..) |input, s| {
        if (input.bytes != outer * inner * elem_bytes) return error.ShapeMismatch;
        const input_handle = try input.metalHandle();
        for (0..outer) |o| {
            const src_offset = o * inner * elem_bytes;
            const dst_offset = (o * inputs.len * inner + s * inner) * elem_bytes;
            const byte_len = inner * elem_bytes;
            const rc = affon_metal_buffer_copy(out_handle, dst_offset, input_handle, src_offset, byte_len);
            if (rc != 0) return error.MetalKernelLaunchFailed;
        }
    }
}

fn dtypeBytes(dtype: DType) usize {
    return switch (dtype) {
        .f32 => @sizeOf(f32),
        .f64 => @sizeOf(f64),
        .i64 => @sizeOf(i64),
    };
}
