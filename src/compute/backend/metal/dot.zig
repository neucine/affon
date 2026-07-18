const DType = @import("../../types/tensor/dtype.zig").DType;
const Storage = @import("../../types/tensor/storage.zig").Storage;
const common = @import("common.zig");

extern fn affon_metal_dot_f32(a_handle: *anyopaque, b_handle: *anyopaque, out_handle: *anyopaque, len: usize) c_int;
extern fn affon_metal_dot_i64(a_handle: *anyopaque, b_handle: *anyopaque, out_handle: *anyopaque, len: usize) c_int;

pub fn run(dtype: DType, a: *const Storage, b: *const Storage, out: *Storage) !void {
    if (!common.isAvailable()) return error.MetalUnavailable;
    if (dtype != .f32 and dtype != .i64) return error.ExecutionNotImplemented;
    const elem_size: usize = switch (dtype) {
        .f32 => @sizeOf(f32),
        .i64 => @sizeOf(i64),
        else => unreachable,
    };
    if (out.bytes != elem_size) return error.ShapeMismatch;
    if (a.bytes != b.bytes) return error.ShapeMismatch;
    if (a.bytes % elem_size != 0) return error.ShapeMismatch;

    const a_handle = try a.metalHandle();
    const b_handle = try b.metalHandle();
    const out_handle = try out.metalHandle();
    const len = a.bytes / elem_size;
    const rc = switch (dtype) {
        .f32 => affon_metal_dot_f32(a_handle, b_handle, out_handle, len),
        .i64 => affon_metal_dot_i64(a_handle, b_handle, out_handle, len),
        else => unreachable,
    };
    if (rc != 0) return error.MetalKernelLaunchFailed;
}
