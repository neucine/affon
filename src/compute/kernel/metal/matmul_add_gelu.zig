const DType = @import("../../tensor/dtype.zig").DType;
const Storage = @import("../../tensor/storage.zig").Storage;
const common = @import("common.zig");

extern fn affon_metal_matmul_add_gelu_f32(a_handle: *anyopaque, b_handle: *anyopaque, bias_handle: *anyopaque, out_handle: *anyopaque, m: usize, n: usize, k: usize) c_int;
extern fn affon_metal_matmul_add_gelu_offset_f32(a_handle: *anyopaque, b_handle: *anyopaque, bias_handle: *anyopaque, out_handle: *anyopaque, a_offset_bytes: usize, b_offset_bytes: usize, bias_offset_bytes: usize, out_offset_bytes: usize, m: usize, n: usize, k: usize) c_int;

pub fn run(dtype: DType, a: *const Storage, b: *const Storage, bias: *const Storage, out: *Storage, m: usize, n: usize, k: usize) !void {
    if (!common.isAvailable()) return error.MetalUnavailable;
    if (dtype != .f32) return error.ExecutionNotImplemented;
    const rc = affon_metal_matmul_add_gelu_f32(
        try a.metalHandle(),
        try b.metalHandle(),
        try bias.metalHandle(),
        try out.metalHandle(),
        m,
        n,
        k,
    );
    if (rc != 0) return error.MetalKernelLaunchFailed;
}

pub fn runOffset(dtype: DType, a: *const Storage, b: *const Storage, bias: *const Storage, out: *Storage, a_offset_bytes: usize, b_offset_bytes: usize, bias_offset_bytes: usize, out_offset_bytes: usize, m: usize, n: usize, k: usize) !void {
    if (!common.isAvailable()) return error.MetalUnavailable;
    if (dtype != .f32) return error.ExecutionNotImplemented;
    const rc = affon_metal_matmul_add_gelu_offset_f32(
        try a.metalHandle(),
        try b.metalHandle(),
        try bias.metalHandle(),
        try out.metalHandle(),
        a_offset_bytes,
        b_offset_bytes,
        bias_offset_bytes,
        out_offset_bytes,
        m,
        n,
        k,
    );
    if (rc != 0) return error.MetalKernelLaunchFailed;
}
