const DType = @import("../../tensor/dtype.zig").DType;
const Storage = @import("../../tensor/storage.zig").Storage;
const common = @import("common.zig");

extern fn affon_metal_matmul_add_f32(a_handle: *anyopaque, b_handle: *anyopaque, bias_handle: *anyopaque, out_handle: *anyopaque, m: usize, n: usize, k: usize) c_int;
extern fn affon_metal_matmul_add_i64(a_handle: *anyopaque, b_handle: *anyopaque, bias_handle: *anyopaque, out_handle: *anyopaque, m: usize, n: usize, k: usize) c_int;
extern fn affon_metal_matmul_add_offset_f32(a_handle: *anyopaque, b_handle: *anyopaque, bias_handle: *anyopaque, out_handle: *anyopaque, a_offset_bytes: usize, b_offset_bytes: usize, bias_offset_bytes: usize, out_offset_bytes: usize, m: usize, n: usize, k: usize) c_int;
extern fn affon_metal_matmul_add_offset_i64(a_handle: *anyopaque, b_handle: *anyopaque, bias_handle: *anyopaque, out_handle: *anyopaque, a_offset_bytes: usize, b_offset_bytes: usize, bias_offset_bytes: usize, out_offset_bytes: usize, m: usize, n: usize, k: usize) c_int;

pub fn run(dtype: DType, a: *const Storage, b: *const Storage, bias: *const Storage, out: *Storage, m: usize, n: usize, k: usize) !void {
    if (!common.isAvailable()) return error.MetalUnavailable;
    const rc = switch (dtype) {
        .f32 => affon_metal_matmul_add_f32(try a.metalHandle(), try b.metalHandle(), try bias.metalHandle(), try out.metalHandle(), m, n, k),
        .i64 => affon_metal_matmul_add_i64(try a.metalHandle(), try b.metalHandle(), try bias.metalHandle(), try out.metalHandle(), m, n, k),
        else => return error.ExecutionNotImplemented,
    };
    if (rc != 0) return error.MetalKernelLaunchFailed;
}

pub fn runOffset(dtype: DType, a: *const Storage, b: *const Storage, bias: *const Storage, out: *Storage, a_offset_bytes: usize, b_offset_bytes: usize, bias_offset_bytes: usize, out_offset_bytes: usize, m: usize, n: usize, k: usize) !void {
    if (!common.isAvailable()) return error.MetalUnavailable;
    const rc = switch (dtype) {
        .f32 => affon_metal_matmul_add_offset_f32(try a.metalHandle(), try b.metalHandle(), try bias.metalHandle(), try out.metalHandle(), a_offset_bytes, b_offset_bytes, bias_offset_bytes, out_offset_bytes, m, n, k),
        .i64 => affon_metal_matmul_add_offset_i64(try a.metalHandle(), try b.metalHandle(), try bias.metalHandle(), try out.metalHandle(), a_offset_bytes, b_offset_bytes, bias_offset_bytes, out_offset_bytes, m, n, k),
        else => return error.ExecutionNotImplemented,
    };
    if (rc != 0) return error.MetalKernelLaunchFailed;
}
