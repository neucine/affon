const DType = @import("../../tensor/dtype.zig").DType;
const Storage = @import("../../tensor/storage.zig").Storage;
const common = @import("common.zig");

extern fn affon_metal_attention_scores_f32(q_handle: *anyopaque, k_t_handle: *anyopaque, scale_handle: *anyopaque, mask_handle: *anyopaque, out_handle: *anyopaque, mask_fill_value: f32, m: usize, n: usize, k: usize) c_int;
extern fn affon_metal_attention_scores_offset_f32(q_handle: *anyopaque, k_t_handle: *anyopaque, scale_handle: *anyopaque, mask_handle: *anyopaque, out_handle: *anyopaque, q_offset_bytes: usize, k_t_offset_bytes: usize, scale_offset_bytes: usize, mask_offset_bytes: usize, out_offset_bytes: usize, mask_fill_value: f32, m: usize, n: usize, k: usize) c_int;

pub fn run(dtype: DType, q: *const Storage, k_t: *const Storage, scale: *const Storage, mask: *const Storage, out: *Storage, mask_fill_value: f64, m: usize, n: usize, k: usize) !void {
    if (!common.isAvailable()) return error.MetalUnavailable;
    if (dtype != .f32) return error.ExecutionNotImplemented;
    const rc = affon_metal_attention_scores_f32(
        try q.metalHandle(),
        try k_t.metalHandle(),
        try scale.metalHandle(),
        try mask.metalHandle(),
        try out.metalHandle(),
        @floatCast(mask_fill_value),
        m,
        n,
        k,
    );
    if (rc != 0) return error.MetalKernelLaunchFailed;
}

pub fn runOffset(dtype: DType, q: *const Storage, k_t: *const Storage, scale: *const Storage, mask: *const Storage, out: *Storage, q_offset_bytes: usize, k_t_offset_bytes: usize, scale_offset_bytes: usize, mask_offset_bytes: usize, out_offset_bytes: usize, m: usize, n: usize, k: usize, mask_fill_value: f64) !void {
    if (!common.isAvailable()) return error.MetalUnavailable;
    if (dtype != .f32) return error.ExecutionNotImplemented;
    const rc = affon_metal_attention_scores_offset_f32(
        try q.metalHandle(),
        try k_t.metalHandle(),
        try scale.metalHandle(),
        try mask.metalHandle(),
        try out.metalHandle(),
        q_offset_bytes,
        k_t_offset_bytes,
        scale_offset_bytes,
        mask_offset_bytes,
        out_offset_bytes,
        @floatCast(mask_fill_value),
        m,
        n,
        k,
    );
    if (rc != 0) return error.MetalKernelLaunchFailed;
}
