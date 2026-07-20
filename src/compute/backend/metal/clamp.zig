const DType = @import("../../shared/types/tensor/dtype.zig").DType;
const Storage = @import("../../shared/types/tensor/storage.zig").Storage;
const common = @import("common.zig");

extern fn affon_metal_clamp_f32(
    a_handle: *anyopaque,
    out_handle: *anyopaque,
    len: usize,
    min_val: f32,
    max_val: f32,
) c_int;
extern fn affon_metal_clamp_i64(
    a_handle: *anyopaque,
    out_handle: *anyopaque,
    len: usize,
    min_val: i64,
    max_val: i64,
) c_int;

pub fn run(dtype: DType, input: *const Storage, out: *Storage, min_val: f64, max_val: f64) !void {
    if (!common.isAvailable()) return error.MetalUnavailable;
    const input_handle = try input.metalHandle();
    const out_handle = try out.metalHandle();
    const rc = switch (dtype) {
        .f32 => affon_metal_clamp_f32(
            input_handle,
            out_handle,
            out.bytes / @sizeOf(f32),
            @floatCast(min_val),
            @floatCast(max_val),
        ),
        .i64 => affon_metal_clamp_i64(
            input_handle,
            out_handle,
            out.bytes / @sizeOf(i64),
            @intFromFloat(min_val),
            @intFromFloat(max_val),
        ),
        else => return error.ExecutionNotImplemented,
    };
    if (rc != 0) return error.MetalKernelLaunchFailed;
}
