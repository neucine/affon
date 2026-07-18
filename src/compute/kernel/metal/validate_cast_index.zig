const Storage = @import("../../tensor/storage.zig").Storage;
const common = @import("common.zig");

extern fn affon_metal_validate_cast_index_f32_i64(input_handle: *anyopaque, out_handle: *anyopaque, len: usize) c_int;

pub fn runF32ToI64(input: *const Storage, out: *Storage, len: usize) !void {
    if (!common.isAvailable()) return error.MetalUnavailable;
    const rc = affon_metal_validate_cast_index_f32_i64(
        try input.metalHandle(),
        try out.metalHandle(),
        len,
    );
    if (rc == 1) return error.InvalidArgument;
    if (rc != 0) return error.MetalKernelLaunchFailed;
}
