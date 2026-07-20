const Storage = @import("../../shared/types/tensor/storage.zig").Storage;
const common = @import("common.zig");

extern fn affon_metal_buffer_copy(
    dst_handle: *anyopaque,
    dst_offset: usize,
    src_handle: *anyopaque,
    src_offset: usize,
    byte_len: usize,
) c_int;

pub fn run(src: *const Storage, dst: *Storage, byte_len: usize) !void {
    if (!common.isAvailable()) return error.MetalUnavailable;
    if (byte_len > src.bytes or byte_len > dst.bytes) return error.SizeMismatch;
    const rc = affon_metal_buffer_copy(try dst.metalHandle(), 0, try src.metalHandle(), 0, byte_len);
    if (rc != 0) return error.MetalKernelLaunchFailed;
}
