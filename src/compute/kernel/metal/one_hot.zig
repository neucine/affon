const Storage = @import("../../tensor/storage.zig").Storage;
const common = @import("common.zig");

extern fn affon_metal_one_hot_i64_f32(index_handle: *anyopaque, out_handle: *anyopaque, num_indices: usize, num_classes: usize) c_int;

pub fn run(index: *const Storage, out: *Storage, num_classes: usize) !void {
    if (!common.isAvailable()) return error.MetalUnavailable;
    if (num_classes == 0) return error.InvalidClassCount;

    const num_indices = index.bytes / @sizeOf(i64);
    const rc = affon_metal_one_hot_i64_f32(
        try index.metalHandle(),
        try out.metalHandle(),
        num_indices,
        num_classes,
    );
    if (rc != 0) return error.MetalKernelLaunchFailed;
}
