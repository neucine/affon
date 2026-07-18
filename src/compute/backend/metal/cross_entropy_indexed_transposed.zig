const DType = @import("../../types/tensor/dtype.zig").DType;
const Storage = @import("../../types/tensor/storage.zig").Storage;
const common = @import("common.zig");

extern fn affon_metal_cross_entropy_indexed_transposed_f32(
    logits_handle: *anyopaque,
    targets_handle: *anyopaque,
    out_handle: *anyopaque,
    rows: usize,
    classes: usize,
) c_int;

pub fn run(dtype: DType, logits: *const Storage, targets: *const Storage, out: *Storage, rows: usize, classes: usize) !void {
    if (!common.isAvailable()) return error.MetalUnavailable;
    if (dtype != .f32) return error.ExecutionNotImplemented;
    if (rows == 0 or classes == 0) return error.ShapeMismatch;
    const rc = affon_metal_cross_entropy_indexed_transposed_f32(
        try logits.metalHandle(),
        try targets.metalHandle(),
        try out.metalHandle(),
        rows,
        classes,
    );
    if (rc != 0) return error.MetalKernelLaunchFailed;
}
