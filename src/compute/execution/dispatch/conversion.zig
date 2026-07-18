const Device = @import("../../types/tensor/device.zig").Device;
const DType = @import("../../types/tensor/dtype.zig").DType;
const Storage = @import("../../types/tensor/storage.zig").Storage;
const Tensor = @import("../../types/tensor/tensor.zig").Tensor;
const kernel_dispatch = @import("../../backend/dispatch.zig");

pub fn dispatchCast(
    device: Device,
    from: DType,
    to: DType,
    input: *const Tensor,
    output: *Storage,
) !void {
    try kernel_dispatch.cast(
        device,
        from,
        to,
        input.storage orelse return error.InputNotMaterialized,
        output,
    );
}
