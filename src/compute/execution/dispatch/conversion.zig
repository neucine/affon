const Device = @import("../../shared/types/tensor/device.zig").Device;
const DType = @import("../../shared/types/tensor/dtype.zig").DType;
const Storage = @import("../../shared/types/tensor/storage.zig").Storage;
const Tensor = @import("../../shared/types/tensor/tensor.zig").Tensor;
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
