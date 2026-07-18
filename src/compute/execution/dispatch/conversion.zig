const Device = @import("../../types/tensor/device.zig").Device;
const DType = @import("../../types/tensor/dtype.zig").DType;
const Storage = @import("../../types/tensor/storage.zig").Storage;
const Value = @import("../../types/tensor/value.zig").Value;
const kernel_dispatch = @import("../../backend/dispatch.zig");

pub fn dispatchCast(
    device: Device,
    from: DType,
    to: DType,
    input: *const Value,
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
