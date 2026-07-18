const Device = @import("../../tensor/device.zig").Device;
const DType = @import("../../tensor/dtype.zig").DType;
const Storage = @import("../../tensor/storage.zig").Storage;
const Value = @import("../../tensor/value.zig").Value;
const kernel_dispatch = @import("../../kernel/dispatch.zig");

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
