const Device = @import("../../tensor/device.zig").Device;
const DType = @import("../../tensor/dtype.zig").DType;
const Storage = @import("../../tensor/storage.zig").Storage;
const Value = @import("../../tensor/value.zig").Value;
const kernel_dispatch = @import("../../kernel/dispatch.zig");

pub fn dispatchLayerNorm(
    device: Device,
    dtype: DType,
    input: *const Value,
    output: *Storage,
    axis: usize,
    eps: f64,
) !void {
    try kernel_dispatch.layerNorm(
        device,
        dtype,
        input.storage orelse return error.InputNotMaterialized,
        output,
        input.shape.dims,
        axis,
        eps,
    );
}

pub fn dispatchRmsNorm(
    device: Device,
    dtype: DType,
    input: *const Value,
    output: *Storage,
    axis: usize,
    eps: f64,
) !void {
    try kernel_dispatch.rmsNorm(
        device,
        dtype,
        input.storage orelse return error.InputNotMaterialized,
        output,
        input.shape.dims,
        axis,
        eps,
    );
}
