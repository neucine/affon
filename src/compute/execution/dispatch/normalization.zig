const Device = @import("../../types/tensor/device.zig").Device;
const DType = @import("../../types/tensor/dtype.zig").DType;
const Storage = @import("../../types/tensor/storage.zig").Storage;
const Value = @import("../../types/tensor/value.zig").Value;
const kernel_dispatch = @import("../../backend/dispatch.zig");

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
