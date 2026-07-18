const Device = @import("../../types/tensor/device.zig").Device;
const DType = @import("../../types/tensor/dtype.zig").DType;
const Storage = @import("../../types/tensor/storage.zig").Storage;
const Tensor = @import("../../types/tensor/tensor.zig").Tensor;
const kernel_capability = @import("../../backend/capability.zig");
const kernel_dispatch = @import("../../backend/dispatch.zig");
const prepared_execution = @import("../prepared.zig");

pub fn dispatchDot(
    device: Device,
    dtype: DType,
    lhs: *const Tensor,
    rhs: *const Tensor,
    output: *Storage,
) !void {
    try kernel_dispatch.dot(
        device,
        dtype,
        lhs.storage orelse return error.InputNotMaterialized,
        rhs.storage orelse return error.InputNotMaterialized,
        output,
    );
}

pub fn dispatchMatmulWithLayouts(
    device: Device,
    dtype: DType,
    lhs: *const Tensor,
    rhs: *const Tensor,
    output: *Storage,
) !void {
    if (!kernel_capability.matmulAcceptsLayouts(
        device,
        dtype,
        lhs.shape.dims,
        lhs.layout,
        rhs.shape.dims,
        rhs.layout,
    )) return error.ExecutionNotImplemented;

    try kernel_dispatch.matmulWithLayouts(
        device,
        dtype,
        lhs.storage orelse return error.InputNotMaterialized,
        rhs.storage orelse return error.InputNotMaterialized,
        output,
        lhs.shape.dims,
        lhs.layout,
        rhs.shape.dims,
        rhs.layout,
    );
}

pub fn dispatchProjectionMatmulFastPath(
    device: Device,
    dtype: DType,
    lhs: *const Tensor,
    rhs: *const Tensor,
    projection_enabled: bool,
    output: *Storage,
) !bool {
    const projection = (try prepared_execution.matmulProjectionDescriptor(lhs, rhs, projection_enabled)) orelse return false;
    const flat_layout = projection.lhsLayout();

    if (!kernel_capability.matmulAcceptsLayouts(
        device,
        dtype,
        projection.lhsShape(),
        flat_layout,
        rhs.shape.dims,
        rhs.layout,
    )) return false;

    try kernel_dispatch.matmulWithLayouts(
        device,
        dtype,
        lhs.storage orelse return error.InputNotMaterialized,
        rhs.storage orelse return error.InputNotMaterialized,
        output,
        projection.lhsShape(),
        flat_layout,
        rhs.shape.dims,
        rhs.layout,
    );
    return true;
}
