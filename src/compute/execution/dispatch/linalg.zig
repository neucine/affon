const Device = @import("../../tensor/device.zig").Device;
const DType = @import("../../tensor/dtype.zig").DType;
const Storage = @import("../../tensor/storage.zig").Storage;
const Value = @import("../../tensor/value.zig").Value;
const ValueSpec = @import("../../tensor/value_spec.zig").ValueSpec;
const ExecutionMetadata = @import("../../op/execution_metadata.zig").ExecutionMetadata;
const kernel_capability = @import("../../kernel/capability.zig");
const kernel_dispatch = @import("../../kernel/dispatch.zig");
const prepared_execution = @import("../prepared.zig");

pub fn dispatchDot(
    device: Device,
    dtype: DType,
    lhs: *const Value,
    rhs: *const Value,
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
    lhs: *const Value,
    rhs: *const Value,
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
    lhs: *const Value,
    rhs: *const Value,
    output_spec: ValueSpec,
    metadata: ExecutionMetadata,
    output: *Storage,
) !bool {
    const projection = (try prepared_execution.matmulProjectionDescriptor(lhs, rhs, output_spec, metadata)) orelse return false;
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
