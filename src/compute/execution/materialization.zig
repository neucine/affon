const std = @import("std");
const tensor_value = @import("../shared/types/tensor/tensor.zig");
const Tensor = tensor_value.Tensor;
const Shape = @import("../shared/types/tensor/shape.zig").Shape;
const Layout = @import("../shared/types/tensor/layout.zig").Layout;
const Storage = @import("../shared/types/tensor/storage.zig").Storage;
const TensorSpec = @import("../shared/types/tensor/tensor_spec.zig").TensorSpec;
const kernel_dispatch = @import("../backend/dispatch.zig");
const transfer_execution = @import("transfer.zig");

pub const MaterializationSummary = struct {
    contiguity_fixup_count: usize = 0,
    contiguity_fixup_bytes: usize = 0,
    transfer: transfer_execution.TransferSummary = .{},
};

pub fn contiguousInto(
    input: *const Tensor,
    output: *Tensor,
) !MaterializationSummary {
    const device = input.device() orelse return error.InputNotMaterialized;
    const byte_len = output.shape.numel() * output.dtype.size();
    try kernel_dispatch.contiguous(
        device,
        input.dtype,
        try input.requireRuntimeBacking(),
        try output.requireRuntimeBacking(),
        input.shape.dims,
        input.layout.strides,
        input.layout.offset,
    );
    return .{
        .contiguity_fixup_count = 1,
        .contiguity_fixup_bytes = byte_len,
    };
}

pub fn materializeContiguousValue(
    allocator: std.mem.Allocator,
    input: *const Tensor,
    spec: TensorSpec,
    source: Storage.Source,
) !struct { value: *Tensor, summary: MaterializationSummary } {
    const output = try Tensor.createContiguousWithSource(
        allocator,
        input.shape.dims,
        input.dtype,
        input.device() orelse return error.InputNotMaterialized,
        false,
        source,
    );
    errdefer output.deinit();
    try output.setAxesCopy(spec.axes);
    const summary = try contiguousInto(input, output);
    return .{ .value = output, .summary = summary };
}

pub fn materializePackedDenseStorage(
    allocator: std.mem.Allocator,
    value: *const Tensor,
    source: Storage.Source,
) !*Storage {
    const byte_len = value.shape.numel() * value.dtype.size();
    const out = switch (value.device() orelse return error.InputNotMaterialized) {
        .cpu => try Storage.createCpuWithMetadata(allocator, byte_len, false, .{
            .reason = .temporary,
            .source = source,
        }),
        .metal => try Storage.createMetalWithMetadata(allocator, byte_len, .{
            .policy = .pooled,
            .reason = .temporary,
            .source = source,
        }),
    };
    errdefer out.release();

    try kernel_dispatch.contiguous(
        value.device() orelse return error.InputNotMaterialized,
        value.dtype,
        try value.requireRuntimeBacking(),
        out,
        value.shape.dims,
        value.layout.strides,
        value.layout.offset,
    );
    return out;
}

pub fn materializePackedDenseValue(
    allocator: std.mem.Allocator,
    value: *const Tensor,
    source: Storage.Source,
) !*Tensor {
    const storage = try materializePackedDenseStorage(allocator, value, source);
    errdefer storage.release();

    var shape = try Shape.initCopy(allocator, value.shape.dims);
    errdefer shape.deinit();
    var layout = try Layout.initContiguous(allocator, shape);
    errdefer layout.deinit();

    const packed_value = try allocator.create(Tensor);
    errdefer allocator.destroy(packed_value);
    packed_value.* = .{
        .allocator = allocator,
        .shape = shape,
        .dtype = value.dtype,
        .layout = layout,
        .storage = storage,
        .axes = try tensor_value.cloneAxes(allocator, value.axes),
    };
    return packed_value;
}
