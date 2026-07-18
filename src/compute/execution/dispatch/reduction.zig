const std = @import("std");
const Device = @import("../../types/tensor/device.zig").Device;
const DType = @import("../../types/tensor/dtype.zig").DType;
const Value = @import("../../types/tensor/value.zig").Value;
const Shape = @import("../../types/tensor/shape.zig").Shape;
const Layout = @import("../../types/tensor/layout.zig").Layout;
const Storage = @import("../../types/tensor/storage.zig").Storage;
const kernel_dispatch = @import("../../backend/dispatch.zig");
const semantic = @import("../../sema/index.zig");
const OpTag = @import("../../types/operation/tag.zig").OpTag;
const transfer_execution = @import("../transfer.zig");

pub fn dispatchAll(
    device: Device,
    tag: OpTag,
    dtype: DType,
    input: *const Value,
    output: *Storage,
) !void {
    try kernel_dispatch.reductionAll(
        device,
        tag,
        dtype,
        input.storage orelse return error.InputNotMaterialized,
        output,
    );
}

pub fn dispatchAxis(
    device: Device,
    tag: OpTag,
    dtype: DType,
    input: *const Value,
    output: *Storage,
    axis: usize,
    keepdim: bool,
) !void {
    try kernel_dispatch.reductionAxis(
        device,
        tag,
        dtype,
        input.storage orelse return error.InputNotMaterialized,
        output,
        input.shape.dims,
        axis,
        keepdim,
    );
}

pub fn dispatchSoftmax(
    device: Device,
    dtype: DType,
    input: *const Value,
    output: *Storage,
    axis: usize,
) !void {
    try kernel_dispatch.softmax(
        device,
        dtype,
        input.storage orelse return error.InputNotMaterialized,
        output,
        input.shape.dims,
        input.layout,
        axis,
    );
}

pub fn dispatchLogSoftmax(
    allocator: std.mem.Allocator,
    device: Device,
    dtype: DType,
    input: *const Value,
    output: *Storage,
    axis: usize,
) !void {
    try kernel_dispatch.logSoftmax(
        allocator,
        device,
        dtype,
        input.storage orelse return error.InputNotMaterialized,
        output,
        input.shape.dims,
        input.layout,
        axis,
    );
}

pub fn dispatchLogSoftmaxNll(
    device: Device,
    dtype: DType,
    logits: *const Value,
    targets: *const Value,
    output: *Storage,
    axis: usize,
) !void {
    try kernel_dispatch.logSoftmaxNll(
        device,
        dtype,
        logits.storage orelse return error.InputNotMaterialized,
        targets.storage orelse return error.InputNotMaterialized,
        output,
        logits.shape.dims,
        axis,
    );
}

pub fn dispatchCrossEntropyIndexed(
    device: Device,
    dtype: DType,
    logits: *const Value,
    targets: *const Value,
    output: *Storage,
) !void {
    try kernel_dispatch.crossEntropyIndexed(
        device,
        dtype,
        logits.storage orelse return error.InputNotMaterialized,
        targets.storage orelse return error.InputNotMaterialized,
        output,
        logits.shape.dims[0],
        logits.shape.dims[1],
    );
}

pub fn dispatchCrossEntropyIndexedBackward(
    device: Device,
    dtype: DType,
    logits: *const Value,
    targets: *const Value,
    grad_out: *const Value,
    output: *Storage,
) !void {
    try kernel_dispatch.crossEntropyIndexedBackward(
        device,
        dtype,
        logits.storage orelse return error.InputNotMaterialized,
        targets.storage orelse return error.InputNotMaterialized,
        grad_out.storage orelse return error.InputNotMaterialized,
        output,
        logits.shape.dims[0],
        logits.shape.dims[1],
    );
}

pub fn reduceToShape(
    allocator: std.mem.Allocator,
    input: *const Value,
    reduce: semantic.ReduceToShapeSpec,
    output: *Value,
    source: Storage.Source,
) !transfer_execution.TransferSummary {
    const input_storage = input.storage orelse return error.InputNotMaterialized;

    var current = input_storage;
    var own_current = false;
    defer if (own_current) current.release();

    var tmp_shape = try allocator.alloc(usize, input.shape.rank());
    defer allocator.free(tmp_shape);
    @memcpy(tmp_shape, input.shape.dims);

    const device = input.device() orelse return error.InputNotMaterialized;
    for (reduce.axes[0..reduce.count]) |axis| {
        var out_dims = try allocator.alloc(usize, tmp_shape.len);
        defer allocator.free(out_dims);
        @memcpy(out_dims, tmp_shape);
        out_dims[axis] = 1;
        var elems: usize = 1;
        for (out_dims) |d| elems *= d;
        const reduced = switch (device) {
            .cpu => try Storage.createCpuWithMetadata(allocator, elems * input.dtype.size(), false, .{
                .reason = .op_output,
                .source = source,
            }),
            .metal => try Storage.createMetalWithMetadata(allocator, elems * input.dtype.size(), .{
                .policy = .pooled,
                .reason = .op_output,
                .source = source,
            }),
        };
        errdefer reduced.release();
        try kernel_dispatch.reductionAxis(device, .sum_axis, input.dtype, current, reduced, tmp_shape, axis, true);
        if (own_current) current.release();
        current = reduced;
        own_current = true;
        tmp_shape[axis] = 1;
    }

    const current_value = try allocator.create(Value);
    defer allocator.destroy(current_value);
    var shape = try Shape.initCopy(allocator, output.shape.dims);
    defer shape.deinit();
    var layout = try Layout.initContiguous(allocator, shape);
    defer layout.deinit();
    current_value.* = .{
        .allocator = allocator,
        .shape = shape,
        .dtype = input.dtype,
        .layout = layout,
        .storage = current,
        .axes = null,
    };
    return transfer_execution.copyValueStorageInto(allocator, current_value, output);
}
