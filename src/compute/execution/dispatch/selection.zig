const std = @import("std");
const Device = @import("../../shared/types/tensor/device.zig").Device;
const DType = @import("../../shared/types/tensor/dtype.zig").DType;
const Storage = @import("../../shared/types/tensor/storage.zig").Storage;
const Tensor = @import("../../shared/types/tensor/tensor.zig").Tensor;
const kernel_dispatch = @import("../../backend/dispatch.zig");

pub fn dispatchGather(
    device: Device,
    dtype: DType,
    input: *const Tensor,
    index: *const Tensor,
    output: *Storage,
    axis: usize,
) !void {
    try kernel_dispatch.gather(
        device,
        dtype,
        input.storage orelse return error.InputNotMaterialized,
        index.storage orelse return error.InputNotMaterialized,
        output,
        input.shape.dims,
        index.shape.dims,
        axis,
    );
}

pub fn dispatchEmbedding(
    device: Device,
    dtype: DType,
    table: *const Tensor,
    index: *const Tensor,
    output: *Storage,
) !void {
    try kernel_dispatch.embedding(
        device,
        dtype,
        table.storage orelse return error.InputNotMaterialized,
        index.storage orelse return error.InputNotMaterialized,
        output,
        table.shape.dims,
        index.shape.dims,
    );
}

pub fn dispatchIndexSelect(
    device: Device,
    dtype: DType,
    input: *const Tensor,
    index: *const Tensor,
    output: *Storage,
    axis: usize,
) !void {
    try kernel_dispatch.indexSelect(
        device,
        dtype,
        input.storage orelse return error.InputNotMaterialized,
        index.storage orelse return error.InputNotMaterialized,
        output,
        input.shape.dims,
        axis,
        index.shape.numel(),
    );
}

pub fn dispatchScatterAdd(
    device: Device,
    dtype: DType,
    index_dtype: DType,
    base: *const Tensor,
    index: *const Tensor,
    updates: *const Tensor,
    output: *Storage,
    axis: usize,
) !void {
    try kernel_dispatch.scatterAdd(
        device,
        dtype,
        index_dtype,
        base.storage orelse return error.InputNotMaterialized,
        index.storage orelse return error.InputNotMaterialized,
        updates.storage orelse return error.InputNotMaterialized,
        output,
        base.shape.dims,
        axis,
    );
}

pub fn dispatchOneHot(
    device: Device,
    index: *const Tensor,
    output: *Storage,
    num_classes: usize,
) !void {
    try kernel_dispatch.oneHot(
        device,
        index.storage orelse return error.InputNotMaterialized,
        output,
        num_classes,
    );
}

pub fn dispatchTopK(
    allocator: std.mem.Allocator,
    device: Device,
    dtype: DType,
    input: *const Tensor,
    values_output: *Storage,
    indices_output: *Storage,
    axis: usize,
    k: usize,
    largest: bool,
) !void {
    try kernel_dispatch.topk(
        allocator,
        device,
        dtype,
        input.storage orelse return error.InputNotMaterialized,
        values_output,
        indices_output,
        input.shape.dims,
        axis,
        k,
        largest,
    );
}
