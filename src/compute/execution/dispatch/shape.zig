const std = @import("std");
const Device = @import("../../types/tensor/device.zig").Device;
const DType = @import("../../types/tensor/dtype.zig").DType;
const Storage = @import("../../types/tensor/storage.zig").Storage;
const Value = @import("../../types/tensor/value.zig").Value;
const SliceRange = @import("../../types/operation/options.zig").SliceRange;
const kernel_dispatch = @import("../../backend/dispatch.zig");
const prepared_execution = @import("../prepared.zig");

pub fn dispatchCat(
    allocator: std.mem.Allocator,
    device: Device,
    dtype: DType,
    inputs: []const prepared_execution.PreparedInputValue,
    output: *Storage,
    axis: usize,
) !void {
    if (inputs.len == 0) return error.InvalidExecutionPlan;
    const storages = try collectStorages(allocator, inputs);
    defer allocator.free(storages);
    try kernel_dispatch.cat(
        device,
        dtype,
        storages,
        output,
        inputs[0].value.shape.dims,
        axis,
    );
}

pub fn dispatchStack(
    allocator: std.mem.Allocator,
    device: Device,
    dtype: DType,
    inputs: []const prepared_execution.PreparedInputValue,
    output: *Storage,
    axis: usize,
) !void {
    if (inputs.len == 0) return error.InvalidExecutionPlan;
    const storages = try collectStorages(allocator, inputs);
    defer allocator.free(storages);
    try kernel_dispatch.stack(
        device,
        dtype,
        storages,
        output,
        inputs[0].value.shape.dims,
        axis,
    );
}

pub fn dispatchSlice(
    device: Device,
    dtype: DType,
    input: *const Value,
    output: *Storage,
    ranges: []const SliceRange,
) !void {
    try kernel_dispatch.slice(
        device,
        dtype,
        input.storage orelse return error.InputNotMaterialized,
        output,
        input.shape.dims,
        ranges,
    );
}

fn collectStorages(
    allocator: std.mem.Allocator,
    inputs: []const prepared_execution.PreparedInputValue,
) ![]*const Storage {
    const storages = try allocator.alloc(*const Storage, inputs.len);
    errdefer allocator.free(storages);
    for (inputs, 0..) |prepared, i| {
        storages[i] = prepared.value.storage orelse return error.InputNotMaterialized;
    }
    return storages;
}
