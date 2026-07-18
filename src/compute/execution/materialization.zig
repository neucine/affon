const std = @import("std");
const tensor_value = @import("../types/tensor/tensor.zig");
const Tensor = tensor_value.Tensor;
const Shape = @import("../types/tensor/shape.zig").Shape;
const Layout = @import("../types/tensor/layout.zig").Layout;
const Storage = @import("../types/tensor/storage.zig").Storage;
const TensorSpec = @import("../types/tensor/tensor_spec.zig").TensorSpec;
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
    const src = try value.requireRuntimeBacking();
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

    const staging_in = try allocator.alignedAlloc(u8, .@"8", src.bytes);
    defer allocator.free(staging_in);
    const staging_out = try allocator.alignedAlloc(u8, .@"8", byte_len);
    defer allocator.free(staging_out);

    try src.copyToHost(staging_in);
    switch (value.dtype) {
        .f32 => remapPackedDense(f32, staging_in, staging_out, value.shape.dims, value.layout.strides, value.layout.offset),
        .f64 => remapPackedDense(f64, staging_in, staging_out, value.shape.dims, value.layout.strides, value.layout.offset),
        .i64 => remapPackedDense(i64, staging_in, staging_out, value.shape.dims, value.layout.strides, value.layout.offset),
    }
    try out.writeFromHost(staging_out);
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

fn remapPackedDense(
    comptime T: type,
    src_bytes: []const u8,
    dst_bytes: []u8,
    shape: []const usize,
    strides: []const isize,
    offset: usize,
) void {
    const src = std.mem.bytesAsSlice(T, src_bytes);
    const dst = std.mem.bytesAsSlice(T, dst_bytes);
    var idx: [8]usize = [_]usize{0} ** 8;
    for (0..dst.len) |flat| {
        var src_index: isize = @intCast(offset);
        for (0..shape.len) |d| src_index += @as(isize, @intCast(idx[d])) * strides[d];
        dst[flat] = src[@intCast(src_index)];
        var dim = shape.len;
        while (dim > 0) {
            dim -= 1;
            idx[dim] += 1;
            if (idx[dim] < shape[dim]) break;
            idx[dim] = 0;
            if (dim == 0) break;
        }
    }
}
