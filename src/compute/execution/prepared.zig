const std = @import("std");
const tensor_value = @import("../shared/types/tensor/tensor.zig");
const Tensor = tensor_value.Tensor;
const Shape = @import("../shared/types/tensor/shape.zig").Shape;
const Layout = @import("../shared/types/tensor/layout.zig").Layout;
const Storage = @import("../shared/types/tensor/storage.zig").Storage;
const TensorSpec = @import("../shared/types/tensor/tensor_spec.zig").TensorSpec;
const ir_plan = @import("../shared/types/ir/plan.zig");
const execution_layout = @import("../plan/layout.zig");
const ExecutionMetadata = @import("../shared/types/operation/execution_metadata.zig").ExecutionMetadata;
const materialization_execution = @import("materialization.zig");

pub const BinaryElementwiseDescriptor = union(enum) {
    dense,
    broadcast: ir_plan.BinaryBroadcastSpec,
};

pub const MaskedFillDescriptor = union(enum) {
    dense,
    broadcast: ir_plan.MaskedFillBroadcastSpec,
};

pub const WhereDescriptor = union(enum) {
    dense,
    broadcast: ir_plan.WhereBroadcastSpec,
};

pub const MatmulProjectionDescriptor = struct {
    lhs_shape: [2]usize,
    lhs_strides: [2]isize,

    pub fn lhsShape(self: *const MatmulProjectionDescriptor) []const usize {
        return self.lhs_shape[0..];
    }

    pub fn lhsLayout(self: *const MatmulProjectionDescriptor) Layout {
        return .{
            .strides = @constCast(self.lhs_strides[0..]),
            .offset = 0,
            .allocator = undefined,
        };
    }
};

pub const PreparedInputValue = struct {
    value: *const Tensor,
    owned_value: ?*Tensor = null,
    materialized_packed_dense: bool = false,

    pub fn deinit(self: PreparedInputValue) void {
        if (self.owned_value) |owned| owned.deinit();
    }
};

pub fn isPackedDenseInput(value: *const Tensor) bool {
    return value.layout.offset == 0 and value.layout.isContiguous(value.shape);
}

pub fn prepareInputValue(
    allocator: std.mem.Allocator,
    value: *const Tensor,
    decision: execution_layout.InputLayoutDecision,
    source: Storage.Source,
) !PreparedInputValue {
    return switch (decision) {
        .accept => .{ .value = value },
        .pack_to_dense => {
            if (isPackedDenseInput(value)) {
                return .{ .value = value };
            }
            const packed_value = try materialization_execution.materializePackedDenseValue(allocator, value, source);
            return .{
                .value = packed_value,
                .owned_value = packed_value,
                .materialized_packed_dense = true,
            };
        },
        .unsupported => error.ExecutionNotImplemented,
    };
}

pub fn createOutputValue(
    allocator: std.mem.Allocator,
    spec: TensorSpec,
    source: Storage.Source,
) !*Tensor {
    const output = try Tensor.createContiguousWithSource(
        allocator,
        spec.shape.dims,
        spec.dtype,
        spec.device,
        false,
        source,
    );
    errdefer output.deinit();
    try output.setAxesCopy(spec.axes);
    return output;
}

pub fn createViewValue(
    allocator: std.mem.Allocator,
    input: *const Tensor,
    spec: TensorSpec,
) !*Tensor {
    const storage = try input.requireRuntimeBacking();
    storage.retain();
    errdefer storage.release();

    var shape = try Shape.initCopy(allocator, spec.shape.dims);
    errdefer shape.deinit();
    var layout = try Layout.initCopy(allocator, spec.layout.strides, spec.layout.offset);
    errdefer layout.deinit();

    const view = try allocator.create(Tensor);
    errdefer allocator.destroy(view);
    view.* = .{
        .allocator = allocator,
        .shape = shape,
        .dtype = spec.dtype,
        .layout = layout,
        .storage = storage,
        .axes = try tensor_value.cloneAxes(allocator, spec.axes),
    };
    return view;
}

pub fn binaryElementwiseDescriptor(
    lhs: *const Tensor,
    rhs: *const Tensor,
    broadcast: ?ir_plan.BroadcastSpec,
) !BinaryElementwiseDescriptor {
    if (Shape.eql(lhs.shape, rhs.shape) and hasExactPackedStorage(lhs) and hasExactPackedStorage(rhs)) {
        return .dense;
    }

    return switch (broadcast orelse return error.InvalidExecutionPlan) {
        .binary => |desc| .{ .broadcast = desc },
        else => error.InvalidExecutionPlan,
    };
}

pub fn maskedFillDescriptor(
    input: *const Tensor,
    mask: *const Tensor,
    broadcast: ?ir_plan.BroadcastSpec,
) !MaskedFillDescriptor {
    const lowered = switch (broadcast orelse return error.InvalidExecutionPlan) {
        .masked_fill => |desc| desc,
        else => return error.InvalidExecutionPlan,
    };
    const out_shape = lowered.shape[0..lowered.rank];
    if (std.mem.eql(usize, input.shape.dims, out_shape) and
        std.mem.eql(usize, mask.shape.dims, out_shape) and
        hasExactPackedStorage(input) and
        hasExactPackedStorage(mask))
    {
        return .dense;
    }
    return .{ .broadcast = lowered };
}

pub fn whereDescriptor(
    cond: *const Tensor,
    on_true: *const Tensor,
    on_false: *const Tensor,
    broadcast: ?ir_plan.BroadcastSpec,
) !WhereDescriptor {
    const lowered = switch (broadcast orelse return error.InvalidExecutionPlan) {
        .where => |desc| desc,
        else => return error.InvalidExecutionPlan,
    };
    const out_shape = lowered.shape[0..lowered.rank];
    if (std.mem.eql(usize, cond.shape.dims, out_shape) and
        std.mem.eql(usize, on_true.shape.dims, out_shape) and
        std.mem.eql(usize, on_false.shape.dims, out_shape) and
        hasExactPackedStorage(cond) and
        hasExactPackedStorage(on_true) and
        hasExactPackedStorage(on_false))
    {
        return .dense;
    }
    return .{ .broadcast = lowered };
}

pub fn matmulProjectionDescriptor(
    lhs: *const Tensor,
    rhs: *const Tensor,
    enabled: bool,
) !?MatmulProjectionDescriptor {
    if (!enabled) return null;
    if (!lhs.layout.isContiguous(lhs.shape) or lhs.layout.offset != 0) return null;
    if (rhs.shape.rank() != 2) return null;

    var leading: usize = 1;
    for (lhs.shape.dims[0 .. lhs.shape.rank() - 1]) |dim| leading *= dim;
    const k = lhs.shape.dims[lhs.shape.rank() - 1];
    return .{
        .lhs_shape = .{ leading, k },
        .lhs_strides = .{ @as(isize, @intCast(k)), 1 },
    };
}

fn hasExactPackedStorage(value: *const Tensor) bool {
    const storage = value.runtimeBacking() orelse return false;
    return isPackedDenseInput(value) and
        storage.bytes == value.shape.numel() * value.dtype.size();
}

fn makeOffsetViewForTest(
    allocator: std.mem.Allocator,
    base: *const Tensor,
    dims: []const usize,
    strides: []const isize,
    offset: usize,
) !*Tensor {
    const storage = try base.requireRuntimeBacking();
    storage.retain();
    errdefer storage.release();

    var shape = try Shape.initCopy(allocator, dims);
    errdefer shape.deinit();
    var layout = try Layout.initCopy(allocator, strides, offset);
    errdefer layout.deinit();

    const view = try allocator.create(Tensor);
    errdefer allocator.destroy(view);
    view.* = .{
        .allocator = allocator,
        .shape = shape,
        .dtype = base.dtype,
        .layout = layout,
        .storage = storage,
        .axes = null,
    };
    return view;
}

test "binary descriptor uses dense path for same-shape contiguous inputs" {
    const allocator = std.testing.allocator;
    const lhs = try Tensor.fromSliceF32(allocator, &.{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer lhs.deinit();
    const rhs = try Tensor.fromSliceF32(allocator, &.{ 2, 2 }, &.{ 5, 6, 7, 8 });
    defer rhs.deinit();

    const broadcast = try @import("prepared_test_support.zig").broadcast(allocator, .add, &.{ lhs, rhs }, .{ .none = {} });
    const descriptor = try binaryElementwiseDescriptor(lhs, rhs, broadcast);
    try std.testing.expectEqual(@as(std.meta.Tag(BinaryElementwiseDescriptor), .dense), std.meta.activeTag(descriptor));
}

test "binary descriptor re-infers broadcast after input preparation" {
    const allocator = std.testing.allocator;
    const lhs = try Tensor.fromSliceF32(allocator, &.{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer lhs.deinit();
    const rhs = try Tensor.fromSliceF32(allocator, &.{3}, &.{ 10, 20, 30 });
    defer rhs.deinit();

    const planned_broadcast = try @import("prepared_test_support.zig").broadcast(allocator, .add, &.{ lhs, rhs }, .{ .none = {} });
    const descriptor = try binaryElementwiseDescriptor(lhs, rhs, planned_broadcast);
    const broadcast = switch (descriptor) {
        .broadcast => |desc| desc,
        .dense => return error.TestExpectedBroadcast,
    };
    try std.testing.expectEqual(@as(usize, 2), broadcast.rank);
    try std.testing.expectEqualSlices(usize, &.{ 2, 3 }, broadcast.shape[0..broadcast.rank]);
    try std.testing.expectEqualSlices(isize, &.{ 3, 1 }, broadcast.lhs_strides[0..broadcast.rank]);
    try std.testing.expectEqualSlices(isize, &.{ 0, 1 }, broadcast.rhs_strides[0..broadcast.rank]);
}

test "binary descriptor treats offset views as broadcast, not dense" {
    const allocator = std.testing.allocator;
    const lhs_base = try Tensor.fromSliceF32(allocator, &.{3}, &.{ 1, 2, 3 });
    defer lhs_base.deinit();
    const lhs = try makeOffsetViewForTest(allocator, lhs_base, &.{2}, &.{1}, 1);
    defer lhs.deinit();
    const rhs = try Tensor.fromSliceF32(allocator, &.{2}, &.{ 10, 20 });
    defer rhs.deinit();

    const broadcast = try @import("prepared_test_support.zig").broadcast(allocator, .add, &.{ lhs, rhs }, .{ .none = {} });
    const descriptor = try binaryElementwiseDescriptor(lhs, rhs, broadcast);
    try std.testing.expectEqual(@as(std.meta.Tag(BinaryElementwiseDescriptor), .broadcast), std.meta.activeTag(descriptor));
}

test "packed dense predicate rejects contiguous offset views" {
    const allocator = std.testing.allocator;
    const base = try Tensor.fromSliceF32(allocator, &.{3}, &.{ 1, 2, 3 });
    defer base.deinit();
    const view = try makeOffsetViewForTest(allocator, base, &.{2}, &.{1}, 1);
    defer view.deinit();

    try std.testing.expect(!isPackedDenseInput(view));
}

test "packed dense materialization preserves logical view order" {
    const allocator = std.testing.allocator;
    const base = try Tensor.fromSliceF32(allocator, &.{6}, &.{ 10, 20, 30, 40, 50, 60 });
    defer base.deinit();
    const view = try makeOffsetViewForTest(allocator, base, &.{3}, &.{1}, 2);
    defer view.deinit();

    const packed_value = try materialization_execution.materializePackedDenseValue(allocator, view, .graph);
    defer packed_value.deinit();

    try std.testing.expect(isPackedDenseInput(packed_value));
    const bytes = try (try packed_value.requireRuntimeBacking()).readableBytes();
    const values = std.mem.bytesAsSlice(f32, bytes);
    try std.testing.expectEqualSlices(f32, &.{ 30, 40, 50 }, values);
}

test "masked_fill descriptor uses dense path for same-shape contiguous inputs" {
    const allocator = std.testing.allocator;
    const input = try Tensor.fromSliceF32(allocator, &.{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer input.deinit();
    const mask = try Tensor.fromSliceI64(allocator, &.{ 2, 2 }, &.{ 1, 0, 1, 0 });
    defer mask.deinit();

    const broadcast = try @import("prepared_test_support.zig").broadcast(allocator, .masked_fill, &.{ input, mask }, .{ .masked_fill = .{ .value = -9.0 } });
    const descriptor = try maskedFillDescriptor(input, mask, broadcast);
    try std.testing.expectEqual(@as(std.meta.Tag(MaskedFillDescriptor), .dense), std.meta.activeTag(descriptor));
}

test "masked_fill descriptor re-infers broadcast after input preparation" {
    const allocator = std.testing.allocator;
    const input = try Tensor.fromSliceF32(allocator, &.{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer input.deinit();
    const mask = try Tensor.fromSliceI64(allocator, &.{3}, &.{ 1, 0, 1 });
    defer mask.deinit();

    const planned_broadcast = try @import("prepared_test_support.zig").broadcast(allocator, .masked_fill, &.{ input, mask }, .{ .masked_fill = .{ .value = -9.0 } });
    const descriptor = try maskedFillDescriptor(input, mask, planned_broadcast);
    const broadcast = switch (descriptor) {
        .broadcast => |desc| desc,
        .dense => return error.TestExpectedBroadcast,
    };
    try std.testing.expectEqual(@as(usize, 2), broadcast.rank);
    try std.testing.expectEqualSlices(usize, &.{ 2, 3 }, broadcast.shape[0..broadcast.rank]);
    try std.testing.expectEqualSlices(isize, &.{ 3, 1 }, broadcast.input_strides[0..broadcast.rank]);
    try std.testing.expectEqualSlices(isize, &.{ 0, 1 }, broadcast.mask_strides[0..broadcast.rank]);
}

test "masked_fill descriptor treats offset views as broadcast, not dense" {
    const allocator = std.testing.allocator;
    const input_base = try Tensor.fromSliceF32(allocator, &.{3}, &.{ 1, 2, 3 });
    defer input_base.deinit();
    const input = try makeOffsetViewForTest(allocator, input_base, &.{2}, &.{1}, 1);
    defer input.deinit();
    const mask = try Tensor.fromSliceI64(allocator, &.{2}, &.{ 1, 0 });
    defer mask.deinit();

    const broadcast = try @import("prepared_test_support.zig").broadcast(allocator, .masked_fill, &.{ input, mask }, .{ .masked_fill = .{ .value = -9.0 } });
    const descriptor = try maskedFillDescriptor(input, mask, broadcast);
    try std.testing.expectEqual(@as(std.meta.Tag(MaskedFillDescriptor), .broadcast), std.meta.activeTag(descriptor));
}

test "where descriptor uses dense path for same-shape contiguous inputs" {
    const allocator = std.testing.allocator;
    const cond = try Tensor.fromSliceI64(allocator, &.{ 2, 2 }, &.{ 1, 0, 1, 0 });
    defer cond.deinit();
    const on_true = try Tensor.fromSliceF32(allocator, &.{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer on_true.deinit();
    const on_false = try Tensor.fromSliceF32(allocator, &.{ 2, 2 }, &.{ 5, 6, 7, 8 });
    defer on_false.deinit();

    const broadcast = try @import("prepared_test_support.zig").broadcast(allocator, .where, &.{ cond, on_true, on_false }, .{ .none = {} });
    const descriptor = try whereDescriptor(cond, on_true, on_false, broadcast);
    try std.testing.expectEqual(@as(std.meta.Tag(WhereDescriptor), .dense), std.meta.activeTag(descriptor));
}

test "where descriptor re-infers broadcast after input preparation" {
    const allocator = std.testing.allocator;
    const cond = try Tensor.fromSliceI64(allocator, &.{ 2, 1 }, &.{ 1, 0 });
    defer cond.deinit();
    const on_true = try Tensor.fromSliceF32(allocator, &.{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer on_true.deinit();
    const on_false = try Tensor.fromSliceF32(allocator, &.{3}, &.{ 10, 20, 30 });
    defer on_false.deinit();

    const planned_broadcast = try @import("prepared_test_support.zig").broadcast(allocator, .where, &.{ cond, on_true, on_false }, .{ .none = {} });
    const descriptor = try whereDescriptor(cond, on_true, on_false, planned_broadcast);
    const broadcast = switch (descriptor) {
        .broadcast => |desc| desc,
        .dense => return error.TestExpectedBroadcast,
    };
    try std.testing.expectEqual(@as(usize, 2), broadcast.rank);
    try std.testing.expectEqualSlices(usize, &.{ 2, 3 }, broadcast.shape[0..broadcast.rank]);
    try std.testing.expectEqualSlices(isize, &.{ 1, 0 }, broadcast.cond_strides[0..broadcast.rank]);
    try std.testing.expectEqualSlices(isize, &.{ 3, 1 }, broadcast.on_true_strides[0..broadcast.rank]);
    try std.testing.expectEqualSlices(isize, &.{ 0, 1 }, broadcast.on_false_strides[0..broadcast.rank]);
}

test "where descriptor treats offset views as broadcast, not dense" {
    const allocator = std.testing.allocator;
    const cond = try Tensor.fromSliceI64(allocator, &.{2}, &.{ 1, 0 });
    defer cond.deinit();
    const true_base = try Tensor.fromSliceF32(allocator, &.{3}, &.{ 1, 2, 3 });
    defer true_base.deinit();
    const on_true = try makeOffsetViewForTest(allocator, true_base, &.{2}, &.{1}, 1);
    defer on_true.deinit();
    const on_false = try Tensor.fromSliceF32(allocator, &.{2}, &.{ 5, 6 });
    defer on_false.deinit();

    const broadcast = try @import("prepared_test_support.zig").broadcast(allocator, .where, &.{ cond, on_true, on_false }, .{ .none = {} });
    const descriptor = try whereDescriptor(cond, on_true, on_false, broadcast);
    try std.testing.expectEqual(@as(std.meta.Tag(WhereDescriptor), .broadcast), std.meta.activeTag(descriptor));
}

test "matmul projection descriptor flattens contiguous leading batch" {
    const allocator = std.testing.allocator;
    const lhs = try Tensor.createContiguous(allocator, &.{ 2, 3, 4 }, .f32, .cpu, false);
    defer lhs.deinit();
    const rhs = try Tensor.createContiguous(allocator, &.{ 4, 5 }, .f32, .cpu, false);
    defer rhs.deinit();

    const descriptor = (try matmulProjectionDescriptor(lhs, rhs, true)) orelse return error.TestExpectedProjectionDescriptor;
    try std.testing.expectEqualSlices(usize, &.{ 6, 4 }, descriptor.lhsShape());
    try std.testing.expectEqualSlices(isize, &.{ 4, 1 }, descriptor.lhsLayout().strides);
}

test "matmul projection descriptor rejects non-contiguous lhs view" {
    const allocator = std.testing.allocator;
    const lhs = try Tensor.createContiguous(allocator, &.{ 2, 3, 4 }, .f32, .cpu, false);
    defer lhs.deinit();
    lhs.layout.strides[0] = 13;
    const rhs = try Tensor.createContiguous(allocator, &.{ 4, 5 }, .f32, .cpu, false);
    defer rhs.deinit();

    try std.testing.expectEqual(@as(?MatmulProjectionDescriptor, null), try matmulProjectionDescriptor(lhs, rhs, true));
}
