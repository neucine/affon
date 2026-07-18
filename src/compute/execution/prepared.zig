const std = @import("std");
const tensor_value = @import("../types/tensor/value.zig");
const Value = tensor_value.Value;
const Shape = @import("../types/tensor/shape.zig").Shape;
const Layout = @import("../types/tensor/layout.zig").Layout;
const Storage = @import("../types/tensor/storage.zig").Storage;
const ValueSpec = @import("../types/tensor/value_spec.zig").ValueSpec;
const execution_layout = @import("layout.zig");
const OpTag = @import("../types/operation/tag.zig").OpTag;
const OpOptions = @import("../types/operation/options.zig").OpOptions;
const ExecutionMetadata = @import("../types/operation/execution_metadata.zig").ExecutionMetadata;
const semantic = @import("../sema/index.zig");
const matmul_planning = @import("../plan/matmul.zig");
const materialization_execution = @import("materialization.zig");

pub const BinaryElementwiseDescriptor = union(enum) {
    dense,
    broadcast: semantic.BinaryBroadcastSpec,
};

pub const MaskedFillDescriptor = union(enum) {
    dense,
    broadcast: semantic.MaskedFillBroadcastSpec,
};

pub const WhereDescriptor = union(enum) {
    dense,
    broadcast: semantic.WhereBroadcastSpec,
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
    value: *const Value,
    owned_value: ?*Value = null,
    materialized_packed_dense: bool = false,

    pub fn deinit(self: PreparedInputValue) void {
        if (self.owned_value) |owned| owned.deinit();
    }
};

pub fn isPackedDenseInput(value: *const Value) bool {
    return value.layout.offset == 0 and value.layout.isContiguous(value.shape);
}

pub fn prepareInputValue(
    allocator: std.mem.Allocator,
    value: *const Value,
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
    spec: ValueSpec,
    source: Storage.Source,
) !*Value {
    const output = try Value.createContiguousWithSource(
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
    input: *const Value,
    spec: ValueSpec,
) !*Value {
    const storage = try input.requireRuntimeBacking();
    storage.retain();
    errdefer storage.release();

    var shape = try Shape.initCopy(allocator, spec.shape.dims);
    errdefer shape.deinit();
    var layout = try Layout.initCopy(allocator, spec.layout.strides, spec.layout.offset);
    errdefer layout.deinit();

    const view = try allocator.create(Value);
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
    allocator: std.mem.Allocator,
    tag: OpTag,
    lhs: *const Value,
    rhs: *const Value,
    options: OpOptions,
    fallback_broadcast: ?semantic.BroadcastSpec,
) !BinaryElementwiseDescriptor {
    if (Shape.eql(lhs.shape, rhs.shape) and hasExactPackedStorage(lhs) and hasExactPackedStorage(rhs)) {
        return .dense;
    }

    var prepared_info = try inferPreparedOpSpec(allocator, tag, &.{ lhs, rhs }, options);
    defer prepared_info.deinit();
    const broadcast = prepared_info.broadcast orelse fallback_broadcast orelse return error.InvalidExecutionPlan;
    return switch (broadcast) {
        .binary => |desc| .{ .broadcast = desc },
        else => error.InvalidExecutionPlan,
    };
}

pub fn maskedFillDescriptor(
    allocator: std.mem.Allocator,
    input: *const Value,
    mask: *const Value,
    options: OpOptions,
    fallback_broadcast: ?semantic.BroadcastSpec,
) !MaskedFillDescriptor {
    var prepared_info = try inferPreparedOpSpec(allocator, .masked_fill, &.{ input, mask }, options);
    defer prepared_info.deinit();
    const broadcast = prepared_info.broadcast orelse fallback_broadcast orelse return error.InvalidExecutionPlan;
    const lowered = switch (broadcast) {
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
    allocator: std.mem.Allocator,
    cond: *const Value,
    on_true: *const Value,
    on_false: *const Value,
    options: OpOptions,
    fallback_broadcast: ?semantic.BroadcastSpec,
) !WhereDescriptor {
    var prepared_info = try inferPreparedOpSpec(allocator, .where, &.{ cond, on_true, on_false }, options);
    defer prepared_info.deinit();
    const broadcast = prepared_info.broadcast orelse fallback_broadcast orelse return error.InvalidExecutionPlan;
    const lowered = switch (broadcast) {
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
    lhs: *const Value,
    rhs: *const Value,
    output: ValueSpec,
    metadata: ExecutionMetadata,
) !?MatmulProjectionDescriptor {
    const lhs_spec = try lhs.spec();
    const rhs_spec = try rhs.spec();
    const descriptor = matmul_planning.classifyFromSpecs(lhs_spec, rhs_spec, output, .{
        .hint = metadata.matmul_hint,
        .hint_source = metadata.hint_source,
    });
    if (descriptor.family != .gemm_projection or !descriptor.flattenable_leading_batch) return null;
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

fn hasExactPackedStorage(value: *const Value) bool {
    const storage = value.runtimeBacking() orelse return false;
    return isPackedDenseInput(value) and
        storage.bytes == value.shape.numel() * value.dtype.size();
}

fn inferPreparedOpSpec(
    allocator: std.mem.Allocator,
    tag: OpTag,
    values: []const *const Value,
    options: OpOptions,
) !semantic.OpSpec {
    var specs: [3]ValueSpec = undefined;
    if (values.len > specs.len) return error.InvalidExecutionPlan;
    for (values, 0..) |value, i| specs[i] = try value.spec();
    return semantic.inferFromSpecs(allocator, tag, specs[0..values.len], options);
}

fn makeOffsetViewForTest(
    allocator: std.mem.Allocator,
    base: *const Value,
    dims: []const usize,
    strides: []const isize,
    offset: usize,
) !*Value {
    const storage = try base.requireRuntimeBacking();
    storage.retain();
    errdefer storage.release();

    var shape = try Shape.initCopy(allocator, dims);
    errdefer shape.deinit();
    var layout = try Layout.initCopy(allocator, strides, offset);
    errdefer layout.deinit();

    const view = try allocator.create(Value);
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
    const lhs = try Value.fromSliceF32(allocator, &.{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer lhs.deinit();
    const rhs = try Value.fromSliceF32(allocator, &.{ 2, 2 }, &.{ 5, 6, 7, 8 });
    defer rhs.deinit();

    const descriptor = try binaryElementwiseDescriptor(allocator, .add, lhs, rhs, .{ .none = {} }, null);
    try std.testing.expectEqual(@as(std.meta.Tag(BinaryElementwiseDescriptor), .dense), std.meta.activeTag(descriptor));
}

test "binary descriptor re-infers broadcast after input preparation" {
    const allocator = std.testing.allocator;
    const lhs = try Value.fromSliceF32(allocator, &.{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer lhs.deinit();
    const rhs = try Value.fromSliceF32(allocator, &.{3}, &.{ 10, 20, 30 });
    defer rhs.deinit();

    const descriptor = try binaryElementwiseDescriptor(allocator, .add, lhs, rhs, .{ .none = {} }, null);
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
    const lhs_base = try Value.fromSliceF32(allocator, &.{3}, &.{ 1, 2, 3 });
    defer lhs_base.deinit();
    const lhs = try makeOffsetViewForTest(allocator, lhs_base, &.{2}, &.{1}, 1);
    defer lhs.deinit();
    const rhs = try Value.fromSliceF32(allocator, &.{2}, &.{ 10, 20 });
    defer rhs.deinit();

    const descriptor = try binaryElementwiseDescriptor(allocator, .add, lhs, rhs, .{ .none = {} }, null);
    try std.testing.expectEqual(@as(std.meta.Tag(BinaryElementwiseDescriptor), .broadcast), std.meta.activeTag(descriptor));
}

test "packed dense predicate rejects contiguous offset views" {
    const allocator = std.testing.allocator;
    const base = try Value.fromSliceF32(allocator, &.{3}, &.{ 1, 2, 3 });
    defer base.deinit();
    const view = try makeOffsetViewForTest(allocator, base, &.{2}, &.{1}, 1);
    defer view.deinit();

    try std.testing.expect(!isPackedDenseInput(view));
}

test "packed dense materialization preserves logical view order" {
    const allocator = std.testing.allocator;
    const base = try Value.fromSliceF32(allocator, &.{6}, &.{ 10, 20, 30, 40, 50, 60 });
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
    const input = try Value.fromSliceF32(allocator, &.{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer input.deinit();
    const mask = try Value.fromSliceI64(allocator, &.{ 2, 2 }, &.{ 1, 0, 1, 0 });
    defer mask.deinit();

    const descriptor = try maskedFillDescriptor(allocator, input, mask, .{ .masked_fill = .{ .value = -9.0 } }, null);
    try std.testing.expectEqual(@as(std.meta.Tag(MaskedFillDescriptor), .dense), std.meta.activeTag(descriptor));
}

test "masked_fill descriptor re-infers broadcast after input preparation" {
    const allocator = std.testing.allocator;
    const input = try Value.fromSliceF32(allocator, &.{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer input.deinit();
    const mask = try Value.fromSliceI64(allocator, &.{3}, &.{ 1, 0, 1 });
    defer mask.deinit();

    const descriptor = try maskedFillDescriptor(allocator, input, mask, .{ .masked_fill = .{ .value = -9.0 } }, null);
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
    const input_base = try Value.fromSliceF32(allocator, &.{3}, &.{ 1, 2, 3 });
    defer input_base.deinit();
    const input = try makeOffsetViewForTest(allocator, input_base, &.{2}, &.{1}, 1);
    defer input.deinit();
    const mask = try Value.fromSliceI64(allocator, &.{2}, &.{ 1, 0 });
    defer mask.deinit();

    const descriptor = try maskedFillDescriptor(allocator, input, mask, .{ .masked_fill = .{ .value = -9.0 } }, null);
    try std.testing.expectEqual(@as(std.meta.Tag(MaskedFillDescriptor), .broadcast), std.meta.activeTag(descriptor));
}

test "where descriptor uses dense path for same-shape contiguous inputs" {
    const allocator = std.testing.allocator;
    const cond = try Value.fromSliceI64(allocator, &.{ 2, 2 }, &.{ 1, 0, 1, 0 });
    defer cond.deinit();
    const on_true = try Value.fromSliceF32(allocator, &.{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer on_true.deinit();
    const on_false = try Value.fromSliceF32(allocator, &.{ 2, 2 }, &.{ 5, 6, 7, 8 });
    defer on_false.deinit();

    const descriptor = try whereDescriptor(allocator, cond, on_true, on_false, .{ .none = {} }, null);
    try std.testing.expectEqual(@as(std.meta.Tag(WhereDescriptor), .dense), std.meta.activeTag(descriptor));
}

test "where descriptor re-infers broadcast after input preparation" {
    const allocator = std.testing.allocator;
    const cond = try Value.fromSliceI64(allocator, &.{ 2, 1 }, &.{ 1, 0 });
    defer cond.deinit();
    const on_true = try Value.fromSliceF32(allocator, &.{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer on_true.deinit();
    const on_false = try Value.fromSliceF32(allocator, &.{3}, &.{ 10, 20, 30 });
    defer on_false.deinit();

    const descriptor = try whereDescriptor(allocator, cond, on_true, on_false, .{ .none = {} }, null);
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
    const cond = try Value.fromSliceI64(allocator, &.{2}, &.{ 1, 0 });
    defer cond.deinit();
    const true_base = try Value.fromSliceF32(allocator, &.{3}, &.{ 1, 2, 3 });
    defer true_base.deinit();
    const on_true = try makeOffsetViewForTest(allocator, true_base, &.{2}, &.{1}, 1);
    defer on_true.deinit();
    const on_false = try Value.fromSliceF32(allocator, &.{2}, &.{ 5, 6 });
    defer on_false.deinit();

    const descriptor = try whereDescriptor(allocator, cond, on_true, on_false, .{ .none = {} }, null);
    try std.testing.expectEqual(@as(std.meta.Tag(WhereDescriptor), .broadcast), std.meta.activeTag(descriptor));
}

test "matmul projection descriptor flattens contiguous leading batch" {
    const allocator = std.testing.allocator;
    const lhs = try Value.createContiguous(allocator, &.{ 2, 3, 4 }, .f32, .cpu, false);
    defer lhs.deinit();
    const rhs = try Value.createContiguous(allocator, &.{ 4, 5 }, .f32, .cpu, false);
    defer rhs.deinit();

    var out_shape = try Shape.initCopy(allocator, &.{ 2, 3, 5 });
    defer out_shape.deinit();
    var out_layout = try Layout.initContiguous(allocator, out_shape);
    defer out_layout.deinit();
    const out_spec = ValueSpec{
        .shape = out_shape,
        .dtype = .f32,
        .layout = out_layout,
        .device = .cpu,
    };

    const descriptor = (try matmulProjectionDescriptor(lhs, rhs, out_spec, .{
        .matmul_hint = .projection,
        .hint_source = .higher_level_module,
    })) orelse return error.TestExpectedProjectionDescriptor;
    try std.testing.expectEqualSlices(usize, &.{ 6, 4 }, descriptor.lhsShape());
    try std.testing.expectEqualSlices(isize, &.{ 4, 1 }, descriptor.lhsLayout().strides);
}

test "matmul projection descriptor rejects non-contiguous lhs view" {
    const allocator = std.testing.allocator;
    const lhs = try Value.createContiguous(allocator, &.{ 2, 3, 4 }, .f32, .cpu, false);
    defer lhs.deinit();
    lhs.layout.strides[0] = 13;
    const rhs = try Value.createContiguous(allocator, &.{ 4, 5 }, .f32, .cpu, false);
    defer rhs.deinit();

    var out_shape = try Shape.initCopy(allocator, &.{ 2, 3, 5 });
    defer out_shape.deinit();
    var out_layout = try Layout.initContiguous(allocator, out_shape);
    defer out_layout.deinit();
    const out_spec = ValueSpec{
        .shape = out_shape,
        .dtype = .f32,
        .layout = out_layout,
        .device = .cpu,
    };

    try std.testing.expectEqual(@as(?MatmulProjectionDescriptor, null), try matmulProjectionDescriptor(lhs, rhs, out_spec, .{
        .matmul_hint = .projection,
        .hint_source = .higher_level_module,
    }));
}
