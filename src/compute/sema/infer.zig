const std = @import("std");
const tensor = @import("../types/tensor/index.zig");
const Device = tensor.Device;
const DType = tensor.DType;
const Shape = tensor.Shape;
const Layout = tensor.Layout;
const Value = tensor.Value;
const ValueSpec = tensor.ValueSpec;
const AxisName = tensor.AxisName;
const Op = @import("../types/operation/op.zig").Op;
const contracts = @import("../types/operation/contracts.zig");
const OpOptions = @import("../types/operation/options.zig").OpOptions;
const SliceRange = @import("../types/operation/options.zig").SliceRange;
const kernel_capability = @import("../backend/capability.zig");
const semantic_spec = @import("spec.zig");

pub const ExecutionKind = semantic_spec.ExecutionKind;
pub const AllocationIntent = semantic_spec.AllocationIntent;
pub const InputRequirement = semantic_spec.InputRequirement;
pub const OutputSpec = semantic_spec.OutputSpec;
pub const BinaryBroadcastSpec = semantic_spec.BinaryBroadcastSpec;
pub const WhereBroadcastSpec = semantic_spec.WhereBroadcastSpec;
pub const MaskedFillBroadcastSpec = semantic_spec.MaskedFillBroadcastSpec;
pub const ReduceToShapeSpec = semantic_spec.ReduceToShapeSpec;
pub const BroadcastSpec = semantic_spec.BroadcastSpec;
pub const PlannerHint = semantic_spec.PlannerHint;
pub const OpSpec = semantic_spec.OpSpec;

fn cloneAxesForShape(allocator: std.mem.Allocator, axes: ?[]const AxisName, rank: usize) !?[]const AxisName {
    const names = axes orelse return null;
    if (names.len != rank) return null;
    return tensor.cloneAxes(allocator, names);
}

fn permuteAxes(allocator: std.mem.Allocator, axes: ?[]const AxisName, order: []const usize) !?[]const AxisName {
    const names = axes orelse return null;
    if (names.len != order.len) return null;
    const out = try allocator.alloc(AxisName, order.len);
    var initialized: usize = 0;
    errdefer {
        for (out[0..initialized]) |axis| allocator.free(axis);
        allocator.free(out);
    }
    for (order, 0..) |source_dim, i| {
        out[i] = try allocator.dupe(u8, names[source_dim]);
        initialized += 1;
    }
    return out;
}

fn reduceAxisNames(
    allocator: std.mem.Allocator,
    axes: ?[]const AxisName,
    removed_axis: usize,
    keepdim: bool,
) !?[]const AxisName {
    const names = axes orelse return null;
    if (removed_axis >= names.len) return null;
    if (keepdim) return tensor.cloneAxes(allocator, names);
    const out = try allocator.alloc(AxisName, names.len - 1);
    var out_i: usize = 0;
    errdefer {
        for (out[0..out_i]) |axis| allocator.free(axis);
        allocator.free(out);
    }
    for (names, 0..) |name, i| {
        if (i == removed_axis) continue;
        out[out_i] = try allocator.dupe(u8, name);
        out_i += 1;
    }
    return out;
}

fn squeezeAxisNames(
    allocator: std.mem.Allocator,
    axes: ?[]const AxisName,
    input_shape: []const usize,
    selected_axis: ?usize,
) !?[]const AxisName {
    const names = axes orelse return null;
    if (names.len != input_shape.len) return null;

    var kept: usize = 0;
    for (input_shape, 0..) |dim, i| {
        if (selected_axis) |selected| {
            if (i == selected) continue;
        } else if (dim == 1) {
            continue;
        }
        kept += 1;
    }

    const out = try allocator.alloc(AxisName, kept);
    var out_i: usize = 0;
    errdefer {
        for (out[0..out_i]) |axis| allocator.free(axis);
        allocator.free(out);
    }
    for (names, 0..) |name, i| {
        if (selected_axis) |selected| {
            if (i == selected) continue;
        } else if (input_shape[i] == 1) {
            continue;
        }
        out[out_i] = try allocator.dupe(u8, name);
        out_i += 1;
    }
    return out;
}

fn axisNamesEqual(a: ?[]const AxisName, b: ?[]const AxisName) bool {
    const lhs = a orelse return b == null;
    const rhs = b orelse return false;
    if (lhs.len != rhs.len) return false;
    for (lhs, rhs) |left, right| {
        if (!std.mem.eql(u8, left, right)) return false;
    }
    return true;
}

fn binaryElementwiseAxes(allocator: std.mem.Allocator, a: ValueSpec, b: ValueSpec, output_shape: Shape) !?[]const AxisName {
    const a_full = Shape.eql(a.shape, output_shape);
    const b_full = Shape.eql(b.shape, output_shape);
    if (a_full and b_full) {
        if (axisNamesEqual(a.axes, b.axes)) return cloneAxesForShape(allocator, a.axes, output_shape.rank());
        if (a.axes == null) return cloneAxesForShape(allocator, b.axes, output_shape.rank());
        if (b.axes == null) return cloneAxesForShape(allocator, a.axes, output_shape.rank());
        return null;
    }
    if (a_full and shapesBroadcastTo(b.shape.dims, output_shape.dims)) {
        return cloneAxesForShape(allocator, a.axes, output_shape.rank());
    }
    if (b_full and shapesBroadcastTo(a.shape.dims, output_shape.dims)) {
        return cloneAxesForShape(allocator, b.axes, output_shape.rank());
    }
    if (a_full and b.shape.numel() == 1) {
        return cloneAxesForShape(allocator, a.axes, output_shape.rank());
    }
    if (b_full and a.shape.numel() == 1) {
        return cloneAxesForShape(allocator, b.axes, output_shape.rank());
    }
    return null;
}

fn shapesBroadcastTo(source: []const usize, target: []const usize) bool {
    if (source.len > target.len) return false;
    var i: usize = 0;
    while (i < source.len) : (i += 1) {
        const source_dim = source[source.len - 1 - i];
        const target_dim = target[target.len - 1 - i];
        if (source_dim != 1 and source_dim != target_dim) return false;
    }
    return true;
}

fn indexSelectAxes(allocator: std.mem.Allocator, input: ValueSpec, index: ValueSpec, axis: usize) !?[]const AxisName {
    const input_axes = input.axes orelse return null;
    if (input_axes.len != input.shape.rank()) return null;
    const out = try allocator.alloc(AxisName, input_axes.len);
    var out_i: usize = 0;
    errdefer {
        for (out[0..out_i]) |name| allocator.free(name);
        allocator.free(out);
    }
    for (input_axes, 0..) |name, i| {
        const source_name = if (i == axis and index.axes != null and index.axes.?.len == 1)
            index.axes.?[0]
        else
            name;
        out[out_i] = try allocator.dupe(u8, source_name);
        out_i += 1;
    }
    return out;
}

fn gatherAxes(allocator: std.mem.Allocator, input: ValueSpec, index: ValueSpec, axis: usize) !?[]const AxisName {
    const input_axes = input.axes orelse return cloneAxesForShape(allocator, index.axes, index.shape.rank());
    if (input_axes.len != input.shape.rank()) return null;
    const index_axes = index.axes;
    if (index_axes) |names| {
        if (names.len != index.shape.rank()) return null;
    }

    const out = try allocator.alloc(AxisName, index.shape.rank());
    var out_i: usize = 0;
    errdefer {
        for (out[0..out_i]) |name| allocator.free(name);
        allocator.free(out);
    }

    for (0..index.shape.rank()) |i| {
        const index_name = if (index_axes) |names| names[i] else null;
        const source_name = if (i == axis and index_name != null)
            index_name.?
        else
            input_axes[i];

        if (i != axis and index_name != null and !std.mem.eql(u8, source_name, index_name.?)) {
            for (out[0..out_i]) |name| allocator.free(name);
            allocator.free(out);
            return null;
        }

        out[out_i] = try allocator.dupe(u8, source_name);
        out_i += 1;
    }
    return out;
}

fn embeddingAxes(allocator: std.mem.Allocator, table: ValueSpec, index: ValueSpec) !?[]const AxisName {
    const table_axes = table.axes orelse return null;
    if (table_axes.len != table.shape.rank()) return null;
    if (index.axes) |index_axes| {
        if (index_axes.len != index.shape.rank()) return null;
    }
    const out_rank = index.shape.rank() + table.shape.rank() - 1;
    const out = try allocator.alloc(AxisName, out_rank);
    var out_i: usize = 0;
    errdefer {
        for (out[0..out_i]) |name| allocator.free(name);
        allocator.free(out);
    }
    if (index.axes) |index_axes| {
        for (index_axes) |name| {
            out[out_i] = try allocator.dupe(u8, name);
            out_i += 1;
        }
    }
    for (table_axes[1..]) |name| {
        out[out_i] = try allocator.dupe(u8, name);
        out_i += 1;
    }
    return out;
}

fn matmulAxes(allocator: std.mem.Allocator, lhs: ValueSpec, rhs: ValueSpec, output_shape: Shape) !?[]const AxisName {
    const lhs_axes = lhs.axes orelse return null;
    const rhs_axes = rhs.axes orelse return null;
    const lhs_rank = lhs.shape.rank();
    const rhs_rank = rhs.shape.rank();
    if (lhs_axes.len != lhs_rank or rhs_axes.len != rhs_rank) return null;
    if (lhs_rank < 2 or rhs_rank < 2) return null;
    if (!std.mem.eql(u8, lhs_axes[lhs_rank - 1], rhs_axes[rhs_rank - 2])) return null;
    if (rhs_rank > 2) {
        if (lhs_rank != rhs_rank) return null;
        for (0..lhs_rank - 2) |i| {
            if (!std.mem.eql(u8, lhs_axes[i], rhs_axes[i])) return null;
        }
    }
    const expected_rank = lhs_rank;
    if (output_shape.rank() != expected_rank) return null;
    const out = try allocator.alloc(AxisName, expected_rank);
    var out_i: usize = 0;
    errdefer {
        for (out[0..out_i]) |name| allocator.free(name);
        allocator.free(out);
    }
    for (lhs_axes[0 .. lhs_rank - 1]) |name| {
        out[out_i] = try allocator.dupe(u8, name);
        out_i += 1;
    }
    out[out_i] = try allocator.dupe(u8, rhs_axes[rhs_rank - 1]);
    return out;
}

fn matchingInputAxes(allocator: std.mem.Allocator, inputs: []const ValueSpec, rank: usize) !?[]const AxisName {
    if (inputs.len == 0) return null;
    const first_axes = inputs[0].axes orelse return null;
    if (first_axes.len != rank) return null;
    for (inputs[1..]) |input| {
        if (!axisNamesEqual(first_axes, input.axes)) return null;
    }
    return tensor.cloneAxes(allocator, first_axes);
}

pub fn infer(allocator: std.mem.Allocator, op: Op) !OpSpec {
    try op.validate();
    const specs = try allocator.alloc(ValueSpec, op.inputs.len);
    defer allocator.free(specs);
    for (op.inputs, 0..) |input, i| specs[i] = try input.spec();

    return inferFromSpecs(allocator, op.tag, specs, op.options);
}

pub fn inferFromSpecs(
    allocator: std.mem.Allocator,
    tag: @import("../types/operation/tag.zig").OpTag,
    specs: []const ValueSpec,
    options: OpOptions,
) !OpSpec {
    return switch (tag) {
        .add => inferAdd(allocator, specs),
        .sub => inferSub(allocator, specs),
        .mul => inferMul(allocator, specs),
        .div => inferDiv(allocator, specs),
        .eq => inferEq(allocator, specs),
        .lt => inferLt(allocator, specs),
        .gt => inferGt(allocator, specs),
        .abs => inferAbs(allocator, specs),
        .exp => inferExp(allocator, specs),
        .log => inferLog(allocator, specs),
        .neg => inferNeg(allocator, specs),
        .sqrt => inferSqrt(allocator, specs),
        .sign => inferSign(allocator, specs),
        .relu => inferRelu(allocator, specs),
        .sigmoid => inferSigmoid(allocator, specs),
        .silu => inferSilu(allocator, specs),
        .tanh => inferTanh(allocator, specs),
        .gelu => inferGelu(allocator, specs),
        .gelu_grad => inferGeluGrad(allocator, specs),
        .clamp => inferClamp(allocator, specs, options),
        .where => inferWhere(allocator, specs),
        .masked_fill => inferMaskedFill(allocator, specs, options),
        .cast => inferCast(allocator, specs, options),
        .softmax => inferSoftmax(allocator, specs, options),
        .log_softmax => inferLogSoftmax(allocator, specs, options),
        .log_softmax_nll => inferLogSoftmaxNll(allocator, specs, options),
        .cross_entropy_indexed => inferCrossEntropyIndexed(allocator, specs, options),
        .cross_entropy_indexed_backward => inferCrossEntropyIndexedBackward(allocator, specs, options),
        .cross_entropy => inferCrossEntropy(allocator, specs, options),
        .layer_norm => inferLayerNorm(allocator, specs, options),
        .rms_norm => inferRmsNorm(allocator, specs, options),
        .sum_all => inferSumAll(allocator, specs, options),
        .mean_all => inferMeanAll(allocator, specs, options),
        .min_all => inferMinAll(allocator, specs, options),
        .max_all => inferMaxAll(allocator, specs, options),
        .variance_all => inferVarianceAll(allocator, specs, options),
        .std_all => inferStdAll(allocator, specs, options),
        .argmin_all => inferArgminAll(allocator, specs, options),
        .argmax_all => inferArgmaxAll(allocator, specs, options),
        .sum_axis => inferSumAxis(allocator, specs, options),
        .mean_axis => inferMeanAxis(allocator, specs, options),
        .min_axis => inferMinAxis(allocator, specs, options),
        .max_axis => inferMaxAxis(allocator, specs, options),
        .variance_axis => inferVarianceAxis(allocator, specs, options),
        .std_axis => inferStdAxis(allocator, specs, options),
        .argmin_axis => inferArgminAxis(allocator, specs, options),
        .argmax_axis => inferArgmaxAxis(allocator, specs, options),
        .cat => inferCat(allocator, specs, options),
        .stack => inferStack(allocator, specs, options),
        .contiguous => inferContiguous(allocator, specs),
        .reshape => inferReshape(allocator, specs, options),
        .slice => inferSlice(allocator, specs, options),
        .gather => inferGather(allocator, specs, options),
        .embedding => inferEmbedding(allocator, specs, options),
        .index_select => inferIndexSelect(allocator, specs, options),
        .scatter_add => inferScatterAdd(allocator, specs, options),
        .topk => inferTopK(allocator, specs, options),
        .one_hot => inferOneHot(allocator, specs, options),
        .reduce_to_shape => inferReduceToShape(allocator, specs, options),
        .permute => inferPermute(allocator, specs, options),
        .transpose => inferTranspose(allocator, specs, options),
        .squeeze => inferSqueeze(allocator, specs, options),
        .unsqueeze => inferUnsqueeze(allocator, specs, options),
        .dot => inferDot(allocator, specs),
        .matmul => inferMatmul(allocator, specs),
    };
}

fn alignRightAxis(out_axis: usize, out_rank: usize, in_rank: usize) ?usize {
    if (out_rank < in_rank) return null;
    if (out_axis < out_rank - in_rank) return null;
    return out_axis - (out_rank - in_rank);
}

fn broadcastShapeInto(buf: *[8]usize, lhs: []const usize, rhs: []const usize) ![]const usize {
    const rank = @max(lhs.len, rhs.len);
    if (rank > buf.len) return error.UnsupportedShape;
    for (0..rank) |axis| {
        const lhs_axis = alignRightAxis(axis, rank, lhs.len);
        const rhs_axis = alignRightAxis(axis, rank, rhs.len);
        const lhs_dim = if (lhs_axis) |idx| lhs[idx] else 1;
        const rhs_dim = if (rhs_axis) |idx| rhs[idx] else 1;
        if (lhs_dim != rhs_dim and lhs_dim != 1 and rhs_dim != 1) return error.ShapeMismatch;
        buf[axis] = @max(lhs_dim, rhs_dim);
    }
    return buf[0..rank];
}

fn inferBroadcastShape(allocator: std.mem.Allocator, lhs: []const usize, rhs: []const usize) ![]usize {
    var buf: [8]usize = undefined;
    const shape = try broadcastShapeInto(&buf, lhs, rhs);
    return allocator.dupe(usize, shape);
}

fn normalizeBroadcastStridesInto(
    buf: *[8]isize,
    out_shape: []const usize,
    in_shape: []const usize,
    in_strides: []const isize,
) !void {
    if (out_shape.len > buf.len or in_shape.len != in_strides.len) return error.ShapeMismatch;
    for (out_shape, 0..) |out_dim, axis| {
        const in_axis = alignRightAxis(axis, out_shape.len, in_shape.len);
        if (in_axis == null) {
            buf[axis] = 0;
            continue;
        }
        const dim = in_shape[in_axis.?];
        if (dim != out_dim and dim != 1) return error.ShapeMismatch;
        buf[axis] = if (dim == 1 and out_dim > 1) 0 else in_strides[in_axis.?];
    }
}

fn validateReduceToShape(input_shape: []const usize, target: []const usize) !void {
    if (target.len > input_shape.len) return error.ShapeMismatch;
    for (input_shape, 0..) |in_dim, axis| {
        const target_axis = alignRightAxis(axis, input_shape.len, target.len);
        const out_dim = if (target_axis) |idx| target[idx] else 1;
        if (in_dim == out_dim) continue;
        if (out_dim != 1 or in_dim == 0) return error.ShapeMismatch;
    }
}

fn inferBinaryBroadcastSpec(lhs: ValueSpec, rhs: ValueSpec) !BroadcastSpec {
    var out = BinaryBroadcastSpec{
        .rank = 0,
        .shape = [_]usize{0} ** 8,
        .lhs_strides = [_]isize{0} ** 8,
        .rhs_strides = [_]isize{0} ** 8,
    };
    const shape = try broadcastShapeInto(&out.shape, lhs.shape.dims, rhs.shape.dims);
    out.rank = shape.len;
    try normalizeBroadcastStridesInto(&out.lhs_strides, shape, lhs.shape.dims, lhs.layout.strides);
    try normalizeBroadcastStridesInto(&out.rhs_strides, shape, rhs.shape.dims, rhs.layout.strides);
    return .{ .binary = out };
}

test "binary broadcast spec handles right-aligned rank-changing shapes" {
    const allocator = std.testing.allocator;

    var lhs_shape = try Shape.initCopy(allocator, &.{2});
    defer lhs_shape.deinit();
    var lhs_layout = try Layout.initContiguous(allocator, lhs_shape);
    defer lhs_layout.deinit();

    var rhs_shape = try Shape.initCopy(allocator, &.{ 1, 2 });
    defer rhs_shape.deinit();
    var rhs_layout = try Layout.initContiguous(allocator, rhs_shape);
    defer rhs_layout.deinit();

    const lhs = ValueSpec{
        .shape = lhs_shape,
        .dtype = .f32,
        .layout = lhs_layout,
        .device = .cpu,
    };
    const rhs = ValueSpec{
        .shape = rhs_shape,
        .dtype = .f32,
        .layout = rhs_layout,
        .device = .cpu,
    };

    const spec = try inferBinaryBroadcastSpec(lhs, rhs);
    switch (spec) {
        .binary => |desc| {
            try std.testing.expectEqual(@as(usize, 2), desc.rank);
            try std.testing.expectEqualSlices(usize, &.{ 1, 2 }, desc.shape[0..desc.rank]);
            try std.testing.expectEqualSlices(isize, &.{ 0, 1 }, desc.lhs_strides[0..desc.rank]);
            try std.testing.expectEqualSlices(isize, &.{ 2, 1 }, desc.rhs_strides[0..desc.rank]);
        },
        else => return error.UnexpectedSpec,
    }
}

fn inferWhereBroadcastSpec(cond: ValueSpec, on_true: ValueSpec, on_false: ValueSpec) !BroadcastSpec {
    var out = WhereBroadcastSpec{
        .rank = 0,
        .shape = [_]usize{0} ** 8,
        .cond_strides = [_]isize{0} ** 8,
        .on_true_strides = [_]isize{0} ** 8,
        .on_false_strides = [_]isize{0} ** 8,
    };
    var cond_true_buf: [8]usize = [_]usize{0} ** 8;
    const cond_true = try broadcastShapeInto(&cond_true_buf, cond.shape.dims, on_true.shape.dims);
    const shape = try broadcastShapeInto(&out.shape, cond_true, on_false.shape.dims);
    out.rank = shape.len;
    try normalizeBroadcastStridesInto(&out.cond_strides, shape, cond.shape.dims, cond.layout.strides);
    try normalizeBroadcastStridesInto(&out.on_true_strides, shape, on_true.shape.dims, on_true.layout.strides);
    try normalizeBroadcastStridesInto(&out.on_false_strides, shape, on_false.shape.dims, on_false.layout.strides);
    return .{ .where = out };
}

fn inferMaskedFillBroadcastSpec(input: ValueSpec, mask: ValueSpec) !BroadcastSpec {
    var out = MaskedFillBroadcastSpec{
        .rank = 0,
        .shape = [_]usize{0} ** 8,
        .input_strides = [_]isize{0} ** 8,
        .mask_strides = [_]isize{0} ** 8,
    };
    const shape = try broadcastShapeInto(&out.shape, input.shape.dims, mask.shape.dims);
    out.rank = shape.len;
    try normalizeBroadcastStridesInto(&out.input_strides, shape, input.shape.dims, input.layout.strides);
    try normalizeBroadcastStridesInto(&out.mask_strides, shape, mask.shape.dims, mask.layout.strides);
    return .{ .masked_fill = out };
}

fn inferReduceToShapeSpec(input_shape: []const usize, target: []const usize) !ReduceToShapeSpec {
    try validateReduceToShape(input_shape, target);
    var out = ReduceToShapeSpec{
        .axes = [_]usize{0} ** 8,
        .count = 0,
    };
    for (input_shape, 0..) |in_dim, axis| {
        const target_axis = alignRightAxis(axis, input_shape.len, target.len);
        const out_dim = if (target_axis) |idx| target[idx] else 1;
        if (in_dim != out_dim) {
            out.axes[out.count] = axis;
            out.count += 1;
        }
    }
    return out;
}

fn inferAdd(allocator: std.mem.Allocator, inputs: []const ValueSpec) !OpSpec {
    return buildBinaryElementwiseSpec(allocator, inputs, .{ .output_dtype = null });
}

fn inferSub(allocator: std.mem.Allocator, inputs: []const ValueSpec) !OpSpec {
    return buildBinaryElementwiseSpec(allocator, inputs, .{ .output_dtype = null });
}

fn inferMul(allocator: std.mem.Allocator, inputs: []const ValueSpec) !OpSpec {
    return buildBinaryElementwiseSpec(allocator, inputs, .{ .output_dtype = null });
}

fn inferDiv(allocator: std.mem.Allocator, inputs: []const ValueSpec) !OpSpec {
    return buildBinaryElementwiseSpec(allocator, inputs, .{ .output_dtype = null });
}

const BinaryElementwiseSpecConfig = struct {
    output_dtype: ?DType,
};

fn buildBinaryElementwiseSpec(
    allocator: std.mem.Allocator,
    inputs: []const ValueSpec,
    config: BinaryElementwiseSpecConfig,
) !OpSpec {
    try contracts.requireInputCount(inputs, 2);
    try contracts.requireAllRanks(inputs, .{ .range = .{ .min = 0, .max = 8 } });
    const a = inputs[0];
    const b = inputs[1];
    if (a.dtype != b.dtype) return error.DTypeMismatch;
    const out_dims = try inferBroadcastShape(allocator, a.shape.dims, b.shape.dims);
    defer allocator.free(out_dims);
    const device = a.device;
    if (device != (b.device)) return error.DeviceMismatch;
    const output_dtype = config.output_dtype orelse a.dtype;
    const input_layout_decision = if (Shape.eql(a.shape, b.shape))
        kernel_capability.binaryInputLayoutDecision(device, a.dtype, a.shape.dims, a.layout, b.shape.dims, b.layout)
    else
        kernel_capability.binaryBroadcastInputLayoutDecision(device, a.dtype, a.shape.dims, a.layout, b.shape.dims, b.layout);

    var shape = try Shape.initCopy(allocator, out_dims);
    errdefer shape.deinit();
    var layout = try Layout.initContiguous(allocator, shape);
    errdefer layout.deinit();
    const axes = try binaryElementwiseAxes(allocator, a, b, shape);
    errdefer tensor.deinitAxes(allocator, axes);

    return .{
        .shape = shape,
        .dtype = output_dtype,
        .layout = layout,
        .axes = axes,
        .device = device,
        .kind = .elementwise_binary,
        .allocation = .new_storage,
        .input_requirement = .require_storage,
        .broadcast = try inferBinaryBroadcastSpec(a, b),
        .planner_hint = .{ .input_layout_decision = input_layout_decision },
    };
}

fn inferEq(allocator: std.mem.Allocator, inputs: []const ValueSpec) !OpSpec {
    return buildBinaryElementwiseSpec(allocator, inputs, .{ .output_dtype = .i64 });
}

fn inferLt(allocator: std.mem.Allocator, inputs: []const ValueSpec) !OpSpec {
    return buildBinaryElementwiseSpec(allocator, inputs, .{ .output_dtype = .i64 });
}

fn inferGt(allocator: std.mem.Allocator, inputs: []const ValueSpec) !OpSpec {
    return buildBinaryElementwiseSpec(allocator, inputs, .{ .output_dtype = .i64 });
}

fn inferAbs(allocator: std.mem.Allocator, inputs: []const ValueSpec) !OpSpec {
    return buildUnaryElementwiseSpec(allocator, inputs);
}

fn inferExp(allocator: std.mem.Allocator, inputs: []const ValueSpec) !OpSpec {
    return buildUnaryElementwiseSpec(allocator, inputs);
}

fn inferLog(allocator: std.mem.Allocator, inputs: []const ValueSpec) !OpSpec {
    return buildUnaryElementwiseSpec(allocator, inputs);
}

fn inferNeg(allocator: std.mem.Allocator, inputs: []const ValueSpec) !OpSpec {
    return buildUnaryElementwiseSpec(allocator, inputs);
}

fn inferSqrt(allocator: std.mem.Allocator, inputs: []const ValueSpec) !OpSpec {
    return buildUnaryElementwiseSpec(allocator, inputs);
}

fn inferSign(allocator: std.mem.Allocator, inputs: []const ValueSpec) !OpSpec {
    return buildUnaryElementwiseSpec(allocator, inputs);
}

fn inferRelu(allocator: std.mem.Allocator, inputs: []const ValueSpec) !OpSpec {
    return buildUnaryElementwiseSpec(allocator, inputs);
}

fn inferSigmoid(allocator: std.mem.Allocator, inputs: []const ValueSpec) !OpSpec {
    return buildUnaryElementwiseSpec(allocator, inputs);
}

fn inferSilu(allocator: std.mem.Allocator, inputs: []const ValueSpec) !OpSpec {
    return buildUnaryElementwiseSpec(allocator, inputs);
}

fn inferTanh(allocator: std.mem.Allocator, inputs: []const ValueSpec) !OpSpec {
    return buildUnaryElementwiseSpec(allocator, inputs);
}

fn inferGelu(allocator: std.mem.Allocator, inputs: []const ValueSpec) !OpSpec {
    return buildUnaryElementwiseSpec(allocator, inputs);
}

fn inferGeluGrad(allocator: std.mem.Allocator, inputs: []const ValueSpec) !OpSpec {
    return buildUnaryElementwiseSpec(allocator, inputs);
}

fn buildUnaryElementwiseSpec(allocator: std.mem.Allocator, inputs: []const ValueSpec) !OpSpec {
    try contracts.requireInputCount(inputs, 1);
    try contracts.requireRank(inputs[0], .{ .range = .{ .min = 0, .max = 8 } });
    const input = inputs[0];
    const device = input.device;
    const input_layout_decision = kernel_capability.unaryInputLayoutDecision(device, input.dtype, input.shape.dims, input.layout);

    var shape = try Shape.initCopy(allocator, input.shape.dims);
    errdefer shape.deinit();
    var layout = try Layout.initContiguous(allocator, shape);
    errdefer layout.deinit();
    const axes = try cloneAxesForShape(allocator, input.axes, input.shape.rank());
    errdefer tensor.deinitAxes(allocator, axes);

    return .{
        .shape = shape,
        .dtype = input.dtype,
        .layout = layout,
        .axes = axes,
        .device = device,
        .kind = .elementwise_unary,
        .allocation = .new_storage,
        .input_requirement = .require_storage,
        .planner_hint = .{ .input_layout_decision = input_layout_decision },
    };
}

fn inferClamp(allocator: std.mem.Allocator, inputs: []const ValueSpec, options: OpOptions) !OpSpec {
    switch (options) {
        .clamp => {},
        else => return error.InvalidOpOptions,
    }
    return buildUnaryElementwiseSpec(allocator, inputs);
}

fn inferWhere(allocator: std.mem.Allocator, inputs: []const ValueSpec) !OpSpec {
    try contracts.requireInputCount(inputs, 3);
    try contracts.requireAllRanks(inputs, .{ .range = .{ .min = 0, .max = 8 } });
    const cond = inputs[0];
    const on_true = inputs[1];
    const on_false = inputs[2];
    const device = cond.device;

    if ((on_true.device) != device) return error.DeviceMismatch;
    if ((on_false.device) != device) return error.DeviceMismatch;
    if (on_true.dtype != on_false.dtype) return error.DTypeMismatch;

    const cond_true = try inferBroadcastShape(allocator, cond.shape.dims, on_true.shape.dims);
    defer allocator.free(cond_true);
    const out_dims = try inferBroadcastShape(allocator, cond_true, on_false.shape.dims);
    defer allocator.free(out_dims);
    const input_layout_decision = if (std.mem.eql(usize, cond.shape.dims, out_dims) and
        std.mem.eql(usize, on_true.shape.dims, out_dims) and
        std.mem.eql(usize, on_false.shape.dims, out_dims))
        kernel_capability.whereInputLayoutDecision(
            device,
            cond.dtype,
            on_true.dtype,
            cond.shape.dims,
            cond.layout,
            on_true.shape.dims,
            on_true.layout,
            on_false.shape.dims,
            on_false.layout,
        )
    else
        kernel_capability.whereBroadcastInputLayoutDecision(
            device,
            cond.dtype,
            on_true.dtype,
            cond.shape.dims,
            cond.layout,
            on_true.shape.dims,
            on_true.layout,
            on_false.shape.dims,
            on_false.layout,
        );

    var shape = try Shape.initCopy(allocator, out_dims);
    errdefer shape.deinit();
    var layout = try Layout.initContiguous(allocator, shape);
    errdefer layout.deinit();
    const axes = try binaryElementwiseAxes(allocator, on_true, on_false, shape);
    errdefer tensor.deinitAxes(allocator, axes);

    return .{
        .shape = shape,
        .dtype = on_true.dtype,
        .layout = layout,
        .axes = axes,
        .device = device,
        .kind = .elementwise_generic,
        .allocation = .new_storage,
        .input_requirement = .require_storage,
        .broadcast = try inferWhereBroadcastSpec(cond, on_true, on_false),
        .planner_hint = .{ .input_layout_decision = input_layout_decision },
    };
}

fn inferMaskedFill(allocator: std.mem.Allocator, inputs: []const ValueSpec, options: OpOptions) !OpSpec {
    try contracts.requireInputCount(inputs, 2);
    try contracts.requireAllRanks(inputs, .{ .range = .{ .min = 0, .max = 8 } });
    switch (options) {
        .masked_fill => {},
        else => return error.InvalidOpOptions,
    }

    const input = inputs[0];
    const mask = inputs[1];
    const device = input.device;
    if ((mask.device) != device) return error.DeviceMismatch;

    const out_dims = try inferBroadcastShape(allocator, input.shape.dims, mask.shape.dims);
    defer allocator.free(out_dims);
    const input_layout_decision = if (std.mem.eql(usize, input.shape.dims, out_dims) and std.mem.eql(usize, mask.shape.dims, out_dims))
        kernel_capability.maskedFillInputLayoutDecision(
            device,
            input.dtype,
            mask.dtype,
            input.shape.dims,
            input.layout,
            mask.shape.dims,
            mask.layout,
        )
    else
        kernel_capability.maskedFillBroadcastInputLayoutDecision(
            device,
            input.dtype,
            mask.dtype,
            input.shape.dims,
            input.layout,
            mask.shape.dims,
            mask.layout,
        );

    var shape = try Shape.initCopy(allocator, out_dims);
    errdefer shape.deinit();
    var layout = try Layout.initContiguous(allocator, shape);
    errdefer layout.deinit();
    const axes = if (Shape.eql(input.shape, shape))
        try cloneAxesForShape(allocator, input.axes, input.shape.rank())
    else
        null;
    errdefer tensor.deinitAxes(allocator, axes);

    return .{
        .shape = shape,
        .dtype = input.dtype,
        .layout = layout,
        .axes = axes,
        .device = device,
        .kind = .elementwise_generic,
        .allocation = .new_storage,
        .input_requirement = .require_storage,
        .broadcast = try inferMaskedFillBroadcastSpec(input, mask),
        .planner_hint = .{ .input_layout_decision = input_layout_decision },
    };
}

fn inferLayerNorm(allocator: std.mem.Allocator, inputs: []const ValueSpec, options: OpOptions) !OpSpec {
    try contracts.requireInputCount(inputs, 1);
    try contracts.requireRank(inputs[0], .{ .range = .{ .min = 1, .max = 8 } });
    const input = inputs[0];
    const ln = switch (options) {
        .layer_norm => |v| v,
        else => return error.InvalidOpOptions,
    };
    if (!(ln.eps > 0.0)) return error.InvalidEpsilon;
    try contracts.requireAxisInBounds(input.shape.rank(), ln.axis, false);
    const input_layout_decision = kernel_capability.layerNormInputLayoutDecision(input.device, input.dtype, input.shape.dims, input.layout);

    var shape = try Shape.initCopy(allocator, input.shape.dims);
    errdefer shape.deinit();
    var layout = try Layout.initContiguous(allocator, shape);
    errdefer layout.deinit();
    const axes = try cloneAxesForShape(allocator, input.axes, input.shape.rank());
    errdefer tensor.deinitAxes(allocator, axes);

    return .{
        .shape = shape,
        .dtype = input.dtype,
        .layout = layout,
        .axes = axes,
        .device = input.device,
        .kind = .elementwise_generic,
        .allocation = .new_storage,
        .input_requirement = .require_storage,
        .planner_hint = .{ .input_layout_decision = input_layout_decision },
    };
}

fn inferRmsNorm(allocator: std.mem.Allocator, inputs: []const ValueSpec, options: OpOptions) !OpSpec {
    try contracts.requireInputCount(inputs, 1);
    try contracts.requireRank(inputs[0], .{ .range = .{ .min = 1, .max = 8 } });
    const input = inputs[0];
    const rn = switch (options) {
        .rms_norm => |v| v,
        else => return error.InvalidOpOptions,
    };
    if (!(rn.eps > 0.0)) return error.InvalidEpsilon;
    try contracts.requireAxisInBounds(input.shape.rank(), rn.axis, false);
    const input_layout_decision = kernel_capability.rmsNormInputLayoutDecision(input.device, input.dtype, input.shape.dims, input.layout);

    var shape = try Shape.initCopy(allocator, input.shape.dims);
    errdefer shape.deinit();
    var layout = try Layout.initContiguous(allocator, shape);
    errdefer layout.deinit();
    const axes = try cloneAxesForShape(allocator, input.axes, input.shape.rank());
    errdefer tensor.deinitAxes(allocator, axes);

    return .{
        .shape = shape,
        .dtype = input.dtype,
        .layout = layout,
        .axes = axes,
        .device = input.device,
        .kind = .elementwise_generic,
        .allocation = .new_storage,
        .input_requirement = .require_storage,
        .planner_hint = .{ .input_layout_decision = input_layout_decision },
    };
}

fn inferCast(allocator: std.mem.Allocator, inputs: []const ValueSpec, options: OpOptions) !OpSpec {
    try contracts.requireInputCount(inputs, 1);
    try contracts.requireRank(inputs[0], .{ .range = .{ .min = 0, .max = 8 } });
    const input = inputs[0];
    const cast = switch (options) {
        .cast => |v| v,
        else => return error.InvalidOpOptions,
    };
    const input_layout_decision = kernel_capability.castInputLayoutDecision(input.device, input.dtype, cast.to, input.shape.dims, input.layout);
    var shape = try Shape.initCopy(allocator, input.shape.dims);
    errdefer shape.deinit();
    var layout = try Layout.initContiguous(allocator, shape);
    errdefer layout.deinit();
    const axes = try cloneAxesForShape(allocator, input.axes, input.shape.rank());
    errdefer tensor.deinitAxes(allocator, axes);
    return .{
        .shape = shape,
        .dtype = cast.to,
        .layout = layout,
        .axes = axes,
        .device = input.device,
        .kind = .elementwise_generic,
        .allocation = .new_storage,
        .input_requirement = .require_storage,
        .planner_hint = .{ .input_layout_decision = input_layout_decision },
    };
}

fn inferSoftmax(allocator: std.mem.Allocator, inputs: []const ValueSpec, options: OpOptions) !OpSpec {
    try contracts.requireInputCount(inputs, 1);
    try contracts.requireRank(inputs[0], .{ .range = .{ .min = 1, .max = 8 } });
    const input = inputs[0];
    const axis = switch (options) {
        .softmax => |softmax| softmax.axis,
        else => return error.InvalidOpOptions,
    };
    const device = input.device;
    const input_layout_decision = kernel_capability.softmaxInputLayoutDecision(device, input.dtype, input.shape.dims, input.layout);
    try contracts.requireAxisInBounds(input.shape.rank(), axis, false);

    var shape = try Shape.initCopy(allocator, input.shape.dims);
    errdefer shape.deinit();
    var layout = try Layout.initContiguous(allocator, shape);
    errdefer layout.deinit();
    const axes = try cloneAxesForShape(allocator, input.axes, input.shape.rank());
    errdefer tensor.deinitAxes(allocator, axes);

    return .{
        .shape = shape,
        .dtype = input.dtype,
        .layout = layout,
        .axes = axes,
        .device = device,
        .kind = .reduction,
        .allocation = .new_storage,
        .input_requirement = .require_storage,
        .planner_hint = .{ .input_layout_decision = input_layout_decision },
    };
}

fn inferLogSoftmax(allocator: std.mem.Allocator, inputs: []const ValueSpec, options: OpOptions) !OpSpec {
    try contracts.requireInputCount(inputs, 1);
    try contracts.requireRank(inputs[0], .{ .range = .{ .min = 1, .max = 8 } });
    const input = inputs[0];
    const axis = switch (options) {
        .log_softmax => |log_softmax| log_softmax.axis,
        else => return error.InvalidOpOptions,
    };
    const device = input.device;
    const input_layout_decision = kernel_capability.logSoftmaxInputLayoutDecision(device, input.dtype, input.shape.dims, input.layout);
    try contracts.requireAxisInBounds(input.shape.rank(), axis, false);

    var shape = try Shape.initCopy(allocator, input.shape.dims);
    errdefer shape.deinit();
    var layout = try Layout.initContiguous(allocator, shape);
    errdefer layout.deinit();
    const axes = try cloneAxesForShape(allocator, input.axes, input.shape.rank());
    errdefer tensor.deinitAxes(allocator, axes);

    return .{
        .shape = shape,
        .dtype = input.dtype,
        .layout = layout,
        .axes = axes,
        .device = device,
        .kind = .reduction,
        .allocation = .new_storage,
        .input_requirement = .require_storage,
        .planner_hint = .{ .input_layout_decision = input_layout_decision },
    };
}

fn inferLogSoftmaxNll(allocator: std.mem.Allocator, inputs: []const ValueSpec, options: OpOptions) !OpSpec {
    const axis = switch (options) {
        .none => @as(usize, 1),
        .log_softmax_nll => |o| o.axis,
        else => return error.InvalidOpOptions,
    };
    return buildLogSoftmaxNllSpec(allocator, inputs, axis);
}

fn buildLogSoftmaxNllSpec(
    allocator: std.mem.Allocator,
    inputs: []const ValueSpec,
    axis: usize,
) !OpSpec {
    try contracts.requireInputCount(inputs, 2);
    try contracts.requireAllRanks(inputs, .{ .range = .{ .min = 2, .max = 8 } });
    const logits = inputs[0];
    const targets = inputs[1];
    if (logits.dtype != targets.dtype) return error.DTypeMismatch;
    if (logits.device != targets.device) return error.DeviceMismatch;
    if (!Shape.eql(logits.shape, targets.shape)) return error.ShapeMismatch;
    if (logits.dtype == .i64) return error.ExecutionNotImplemented;
    const input_layout_decision = kernel_capability.logSoftmaxNllInputLayoutDecision(
        logits.device,
        logits.dtype,
        logits.shape.dims,
        logits.layout,
        targets.shape.dims,
        targets.layout,
    );
    try contracts.requireAxisInBounds(logits.shape.rank(), axis, false);
    if (logits.shape.dims[axis] == 0) return error.InvalidAxis;

    var shape = try Shape.initCopy(allocator, &.{1});
    errdefer shape.deinit();
    var layout = try Layout.initContiguous(allocator, shape);
    errdefer layout.deinit();
    return .{
        .shape = shape,
        .dtype = logits.dtype,
        .layout = layout,
        .device = logits.device,
        .kind = .reduction_all,
        .allocation = .new_storage,
        .input_requirement = .require_storage,
        .planner_hint = .{ .input_layout_decision = input_layout_decision },
    };
}

fn inferCrossEntropyIndexed(allocator: std.mem.Allocator, inputs: []const ValueSpec, options: OpOptions) !OpSpec {
    try contracts.requireInputCount(inputs, 2);
    const logits = inputs[0];
    const targets = inputs[1];
    try contracts.requireRank(logits, .{ .exact = 2 });
    if (targets.shape.rank() != 1 and !(targets.shape.rank() == 2 and targets.shape.dims[1] == 1)) return error.ShapeMismatch;
    if (targets.shape.dims[0] != logits.shape.dims[0]) return error.ShapeMismatch;
    if (logits.device != targets.device) return error.DeviceMismatch;
    if (targets.dtype != .i64) return error.InvalidIndexDType;
    if (logits.dtype == .i64) return error.ExecutionNotImplemented;
    const input_layout_decision = kernel_capability.crossEntropyIndexedInputLayoutDecision(
        logits.device,
        logits.dtype,
        logits.shape.dims,
        logits.layout,
        targets.shape.dims,
        targets.layout,
    );
    const axis = switch (options) {
        .none => @as(usize, 1),
        .cross_entropy_indexed => |o| o.axis,
        else => return error.InvalidOpOptions,
    };
    if (axis != 1) return error.InvalidAxis;
    if (logits.shape.dims[1] == 0) return error.InvalidAxis;

    var shape = try Shape.initCopy(allocator, &.{1});
    errdefer shape.deinit();
    var layout = try Layout.initContiguous(allocator, shape);
    errdefer layout.deinit();
    return .{
        .shape = shape,
        .dtype = logits.dtype,
        .layout = layout,
        .device = logits.device,
        .kind = .reduction_all,
        .allocation = .new_storage,
        .input_requirement = .require_storage,
        .planner_hint = .{ .input_layout_decision = input_layout_decision },
    };
}

fn inferCrossEntropyIndexedBackward(allocator: std.mem.Allocator, inputs: []const ValueSpec, options: OpOptions) !OpSpec {
    try contracts.requireInputCount(inputs, 3);
    const logits = inputs[0];
    const targets = inputs[1];
    const grad_out = inputs[2];
    try contracts.requireRank(logits, .{ .exact = 2 });
    if (targets.shape.rank() != 1 and !(targets.shape.rank() == 2 and targets.shape.dims[1] == 1)) return error.ShapeMismatch;
    if (targets.shape.dims[0] != logits.shape.dims[0]) return error.ShapeMismatch;
    if (grad_out.shape.rank() != 1 or grad_out.shape.dims[0] != 1) return error.ShapeMismatch;
    if (logits.device != targets.device or logits.device != grad_out.device) return error.DeviceMismatch;
    if (targets.dtype != .i64) return error.InvalidIndexDType;
    if (logits.dtype == .i64 or grad_out.dtype != logits.dtype) return error.ExecutionNotImplemented;
    const input_layout_decision = kernel_capability.crossEntropyIndexedBackwardInputLayoutDecision(
        logits.device,
        logits.dtype,
        logits.shape.dims,
        logits.layout,
        targets.shape.dims,
        targets.layout,
        grad_out.shape.dims,
        grad_out.layout,
    );
    const axis = switch (options) {
        .none => @as(usize, 1),
        .cross_entropy_indexed_backward => |o| o.axis,
        else => return error.InvalidOpOptions,
    };
    if (axis != 1) return error.InvalidAxis;
    if (logits.shape.dims[1] == 0) return error.InvalidAxis;

    var shape = try Shape.initCopy(allocator, logits.shape.dims);
    errdefer shape.deinit();
    var layout = try Layout.initContiguous(allocator, shape);
    errdefer layout.deinit();
    const axes = try cloneAxesForShape(allocator, logits.axes, logits.shape.rank());
    errdefer tensor.deinitAxes(allocator, axes);
    return .{
        .shape = shape,
        .dtype = logits.dtype,
        .layout = layout,
        .axes = axes,
        .device = logits.device,
        .kind = .elementwise_generic,
        .allocation = .new_storage,
        .input_requirement = .require_storage,
        .planner_hint = .{ .input_layout_decision = input_layout_decision },
    };
}

fn inferCrossEntropy(allocator: std.mem.Allocator, inputs: []const ValueSpec, options: OpOptions) !OpSpec {
    const axis = switch (options) {
        .none => @as(usize, 1),
        .cross_entropy => |o| o.axis,
        else => return error.InvalidOpOptions,
    };
    return buildLogSoftmaxNllSpec(allocator, inputs, axis);
}

fn inferSumAll(allocator: std.mem.Allocator, inputs: []const ValueSpec, options: OpOptions) !OpSpec {
    return buildReduceAllSpec(allocator, inputs, options, null);
}

fn inferMeanAll(allocator: std.mem.Allocator, inputs: []const ValueSpec, options: OpOptions) !OpSpec {
    return buildReduceAllSpec(allocator, inputs, options, null);
}

fn inferMinAll(allocator: std.mem.Allocator, inputs: []const ValueSpec, options: OpOptions) !OpSpec {
    return buildReduceAllSpec(allocator, inputs, options, null);
}

fn inferMaxAll(allocator: std.mem.Allocator, inputs: []const ValueSpec, options: OpOptions) !OpSpec {
    return buildReduceAllSpec(allocator, inputs, options, null);
}

fn inferVarianceAll(allocator: std.mem.Allocator, inputs: []const ValueSpec, options: OpOptions) !OpSpec {
    return buildReduceAllSpec(allocator, inputs, options, null);
}

fn inferStdAll(allocator: std.mem.Allocator, inputs: []const ValueSpec, options: OpOptions) !OpSpec {
    return buildReduceAllSpec(allocator, inputs, options, null);
}

fn inferArgminAll(allocator: std.mem.Allocator, inputs: []const ValueSpec, options: OpOptions) !OpSpec {
    return buildReduceAllSpec(allocator, inputs, options, .i64);
}

fn inferArgmaxAll(allocator: std.mem.Allocator, inputs: []const ValueSpec, options: OpOptions) !OpSpec {
    return buildReduceAllSpec(allocator, inputs, options, .i64);
}

fn buildReduceAllSpec(
    allocator: std.mem.Allocator,
    inputs: []const ValueSpec,
    options: OpOptions,
    output_dtype: ?DType,
) !OpSpec {
    try contracts.requireInputCount(inputs, 1);
    try contracts.requireRank(inputs[0], .{ .range = .{ .min = 1, .max = 8 } });
    const input = inputs[0];
    const device = input.device;
    const input_layout_decision = kernel_capability.reductionAllInputLayoutDecision(device, input.dtype, input.shape.dims, input.layout);
    const keepdim = switch (options) {
        .none => false,
        .reduce_all => |reduce| reduce.keepdim,
        else => return error.InvalidOpOptions,
    };

    var shape = if (keepdim)
        try shapeOfOnes(allocator, input.shape.rank())
    else
        try Shape.initCopy(allocator, &.{1});
    errdefer shape.deinit();
    var layout = try Layout.initContiguous(allocator, shape);
    errdefer layout.deinit();

    return .{
        .shape = shape,
        .dtype = output_dtype orelse input.dtype,
        .layout = layout,
        .device = device,
        .kind = .reduction_all,
        .allocation = .new_storage,
        .input_requirement = .require_storage,
        .planner_hint = .{ .input_layout_decision = input_layout_decision },
    };
}

fn inferSumAxis(allocator: std.mem.Allocator, inputs: []const ValueSpec, options: OpOptions) !OpSpec {
    return buildReduceAxisSpec(allocator, inputs, options, null);
}

fn inferMeanAxis(allocator: std.mem.Allocator, inputs: []const ValueSpec, options: OpOptions) !OpSpec {
    return buildReduceAxisSpec(allocator, inputs, options, null);
}

fn inferMinAxis(allocator: std.mem.Allocator, inputs: []const ValueSpec, options: OpOptions) !OpSpec {
    return buildReduceAxisSpec(allocator, inputs, options, null);
}

fn inferMaxAxis(allocator: std.mem.Allocator, inputs: []const ValueSpec, options: OpOptions) !OpSpec {
    return buildReduceAxisSpec(allocator, inputs, options, null);
}

fn inferVarianceAxis(allocator: std.mem.Allocator, inputs: []const ValueSpec, options: OpOptions) !OpSpec {
    return buildReduceAxisSpec(allocator, inputs, options, null);
}

fn inferStdAxis(allocator: std.mem.Allocator, inputs: []const ValueSpec, options: OpOptions) !OpSpec {
    return buildReduceAxisSpec(allocator, inputs, options, null);
}

fn inferArgminAxis(allocator: std.mem.Allocator, inputs: []const ValueSpec, options: OpOptions) !OpSpec {
    return buildReduceAxisSpec(allocator, inputs, options, .i64);
}

fn inferArgmaxAxis(allocator: std.mem.Allocator, inputs: []const ValueSpec, options: OpOptions) !OpSpec {
    return buildReduceAxisSpec(allocator, inputs, options, .i64);
}

fn buildReduceAxisSpec(
    allocator: std.mem.Allocator,
    inputs: []const ValueSpec,
    options: OpOptions,
    output_dtype: ?DType,
) !OpSpec {
    try contracts.requireInputCount(inputs, 1);
    try contracts.requireRank(inputs[0], .{ .range = .{ .min = 1, .max = 8 } });
    const input = inputs[0];
    const device = input.device;
    const input_layout_decision = kernel_capability.reductionAxisInputLayoutDecision(device, input.dtype, input.shape.dims, input.layout);
    const reduce = switch (options) {
        .reduce_axis => |axis_options| axis_options,
        else => return error.InvalidOpOptions,
    };
    try contracts.requireAxisInBounds(input.shape.rank(), reduce.axis, false);

    const rank = input.shape.rank();
    const out_rank = if (reduce.keepdim) rank else rank - 1;
    const dims = try allocator.alloc(usize, out_rank);
    defer allocator.free(dims);

    var out_i: usize = 0;
    for (input.shape.dims, 0..) |dim, i| {
        if (i == reduce.axis) {
            if (reduce.keepdim) {
                dims[out_i] = 1;
                out_i += 1;
            }
            continue;
        }
        dims[out_i] = dim;
        out_i += 1;
    }

    var shape = try Shape.initCopy(allocator, dims);
    errdefer shape.deinit();
    var layout = try Layout.initContiguous(allocator, shape);
    errdefer layout.deinit();
    const axes = try reduceAxisNames(allocator, input.axes, reduce.axis, reduce.keepdim);
    errdefer tensor.deinitAxes(allocator, axes);

    return .{
        .shape = shape,
        .dtype = output_dtype orelse input.dtype,
        .layout = layout,
        .axes = axes,
        .device = device,
        .kind = .reduction,
        .allocation = .new_storage,
        .input_requirement = .require_storage,
        .planner_hint = .{ .input_layout_decision = input_layout_decision },
    };
}

fn inferCat(allocator: std.mem.Allocator, inputs: []const ValueSpec, options: OpOptions) !OpSpec {
    try contracts.requireMinInputCount(inputs, 1);
    try contracts.requireSameRank(inputs);
    try contracts.requireAllRanks(inputs, .{ .range = .{ .min = 1, .max = 8 } });
    const axis = switch (options) {
        .none => @as(usize, 0),
        .concat => |concat| concat.axis,
        else => return error.InvalidOpOptions,
    };
    const first = inputs[0];
    const device = first.device;
    try contracts.requireAxisInBounds(first.shape.rank(), axis, false);
    const input_shapes = try allocator.alloc([]const usize, inputs.len);
    defer allocator.free(input_shapes);
    const input_layouts = try allocator.alloc(Layout, inputs.len);
    defer allocator.free(input_layouts);
    for (inputs, 0..) |input, i| {
        input_shapes[i] = input.shape.dims;
        input_layouts[i] = input.layout;
    }
    const input_layout_decision = kernel_capability.catInputLayoutDecision(device, first.dtype, input_shapes, input_layouts);

    const dims = try allocator.alloc(usize, first.shape.rank());
    defer allocator.free(dims);
    @memcpy(dims, first.shape.dims);

    var axis_total = first.shape.dims[axis];
    for (inputs[1..]) |input| {
        if (input.dtype != first.dtype) return error.DTypeMismatch;
        if ((input.device) != device) return error.DeviceMismatch;
        if (input.shape.rank() != first.shape.rank()) return error.ShapeMismatch;
        for (input.shape.dims, 0..) |dim, i| {
            if (i == axis) continue;
            if (dim != first.shape.dims[i]) return error.ShapeMismatch;
        }
        axis_total += input.shape.dims[axis];
    }
    dims[axis] = axis_total;

    var shape = try Shape.initCopy(allocator, dims);
    errdefer shape.deinit();
    var layout = try Layout.initContiguous(allocator, shape);
    errdefer layout.deinit();
    const axes = try matchingInputAxes(allocator, inputs, first.shape.rank());
    errdefer tensor.deinitAxes(allocator, axes);

    return .{
        .shape = shape,
        .dtype = first.dtype,
        .layout = layout,
        .axes = axes,
        .device = device,
        .kind = .elementwise_generic,
        .allocation = .new_storage,
        .input_requirement = .require_storage,
        .planner_hint = .{ .input_layout_decision = input_layout_decision },
    };
}

fn inferStack(allocator: std.mem.Allocator, inputs: []const ValueSpec, options: OpOptions) !OpSpec {
    try contracts.requireMinInputCount(inputs, 1);
    try contracts.requireSameRank(inputs);
    try contracts.requireAllRanks(inputs, .{ .range = .{ .min = 0, .max = 8 } });
    const axis = switch (options) {
        .none => @as(usize, 0),
        .stack => |stack| stack.axis,
        else => return error.InvalidOpOptions,
    };
    const first = inputs[0];
    const device = first.device;
    try contracts.requireAxisInBounds(first.shape.rank(), axis, true);
    const input_shapes = try allocator.alloc([]const usize, inputs.len);
    defer allocator.free(input_shapes);
    const input_layouts = try allocator.alloc(Layout, inputs.len);
    defer allocator.free(input_layouts);
    for (inputs, 0..) |input, i| {
        input_shapes[i] = input.shape.dims;
        input_layouts[i] = input.layout;
    }
    const input_layout_decision = kernel_capability.stackInputLayoutDecision(device, first.dtype, input_shapes, input_layouts);

    for (inputs[1..]) |input| {
        if (input.dtype != first.dtype) return error.DTypeMismatch;
        if ((input.device) != device) return error.DeviceMismatch;
        if (!Shape.eql(input.shape, first.shape)) return error.ShapeMismatch;
    }

    const dims = try allocator.alloc(usize, first.shape.rank() + 1);
    defer allocator.free(dims);
    var src_i: usize = 0;
    for (0..dims.len) |i| {
        if (i == axis) {
            dims[i] = inputs.len;
        } else {
            dims[i] = first.shape.dims[src_i];
            src_i += 1;
        }
    }

    var shape = try Shape.initCopy(allocator, dims);
    errdefer shape.deinit();
    var layout = try Layout.initContiguous(allocator, shape);
    errdefer layout.deinit();

    return .{
        .shape = shape,
        .dtype = first.dtype,
        .layout = layout,
        .device = device,
        .kind = .elementwise_generic,
        .allocation = .new_storage,
        .input_requirement = .require_storage,
        .planner_hint = .{ .input_layout_decision = input_layout_decision },
    };
}

fn inferContiguous(allocator: std.mem.Allocator, inputs: []const ValueSpec) !OpSpec {
    try contracts.requireInputCount(inputs, 1);
    try contracts.requireRank(inputs[0], .{ .range = .{ .min = 0, .max = 8 } });
    const input = inputs[0];
    const device = input.device;

    var shape = try Shape.initCopy(allocator, input.shape.dims);
    errdefer shape.deinit();
    var layout = try Layout.initContiguous(allocator, shape);
    errdefer layout.deinit();
    const axes = try cloneAxesForShape(allocator, input.axes, input.shape.rank());
    errdefer tensor.deinitAxes(allocator, axes);

    const aliases_input = input.layout.offset == 0 and input.layout.isContiguous(input.shape);

    return .{
        .shape = shape,
        .dtype = input.dtype,
        .layout = layout,
        .axes = axes,
        .device = device,
        .kind = if (aliases_input) .view else .elementwise_generic,
        .allocation = if (aliases_input) .view_only else .new_storage,
        .input_requirement = .require_storage,
    };
}

fn inferReshape(allocator: std.mem.Allocator, inputs: []const ValueSpec, options: OpOptions) !OpSpec {
    try contracts.requireInputCount(inputs, 1);
    try contracts.requireRank(inputs[0], .{ .range = .{ .min = 0, .max = 8 } });
    const input = inputs[0];
    const device = input.device;
    const target_shape = switch (options) {
        .reshape => |reshape| reshape.shape,
        else => return error.InvalidOpOptions,
    };

    var shape = try Shape.initCopy(allocator, target_shape);
    errdefer shape.deinit();
    if (shape.numel() != input.shape.numel()) return error.ShapeMismatch;

    var layout = try Layout.initContiguous(allocator, shape);
    errdefer layout.deinit();
    const axes = if (Shape.eql(input.shape, shape))
        try cloneAxesForShape(allocator, input.axes, input.shape.rank())
    else
        null;
    errdefer tensor.deinitAxes(allocator, axes);

    return .{
        .shape = shape,
        .dtype = input.dtype,
        .layout = layout,
        .axes = axes,
        .device = device,
        .kind = .view,
        .allocation = .view_only,
        .input_requirement = .require_contiguous_input,
    };
}

fn inferSlice(allocator: std.mem.Allocator, inputs: []const ValueSpec, options: OpOptions) !OpSpec {
    try contracts.requireInputCount(inputs, 1);
    try contracts.requireRank(inputs[0], .{ .range = .{ .min = 1, .max = 8 } });
    const input = inputs[0];
    const device = input.device;
    const slice = switch (options) {
        .slice => |slice_options| slice_options,
        else => return error.InvalidOpOptions,
    };
    if (slice.ranges.len > input.shape.rank()) return error.TooManySliceDimensions;

    const rank = input.shape.rank();
    const dims = try allocator.alloc(usize, rank);
    defer allocator.free(dims);
    const strides = try allocator.alloc(isize, rank);
    defer allocator.free(strides);

    var offset_elements: usize = input.layout.offset;
    var needs_copy = false;
    for (0..rank) |i| {
        const range = if (i < slice.ranges.len) slice.ranges[i] else SliceRange{
            .start = 0,
            .stop = input.shape.dims[i],
            .step = 1,
        };
        const dim_size = input.shape.dims[i];
        validateSliceRange(range, dim_size) catch return error.InvalidSlice;
        dims[i] = computeSliceLength(range.start, range.stop, range.step);
        if (range.step != 1) needs_copy = true;
        if (!needs_copy) {
            offset_elements += range.start * @as(usize, @intCast(input.layout.strides[i]));
            strides[i] = input.layout.strides[i];
        } else {
            strides[i] = input.layout.strides[i] * range.step;
        }
    }

    var shape = try Shape.initCopy(allocator, dims);
    errdefer shape.deinit();
    const needs_copy_materialization = needsCopyMaterialization(slice.ranges);
    var layout = if (needsCopyMaterialization(slice.ranges))
        try Layout.initContiguous(allocator, shape)
    else
        try Layout.initCopy(allocator, strides, offset_elements);
    errdefer layout.deinit();
    const axes = try cloneAxesForShape(allocator, input.axes, input.shape.rank());
    errdefer tensor.deinitAxes(allocator, axes);

    return .{
        .shape = shape,
        .dtype = input.dtype,
        .layout = layout,
        .axes = axes,
        .device = device,
        .kind = if (needs_copy) .elementwise_generic else .view,
        .allocation = if (needs_copy) .new_storage else .view_only,
        .input_requirement = if (needs_copy) .require_storage else .preserve,
        .planner_hint = if (needs_copy_materialization)
            .{ .input_layout_decision = kernel_capability.sliceInputLayoutDecision(device, input.dtype, input.shape.dims, input.layout) }
        else
            null,
    };
}

fn inferGather(allocator: std.mem.Allocator, inputs: []const ValueSpec, options: OpOptions) !OpSpec {
    try contracts.requireInputCount(inputs, 2);
    try contracts.requireSameRank(inputs);
    try contracts.requireAllRanks(inputs, .{ .range = .{ .min = 1, .max = 8 } });
    const input = inputs[0];
    const index = inputs[1];
    const axis = switch (options) {
        .gather => |gather| gather.axis,
        else => return error.InvalidOpOptions,
    };
    if (index.dtype != .i64) return error.InvalidIndexDType;

    const device = input.device;
    if ((index.device) != device) return error.DeviceMismatch;
    try contracts.requireAxisInBounds(input.shape.rank(), axis, false);
    const input_layout_decision = kernel_capability.gatherInputLayoutDecision(
        device,
        input.dtype,
        input.shape.dims,
        input.layout,
        index.shape.dims,
        index.layout,
    );

    for (input.shape.dims, 0..) |dim, i| {
        if (i == axis) continue;
        if (index.shape.dims[i] != dim) return error.ShapeMismatch;
    }

    var shape = try Shape.initCopy(allocator, index.shape.dims);
    errdefer shape.deinit();
    var layout = try Layout.initContiguous(allocator, shape);
    errdefer layout.deinit();
    const axes = try gatherAxes(allocator, input, index, axis);
    errdefer tensor.deinitAxes(allocator, axes);

    return .{
        .shape = shape,
        .dtype = input.dtype,
        .layout = layout,
        .axes = axes,
        .device = device,
        .kind = .index,
        .allocation = .new_storage,
        .input_requirement = .require_storage,
        .planner_hint = .{ .input_layout_decision = input_layout_decision },
    };
}

fn inferIndexSelect(allocator: std.mem.Allocator, inputs: []const ValueSpec, options: OpOptions) !OpSpec {
    try contracts.requireInputCount(inputs, 2);
    try contracts.requireRank(inputs[0], .{ .range = .{ .min = 1, .max = 8 } });
    try contracts.requireRank(inputs[1], .{ .exact = 1 });
    const input = inputs[0];
    const index = inputs[1];
    const axis = switch (options) {
        .index_select => |index_select| index_select.axis,
        else => return error.InvalidOpOptions,
    };
    if (index.dtype != .i64) return error.InvalidIndexDType;

    const device = input.device;
    if ((index.device) != device) return error.DeviceMismatch;
    try contracts.requireAxisInBounds(input.shape.rank(), axis, false);
    const input_layout_decision = kernel_capability.indexSelectInputLayoutDecision(
        device,
        input.dtype,
        input.shape.dims,
        input.layout,
        index.shape.dims,
        index.layout,
    );

    const dims = try allocator.alloc(usize, input.shape.rank());
    defer allocator.free(dims);
    @memcpy(dims, input.shape.dims);
    dims[axis] = index.shape.dims[0];

    var shape = try Shape.initCopy(allocator, dims);
    errdefer shape.deinit();
    var layout = try Layout.initContiguous(allocator, shape);
    errdefer layout.deinit();
    const axes = try indexSelectAxes(allocator, input, index, axis);
    errdefer tensor.deinitAxes(allocator, axes);

    return .{
        .shape = shape,
        .dtype = input.dtype,
        .layout = layout,
        .axes = axes,
        .device = device,
        .kind = .index,
        .allocation = .new_storage,
        .input_requirement = .require_storage,
        .planner_hint = .{ .input_layout_decision = input_layout_decision },
    };
}

fn inferScatterAdd(allocator: std.mem.Allocator, inputs: []const ValueSpec, options: OpOptions) !OpSpec {
    try contracts.requireInputCount(inputs, 3);
    try contracts.requireAllRanks(inputs, .{ .range = .{ .min = 1, .max = 8 } });
    const base = inputs[0];
    const index = inputs[1];
    const updates = inputs[2];
    const axis = switch (options) {
        .scatter_add => |scatter_add| scatter_add.axis,
        else => return error.InvalidOpOptions,
    };
    if (base.dtype != updates.dtype) return error.DTypeMismatch;
    if (index.dtype != .i64) return error.InvalidIndexDType;
    if (index.device != base.device or updates.device != base.device) return error.DeviceMismatch;
    if (base.shape.rank() != index.shape.rank() or base.shape.rank() != updates.shape.rank()) return error.ShapeMismatch;
    try contracts.requireAxisInBounds(base.shape.rank(), axis, false);
    const input_layout_decision = kernel_capability.scatterAddInputLayoutDecision(
        base.device,
        base.dtype,
        base.shape.dims,
        base.layout,
        index.shape.dims,
        index.layout,
        updates.shape.dims,
        updates.layout,
    );
    if (!Shape.eql(index.shape, updates.shape)) return error.ShapeMismatch;
    for (base.shape.dims, index.shape.dims, 0..) |bd, id, d| {
        if (d == axis) continue;
        if (bd != id) return error.ShapeMismatch;
    }

    var shape = try Shape.initCopy(allocator, base.shape.dims);
    errdefer shape.deinit();
    var layout = try Layout.initContiguous(allocator, shape);
    errdefer layout.deinit();
    const axes = try cloneAxesForShape(allocator, base.axes, base.shape.rank());
    errdefer tensor.deinitAxes(allocator, axes);
    return .{
        .shape = shape,
        .dtype = base.dtype,
        .layout = layout,
        .axes = axes,
        .device = base.device,
        .kind = .index,
        .allocation = .new_storage,
        .input_requirement = .require_storage,
        .planner_hint = .{ .input_layout_decision = input_layout_decision },
    };
}

fn inferReduceToShape(allocator: std.mem.Allocator, inputs: []const ValueSpec, options: OpOptions) !OpSpec {
    try contracts.requireInputCount(inputs, 1);
    try contracts.requireRank(inputs[0], .{ .range = .{ .min = 1, .max = 8 } });
    const input = inputs[0];
    const target = switch (options) {
        .reduce_to_shape => |opt| opt.shape,
        else => return error.InvalidOpOptions,
    };
    const input_layout_decision = kernel_capability.reduceToShapeInputLayoutDecision(input.device, input.dtype, input.shape.dims, input.layout);
    try validateReduceToShape(input.shape.dims, target);

    var shape = try Shape.initCopy(allocator, target);
    errdefer shape.deinit();
    var layout = try Layout.initContiguous(allocator, shape);
    errdefer layout.deinit();
    return .{
        .shape = shape,
        .dtype = input.dtype,
        .layout = layout,
        .device = input.device,
        .kind = .reduction,
        .allocation = .new_storage,
        .input_requirement = .require_storage,
        .reduce_to_shape = try inferReduceToShapeSpec(input.shape.dims, target),
        .planner_hint = .{ .input_layout_decision = input_layout_decision },
    };
}

fn inferEmbedding(allocator: std.mem.Allocator, inputs: []const ValueSpec, options: OpOptions) !OpSpec {
    try contracts.requireInputCount(inputs, 2);
    try contracts.requireRank(inputs[0], .{ .range = .{ .min = 2, .max = 8 } });
    try contracts.requireRank(inputs[1], .{ .range = .{ .min = 0, .max = 7 } });
    switch (options) {
        .none, .embedding => {},
        else => return error.InvalidOpOptions,
    }
    const table = inputs[0];
    const index = inputs[1];
    if (index.dtype != .i64) return error.InvalidIndexDType;
    if (table.device != index.device) return error.DeviceMismatch;
    const input_layout_decision = kernel_capability.embeddingInputLayoutDecision(
        table.device,
        table.dtype,
        table.shape.dims,
        table.layout,
        index.shape.dims,
        index.layout,
    );

    const out_rank = index.shape.rank() + table.shape.rank() - 1;
    if (out_rank > 8) return error.ShapeMismatch;
    const dims = try allocator.alloc(usize, out_rank);
    defer allocator.free(dims);
    if (index.shape.rank() > 0) @memcpy(dims[0..index.shape.rank()], index.shape.dims);
    if (table.shape.rank() > 1) {
        @memcpy(dims[index.shape.rank()..], table.shape.dims[1..]);
    }

    var shape = try Shape.initCopy(allocator, dims);
    errdefer shape.deinit();
    var layout = try Layout.initContiguous(allocator, shape);
    errdefer layout.deinit();
    const axes = try embeddingAxes(allocator, table, index);
    errdefer tensor.deinitAxes(allocator, axes);
    return .{
        .shape = shape,
        .dtype = table.dtype,
        .layout = layout,
        .axes = axes,
        .device = table.device,
        .kind = .index,
        .allocation = .new_storage,
        .input_requirement = .require_storage,
        .planner_hint = .{ .input_layout_decision = input_layout_decision },
    };
}

fn inferOneHot(allocator: std.mem.Allocator, inputs: []const ValueSpec, options: OpOptions) !OpSpec {
    try contracts.requireInputCount(inputs, 1);
    try contracts.requireRank(inputs[0], .{ .range = .{ .min = 0, .max = 7 } });
    const input = inputs[0];
    const one_hot = switch (options) {
        .one_hot => |v| v,
        else => return error.InvalidOpOptions,
    };
    if (one_hot.num_classes == 0) return error.InvalidClassCount;
    if (input.dtype != .i64) return error.InvalidIndexDType;
    const device = input.device;
    const input_layout_decision = kernel_capability.oneHotInputLayoutDecision(device, input.shape.dims, input.layout);

    const out_rank = input.shape.rank() + 1;
    const dims = try allocator.alloc(usize, out_rank);
    defer allocator.free(dims);
    if (input.shape.rank() > 0) @memcpy(dims[0..input.shape.rank()], input.shape.dims);
    dims[out_rank - 1] = one_hot.num_classes;

    var shape = try Shape.initCopy(allocator, dims);
    errdefer shape.deinit();
    var layout = try Layout.initContiguous(allocator, shape);
    errdefer layout.deinit();

    return .{
        .shape = shape,
        .dtype = .f32,
        .layout = layout,
        .device = device,
        .kind = .index,
        .allocation = .new_storage,
        .input_requirement = .require_storage,
        .planner_hint = .{ .input_layout_decision = input_layout_decision },
    };
}

fn inferTopK(allocator: std.mem.Allocator, inputs: []const ValueSpec, options: OpOptions) !OpSpec {
    try contracts.requireInputCount(inputs, 1);
    try contracts.requireRank(inputs[0], .{ .range = .{ .min = 1, .max = 8 } });
    const input = inputs[0];
    const topk = switch (options) {
        .topk => |v| v,
        else => return error.InvalidOpOptions,
    };
    if (topk.k == 0) return error.InvalidTopK;
    try contracts.requireAxisInBounds(input.shape.rank(), topk.axis, false);
    if (topk.k > input.shape.dims[topk.axis]) return error.InvalidTopK;
    const device = input.device;
    const input_layout_decision = kernel_capability.topKInputLayoutDecision(device, input.dtype, input.shape.dims, input.layout);

    const dims = try allocator.alloc(usize, input.shape.rank());
    defer allocator.free(dims);
    @memcpy(dims, input.shape.dims);
    dims[topk.axis] = topk.k;

    var shape = try Shape.initCopy(allocator, dims);
    errdefer shape.deinit();
    var layout = try Layout.initContiguous(allocator, shape);
    errdefer layout.deinit();

    var indices_shape = try Shape.initCopy(allocator, dims);
    errdefer indices_shape.deinit();
    var indices_layout = try Layout.initContiguous(allocator, indices_shape);
    errdefer indices_layout.deinit();

    return .{
        .shape = shape,
        .dtype = input.dtype,
        .layout = layout,
        .device = device,
        .kind = .index,
        .allocation = .new_storage,
        .input_requirement = .require_storage,
        .planner_hint = .{ .input_layout_decision = input_layout_decision },
        .secondary_output = .{
            .shape = indices_shape,
            .dtype = .i64,
            .layout = indices_layout,
        },
    };
}

fn inferDot(allocator: std.mem.Allocator, inputs: []const ValueSpec) !OpSpec {
    try contracts.requireInputCount(inputs, 2);
    try contracts.requireRank(inputs[0], .{ .exact = 1 });
    try contracts.requireRank(inputs[1], .{ .exact = 1 });
    const lhs = inputs[0];
    const rhs = inputs[1];
    if (lhs.dtype != rhs.dtype) return error.DTypeMismatch;
    if (lhs.shape.dims[0] != rhs.shape.dims[0]) return error.ShapeMismatch;
    const device = lhs.device;
    if ((rhs.device) != device) return error.DeviceMismatch;
    const input_layout_decision = kernel_capability.dotInputLayoutDecision(device, lhs.dtype, lhs.shape.dims, lhs.layout, rhs.shape.dims, rhs.layout);

    var shape = try Shape.initCopy(allocator, &.{});
    errdefer shape.deinit();
    var layout = try Layout.initContiguous(allocator, shape);
    errdefer layout.deinit();

    return .{
        .shape = shape,
        .dtype = lhs.dtype,
        .layout = layout,
        .device = device,
        .kind = .reduction,
        .allocation = .new_storage,
        .input_requirement = .preserve,
        .planner_hint = .{ .input_layout_decision = input_layout_decision },
    };
}

fn inferMatmul(allocator: std.mem.Allocator, inputs: []const ValueSpec) !OpSpec {
    try contracts.requireInputCount(inputs, 2);
    try contracts.requireRank(inputs[0], .{ .range = .{ .min = 1, .max = 8 } });
    try contracts.requireRank(inputs[1], .{ .range = .{ .min = 1, .max = 8 } });
    const lhs = inputs[0];
    const rhs = inputs[1];
    if (lhs.dtype != rhs.dtype) return error.DTypeMismatch;
    const device = lhs.device;
    if ((rhs.device) != device) return error.DeviceMismatch;

    const out_dims = try inferMatmulOutputShape(allocator, lhs.shape.dims, rhs.shape.dims);
    defer allocator.free(out_dims);
    var shape = try Shape.initCopy(allocator, out_dims);
    errdefer shape.deinit();
    var layout = try Layout.initContiguous(allocator, shape);
    errdefer layout.deinit();
    const axes = try matmulAxes(allocator, lhs, rhs, shape);
    errdefer tensor.deinitAxes(allocator, axes);

    return .{
        .shape = shape,
        .dtype = lhs.dtype,
        .layout = layout,
        .axes = axes,
        .device = device,
        .kind = .reduction,
        .allocation = .new_storage,
        .input_requirement = .preserve,
        .planner_hint = .{
            .input_layout_decision = kernel_capability.matmulInputLayoutDecision(
                device,
                lhs.dtype,
                lhs.shape.dims,
                lhs.layout,
                rhs.shape.dims,
                rhs.layout,
            ),
        },
    };
}

fn inferMatmulOutputShape(allocator: std.mem.Allocator, lhs: []const usize, rhs: []const usize) ![]usize {
    const lhs_rank = lhs.len;
    const rhs_rank = rhs.len;

    // 1D @ 1D -> scalar
    if (lhs_rank == 1 and rhs_rank == 1) {
        if (lhs[0] != rhs[0]) return error.ShapeMismatch;
        return allocator.alloc(usize, 0);
    }

    // Normalize operands to matrix semantics, then apply broadcasted batched matmul.
    const lhs_m = if (lhs_rank == 1) @as(usize, 1) else lhs[lhs_rank - 2];
    const lhs_k = lhs[lhs_rank - 1];
    const rhs_k = if (rhs_rank == 1) rhs[0] else rhs[rhs_rank - 2];
    const rhs_n = if (rhs_rank == 1) @as(usize, 1) else rhs[rhs_rank - 1];
    if (lhs_k != rhs_k) return error.ShapeMismatch;

    const lhs_batch = if (lhs_rank <= 2) lhs[0..0] else lhs[0 .. lhs_rank - 2];
    const rhs_batch = if (rhs_rank <= 2) rhs[0..0] else rhs[0 .. rhs_rank - 2];
    const batch = try inferBroadcastShape(allocator, lhs_batch, rhs_batch);
    defer allocator.free(batch);

    // Output matrix rank before squeezing implicit dims from 1D operands.
    const base_len = batch.len + 2;
    const base = try allocator.alloc(usize, base_len);
    defer allocator.free(base);
    @memcpy(base[0..batch.len], batch);
    base[base_len - 2] = lhs_m;
    base[base_len - 1] = rhs_n;

    // 1D @ ND removes leading implicit 1; ND @ 1D removes trailing implicit 1.
    if (lhs_rank == 1 and rhs_rank > 1) {
        const out = try allocator.alloc(usize, base_len - 1);
        if (base_len > 1) @memcpy(out, base[1..]);
        return out;
    }
    if (lhs_rank > 1 and rhs_rank == 1) {
        const out = try allocator.alloc(usize, base_len - 1);
        if (base_len > 1) @memcpy(out, base[0 .. base_len - 1]);
        return out;
    }

    const out = try allocator.alloc(usize, base_len);
    @memcpy(out, base);
    return out;
}

fn inferPermute(allocator: std.mem.Allocator, inputs: []const ValueSpec, options: OpOptions) !OpSpec {
    try contracts.requireInputCount(inputs, 1);
    try contracts.requireRank(inputs[0], .{ .range = .{ .min = 1, .max = 8 } });
    const axes = switch (options) {
        .permute => |permute| permute.axes,
        else => return error.InvalidOpOptions,
    };
    return inferPermutationView(allocator, inputs[0], axes);
}

fn inferTranspose(allocator: std.mem.Allocator, inputs: []const ValueSpec, options: OpOptions) !OpSpec {
    try contracts.requireInputCount(inputs, 1);
    const input = inputs[0];
    try contracts.requireRank(input, .{ .range = .{ .min = 1, .max = 8 } });
    const rank = input.shape.rank();
    const permutation = switch (options) {
        .none => null,
        .transpose => |transpose| transpose.permutation,
        else => return error.InvalidOpOptions,
    };

    const order = permutation orelse try reversePermutation(allocator, rank);
    defer if (permutation == null) allocator.free(order);
    return inferPermutationView(allocator, input, order);
}

fn inferSqueeze(allocator: std.mem.Allocator, inputs: []const ValueSpec, options: OpOptions) !OpSpec {
    try contracts.requireInputCount(inputs, 1);
    try contracts.requireRank(inputs[0], .{ .range = .{ .min = 1, .max = 8 } });
    const input = inputs[0];
    const device = input.device;
    const axis = switch (options) {
        .none => null,
        .squeeze => |squeeze| squeeze.axis,
        else => return error.InvalidOpOptions,
    };

    if (axis) |selected| {
        try contracts.requireAxisInBounds(input.shape.rank(), selected, false);
        if (input.shape.dims[selected] != 1) return error.DimensionNotSingleton;
    }

    var kept: usize = 0;
    for (input.shape.dims, 0..) |dim, i| {
        if (axis) |selected| {
            if (i == selected) continue;
        } else if (dim == 1) {
            continue;
        }
        kept += 1;
    }

    const dims = try allocator.alloc(usize, kept);
    defer allocator.free(dims);
    const strides = try allocator.alloc(isize, kept);
    defer allocator.free(strides);

    var out_i: usize = 0;
    for (input.shape.dims, 0..) |dim, i| {
        if (axis) |selected| {
            if (i == selected) continue;
        } else if (dim == 1) {
            continue;
        }
        dims[out_i] = dim;
        strides[out_i] = input.layout.strides[i];
        out_i += 1;
    }

    var shape = try Shape.initCopy(allocator, dims);
    errdefer shape.deinit();
    var layout = try Layout.initCopy(allocator, strides, input.layout.offset);
    errdefer layout.deinit();
    const axes = try squeezeAxisNames(allocator, input.axes, input.shape.dims, axis);
    errdefer tensor.deinitAxes(allocator, axes);

    return .{
        .shape = shape,
        .dtype = input.dtype,
        .layout = layout,
        .axes = axes,
        .device = device,
        .kind = .view,
        .allocation = .view_only,
        .input_requirement = .preserve,
    };
}

fn inferUnsqueeze(allocator: std.mem.Allocator, inputs: []const ValueSpec, options: OpOptions) !OpSpec {
    try contracts.requireInputCount(inputs, 1);
    try contracts.requireRank(inputs[0], .{ .range = .{ .min = 0, .max = 7 } });
    const input = inputs[0];
    const device = input.device;
    const axis = switch (options) {
        .unsqueeze => |unsqueeze| unsqueeze.axis,
        else => return error.InvalidOpOptions,
    };
    try contracts.requireAxisInBounds(input.shape.rank(), axis, true);

    const new_rank = input.shape.rank() + 1;
    const dims = try allocator.alloc(usize, new_rank);
    defer allocator.free(dims);
    const strides = try allocator.alloc(isize, new_rank);
    defer allocator.free(strides);

    var src_i: usize = 0;
    for (0..new_rank) |i| {
        if (i == axis) {
            dims[i] = 1;
            strides[i] = unsqueezedStride(input, axis);
            continue;
        }
        dims[i] = input.shape.dims[src_i];
        strides[i] = input.layout.strides[src_i];
        src_i += 1;
    }

    var shape = try Shape.initCopy(allocator, dims);
    errdefer shape.deinit();
    var layout = try Layout.initCopy(allocator, strides, input.layout.offset);
    errdefer layout.deinit();

    return .{
        .shape = shape,
        .dtype = input.dtype,
        .layout = layout,
        .device = device,
        .kind = .view,
        .allocation = .view_only,
        .input_requirement = .preserve,
    };
}

fn shapeOfOnes(allocator: std.mem.Allocator, rank: usize) !Shape {
    const dims = try allocator.alloc(usize, rank);
    defer allocator.free(dims);
    @memset(dims, 1);
    return Shape.initCopy(allocator, dims);
}

fn reversePermutation(allocator: std.mem.Allocator, rank: usize) ![]usize {
    const order = try allocator.alloc(usize, rank);
    for (0..rank) |i| order[i] = rank - 1 - i;
    return order;
}

fn validatePermutation(allocator: std.mem.Allocator, order: []const usize, rank: usize) !void {
    try contracts.requirePermutation(allocator, order, rank);
}

fn inferPermutationView(allocator: std.mem.Allocator, input: ValueSpec, order: []const usize) !OpSpec {
    const device = input.device;
    const rank = input.shape.rank();
    try validatePermutation(allocator, order, rank);

    const dims = try allocator.alloc(usize, rank);
    defer allocator.free(dims);
    const strides = try allocator.alloc(isize, rank);
    defer allocator.free(strides);
    for (order, 0..) |source_dim, i| {
        dims[i] = input.shape.dims[source_dim];
        strides[i] = input.layout.strides[source_dim];
    }

    var shape = try Shape.initCopy(allocator, dims);
    errdefer shape.deinit();
    var layout = try Layout.initCopy(allocator, strides, input.layout.offset);
    errdefer layout.deinit();
    const axes = try permuteAxes(allocator, input.axes, order);
    errdefer tensor.deinitAxes(allocator, axes);

    return .{
        .shape = shape,
        .dtype = input.dtype,
        .layout = layout,
        .axes = axes,
        .device = device,
        .kind = .view,
        .allocation = .view_only,
        .input_requirement = .preserve,
    };
}

fn unsqueezedStride(input: ValueSpec, axis: usize) isize {
    if (input.shape.rank() == 0) return 1;
    if (axis >= input.shape.rank()) return 1;
    return input.layout.strides[axis] * @as(isize, @intCast(input.shape.dims[axis]));
}

fn computeSliceLength(start: usize, stop: usize, step: isize) usize {
    if (step <= 0) return 0;
    if (start >= stop) return 0;
    return @intCast(@divFloor(@as(isize, @intCast(stop - start - 1)), step) + 1);
}

fn validateSliceRange(range: SliceRange, dim_size: usize) !void {
    if (range.step == 0) return error.InvalidSliceStep;
    if (range.step < 0) return error.NegativeSliceStepNotYetSupported;
    if (range.start > dim_size) return error.InvalidSliceBound;
    if (range.stop > dim_size) return error.InvalidSliceBound;
    if (range.start > range.stop) return error.InvalidSliceBound;
}

fn needsCopyMaterialization(ranges: []const SliceRange) bool {
    for (ranges) |range| {
        if (range.step != 1) return true;
    }
    return false;
}

test "add inference keeps shape dtype and device" {
    const allocator = std.testing.allocator;
    const a = try Value.fromSliceF32(allocator, &.{2}, &.{ 1, 2 });
    defer a.deinit();
    const b = try Value.fromSliceF32(allocator, &.{2}, &.{ 3, 4 });
    defer b.deinit();

    const op = try Op.init(.add, &.{ a, b }, .{ .binary = .{} });
    var info = try infer(allocator, op);
    defer info.deinit();

    try std.testing.expectEqual(@as(usize, 2), info.shape.numel());
    try std.testing.expectEqual(DType.f32, info.dtype);
    try std.testing.expectEqual(Device.cpu, info.device);
    try std.testing.expectEqual(ExecutionKind.elementwise_binary, info.kind);
    try std.testing.expectEqual(InputRequirement.require_storage, info.input_requirement);
}

test "add inference right-aligns broadcast dims" {
    const allocator = std.testing.allocator;
    const a = try Value.fromSliceF32(allocator, &.{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer a.deinit();
    const b = try Value.fromSliceF32(allocator, &.{3}, &.{ 10, 20, 30 });
    defer b.deinit();

    const op = try Op.init(.add, &.{ a, b }, .{ .binary = .{} });
    var info = try infer(allocator, op);
    defer info.deinit();

    try std.testing.expectEqualSlices(usize, &.{ 2, 3 }, info.shape.dims);
    try std.testing.expectEqual(DType.f32, info.dtype);
    try std.testing.expectEqual(Device.cpu, info.device);
}

test "unary relu inference keeps same shape dtype and device" {
    const allocator = std.testing.allocator;
    const value = try Value.fromSliceF32(allocator, &.{ 2, 3 }, &.{ 1, -2, 3, -4, 5, -6 });
    defer value.deinit();

    const op = try Op.init(.relu, &.{value}, .{ .unary = .{} });
    var info = try infer(allocator, op);
    defer info.deinit();

    try std.testing.expectEqualSlices(usize, &.{ 2, 3 }, info.shape.dims);
    try std.testing.expectEqual(DType.f32, info.dtype);
    try std.testing.expectEqual(Device.cpu, info.device);
    try std.testing.expectEqual(ExecutionKind.elementwise_unary, info.kind);
}

test "clamp inference keeps same shape and validates bounds" {
    const allocator = std.testing.allocator;
    const value = try Value.fromSliceF64(allocator, &.{3}, &.{ -1, 0, 1 });
    defer value.deinit();

    const op = try Op.init(.clamp, &.{value}, .{ .clamp = .{ .min = -0.5, .max = 0.5 } });
    var info = try infer(allocator, op);
    defer info.deinit();

    try std.testing.expectEqualSlices(usize, &.{3}, info.shape.dims);
    try std.testing.expectEqual(DType.f64, info.dtype);
    try std.testing.expectEqual(InputRequirement.require_storage, info.input_requirement);
}

test "reshape inference is view-like" {
    const allocator = std.testing.allocator;
    const value = try Value.fromSliceF32(allocator, &.{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer value.deinit();

    const op = try Op.init(.reshape, &.{value}, .{ .reshape = .{ .shape = &.{ 3, 2 } } });
    var info = try infer(allocator, op);
    defer info.deinit();

    try std.testing.expectEqual(AllocationIntent.view_only, info.allocation);
    try std.testing.expectEqual(@as(usize, 2), info.shape.rank());
    try std.testing.expectEqual(InputRequirement.require_contiguous_input, info.input_requirement);
}

test "cat inference joins along requested axis" {
    const allocator = std.testing.allocator;
    const a = try Value.fromSliceF32(allocator, &.{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer a.deinit();
    const b = try Value.fromSliceF32(allocator, &.{ 2, 2 }, &.{ 7, 8, 9, 10 });
    defer b.deinit();

    const op = try Op.init(.cat, &.{ a, b }, .{ .concat = .{ .axis = 1 } });
    var info = try infer(allocator, op);
    defer info.deinit();

    try std.testing.expectEqualSlices(usize, &.{ 2, 5 }, info.shape.dims);
    try std.testing.expectEqual(AllocationIntent.new_storage, info.allocation);
    try std.testing.expectEqual(InputRequirement.require_storage, info.input_requirement);
    try std.testing.expect(info.planner_hint != null);
    try std.testing.expectEqual(kernel_capability.InputLayoutDecision.accept, info.planner_hint.?.input_layout_decision);
}

test "stack inference inserts axis and counts inputs" {
    const allocator = std.testing.allocator;
    const a = try Value.fromSliceF64(allocator, &.{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer a.deinit();
    const b = try Value.fromSliceF64(allocator, &.{ 2, 3 }, &.{ 7, 8, 9, 10, 11, 12 });
    defer b.deinit();

    const op = try Op.init(.stack, &.{ a, b }, .{ .stack = .{ .axis = 0 } });
    var info = try infer(allocator, op);
    defer info.deinit();

    try std.testing.expectEqualSlices(usize, &.{ 2, 2, 3 }, info.shape.dims);
    try std.testing.expectEqual(DType.f64, info.dtype);
    try std.testing.expect(info.planner_hint != null);
    try std.testing.expectEqual(kernel_capability.InputLayoutDecision.accept, info.planner_hint.?.input_layout_decision);
}

test "contiguous inference aliases already contiguous input" {
    const allocator = std.testing.allocator;
    const value = try Value.fromSliceF32(allocator, &.{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer value.deinit();

    const op = try Op.init(.contiguous, &.{value}, .{ .none = {} });
    var info = try infer(allocator, op);
    defer info.deinit();

    try std.testing.expectEqual(AllocationIntent.view_only, info.allocation);
    try std.testing.expectEqual(ExecutionKind.view, info.kind);
}

test "permute inference is explicit-axis view" {
    const allocator = std.testing.allocator;
    const value = try Value.fromSliceF32(allocator, &.{ 2, 3, 4 }, &.{
        1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
    });
    defer value.deinit();

    const op = try Op.init(.permute, &.{value}, .{ .permute = .{ .axes = &.{ 2, 0, 1 } } });
    var info = try infer(allocator, op);
    defer info.deinit();

    try std.testing.expectEqualSlices(usize, &.{ 4, 2, 3 }, info.shape.dims);
    try std.testing.expectEqualSlices(isize, &.{ 1, 12, 4 }, info.layout.strides);
    try std.testing.expectEqual(AllocationIntent.view_only, info.allocation);
}

test "sum_all keepdim preserves rank with singleton extents" {
    const allocator = std.testing.allocator;
    const value = try Value.fromSliceF32(allocator, &.{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer value.deinit();

    const op = try Op.init(.sum_all, &.{value}, .{ .reduce_all = .{ .keepdim = true } });
    var info = try infer(allocator, op);
    defer info.deinit();

    try std.testing.expectEqual(@as(usize, 2), info.shape.rank());
    try std.testing.expectEqual(@as(usize, 1), info.shape.dims[0]);
    try std.testing.expectEqual(@as(usize, 1), info.shape.dims[1]);
    try std.testing.expectEqual(ExecutionKind.reduction_all, info.kind);
}

test "transpose inference permutes shape and strides as a view" {
    const allocator = std.testing.allocator;
    const value = try Value.fromSliceF32(allocator, &.{ 2, 3, 4 }, &.{
        1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
    });
    defer value.deinit();

    const op = try Op.init(.transpose, &.{value}, .{ .transpose = .{ .permutation = &.{ 1, 0, 2 } } });
    var info = try infer(allocator, op);
    defer info.deinit();

    try std.testing.expectEqualSlices(usize, &.{ 3, 2, 4 }, info.shape.dims);
    try std.testing.expectEqualSlices(isize, &.{ 4, 12, 1 }, info.layout.strides);
    try std.testing.expectEqual(AllocationIntent.view_only, info.allocation);
}

test "squeeze inference removes singleton axis as a view" {
    const allocator = std.testing.allocator;
    const value = try Value.fromSliceF32(allocator, &.{ 2, 1, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer value.deinit();

    const op = try Op.init(.squeeze, &.{value}, .{ .squeeze = .{ .axis = 1 } });
    var info = try infer(allocator, op);
    defer info.deinit();

    try std.testing.expectEqualSlices(usize, &.{ 2, 3 }, info.shape.dims);
    try std.testing.expectEqualSlices(isize, &.{ 3, 1 }, info.layout.strides);
    try std.testing.expectEqual(AllocationIntent.view_only, info.allocation);
}

test "unsqueeze inference inserts singleton axis as a view" {
    const allocator = std.testing.allocator;
    const value = try Value.fromSliceF32(allocator, &.{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer value.deinit();

    const op = try Op.init(.unsqueeze, &.{value}, .{ .unsqueeze = .{ .axis = 1 } });
    var info = try infer(allocator, op);
    defer info.deinit();

    try std.testing.expectEqualSlices(usize, &.{ 2, 1, 3 }, info.shape.dims);
    try std.testing.expectEqualSlices(isize, &.{ 3, 3, 1 }, info.layout.strides);
    try std.testing.expectEqual(ExecutionKind.view, info.kind);
}

test "sum_axis inference removes reduced axis by default" {
    const allocator = std.testing.allocator;
    const value = try Value.fromSliceF32(allocator, &.{ 2, 3, 4 }, &.{
        1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
    });
    defer value.deinit();

    const op = try Op.init(.sum_axis, &.{value}, .{ .reduce_axis = .{ .axis = 1 } });
    var info = try infer(allocator, op);
    defer info.deinit();

    try std.testing.expectEqualSlices(usize, &.{ 2, 4 }, info.shape.dims);
    try std.testing.expectEqual(ExecutionKind.reduction, info.kind);
}

test "mean_axis inference preserves dtype and keepdim semantics" {
    const allocator = std.testing.allocator;
    const value = try Value.fromSliceF64(allocator, &.{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer value.deinit();

    const op = try Op.init(.mean_axis, &.{value}, .{ .reduce_axis = .{ .axis = 1, .keepdim = true } });
    var info = try infer(allocator, op);
    defer info.deinit();

    try std.testing.expectEqual(DType.f64, info.dtype);
    try std.testing.expectEqualSlices(usize, &.{ 2, 1 }, info.shape.dims);
}

test "max_all inference produces scalar-like output with reduction metadata" {
    const allocator = std.testing.allocator;
    const value = try Value.fromSliceF32(allocator, &.{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer value.deinit();

    const op = try Op.init(.max_all, &.{value}, .{ .reduce_all = .{} });
    var info = try infer(allocator, op);
    defer info.deinit();

    try std.testing.expectEqualSlices(usize, &.{1}, info.shape.dims);
    try std.testing.expectEqual(ExecutionKind.reduction_all, info.kind);
}

test "variance_axis inference keeps numeric dtype and reduced shape" {
    const allocator = std.testing.allocator;
    const value = try Value.fromSliceF32(allocator, &.{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer value.deinit();

    const op = try Op.init(.variance_axis, &.{value}, .{ .reduce_axis = .{ .axis = 0 } });
    var info = try infer(allocator, op);
    defer info.deinit();

    try std.testing.expectEqual(DType.f32, info.dtype);
    try std.testing.expectEqualSlices(usize, &.{3}, info.shape.dims);
}

test "std_all inference keeps numeric dtype and scalar-like shape" {
    const allocator = std.testing.allocator;
    const value = try Value.fromSliceF64(allocator, &.{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer value.deinit();

    const op = try Op.init(.std_all, &.{value}, .{ .reduce_all = .{} });
    var info = try infer(allocator, op);
    defer info.deinit();

    try std.testing.expectEqual(DType.f64, info.dtype);
    try std.testing.expectEqualSlices(usize, &.{1}, info.shape.dims);
}

test "argmax_axis inference changes output dtype to i64" {
    const allocator = std.testing.allocator;
    const value = try Value.fromSliceF32(allocator, &.{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer value.deinit();

    const op = try Op.init(.argmax_axis, &.{value}, .{ .reduce_axis = .{ .axis = 1 } });
    var info = try infer(allocator, op);
    defer info.deinit();

    try std.testing.expectEqual(DType.i64, info.dtype);
    try std.testing.expectEqualSlices(usize, &.{2}, info.shape.dims);
}

test "argmin_all inference changes output dtype to i64" {
    const allocator = std.testing.allocator;
    const value = try Value.fromSliceF64(allocator, &.{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer value.deinit();

    const op = try Op.init(.argmin_all, &.{value}, .{ .reduce_all = .{} });
    var info = try infer(allocator, op);
    defer info.deinit();

    try std.testing.expectEqual(DType.i64, info.dtype);
    try std.testing.expectEqualSlices(usize, &.{1}, info.shape.dims);
}

test "slice inference is a view for unit-step ranges" {
    const allocator = std.testing.allocator;
    const value = try Value.fromSliceF32(allocator, &.{ 4, 5 }, &.{
        1,  2,  3,  4,  5,
        6,  7,  8,  9,  10,
        11, 12, 13, 14, 15,
        16, 17, 18, 19, 20,
    });
    defer value.deinit();

    const op = try Op.init(.slice, &.{value}, .{ .slice = .{ .ranges = &.{
        .{ .start = 1, .stop = 3, .step = 1 },
        .{ .start = 0, .stop = 5, .step = 1 },
    } } });
    var info = try infer(allocator, op);
    defer info.deinit();

    try std.testing.expectEqual(AllocationIntent.view_only, info.allocation);
    try std.testing.expectEqual(ExecutionKind.view, info.kind);
    try std.testing.expectEqualSlices(usize, &.{ 2, 5 }, info.shape.dims);
    try std.testing.expectEqual(@as(?PlannerHint, null), info.planner_hint);
}

test "slice inference requires new storage for stepped ranges" {
    const allocator = std.testing.allocator;
    const value = try Value.fromSliceF32(allocator, &.{6}, &.{ 1, 2, 3, 4, 5, 6 });
    defer value.deinit();

    const op = try Op.init(.slice, &.{value}, .{ .slice = .{ .ranges = &.{
        .{ .start = 0, .stop = 6, .step = 2 },
    } } });
    var info = try infer(allocator, op);
    defer info.deinit();

    try std.testing.expectEqual(AllocationIntent.new_storage, info.allocation);
    try std.testing.expectEqual(InputRequirement.require_storage, info.input_requirement);
    try std.testing.expect(info.planner_hint != null);
    try std.testing.expectEqual(kernel_capability.InputLayoutDecision.accept, info.planner_hint.?.input_layout_decision);
}

test "where inference keeps branch dtype and same-shape output" {
    const allocator = std.testing.allocator;
    const cond = try Value.fromSliceI64(allocator, &.{ 2, 2 }, &.{ 1, 0, 0, 1 });
    defer cond.deinit();
    const on_true = try Value.fromSliceF32(allocator, &.{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer on_true.deinit();
    const on_false = try Value.fromSliceF32(allocator, &.{ 2, 2 }, &.{ 5, 6, 7, 8 });
    defer on_false.deinit();

    const op = try Op.init(.where, &.{ cond, on_true, on_false }, .{ .none = {} });
    var info = try infer(allocator, op);
    defer info.deinit();

    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, info.shape.dims);
    try std.testing.expectEqual(DType.f32, info.dtype);
    try std.testing.expectEqual(ExecutionKind.elementwise_generic, info.kind);
    try std.testing.expect(info.planner_hint != null);
    try std.testing.expectEqual(kernel_capability.InputLayoutDecision.accept, info.planner_hint.?.input_layout_decision);
}

test "masked_fill inference keeps data shape/dtype and validates mask shape" {
    const allocator = std.testing.allocator;
    const input = try Value.fromSliceF32(allocator, &.{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer input.deinit();
    const mask = try Value.fromSliceI64(allocator, &.{ 2, 2 }, &.{ 1, 0, 1, 0 });
    defer mask.deinit();

    const op = try Op.init(.masked_fill, &.{ input, mask }, .{ .masked_fill = .{ .value = 0.0 } });
    var info = try infer(allocator, op);
    defer info.deinit();

    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, info.shape.dims);
    try std.testing.expectEqual(DType.f32, info.dtype);
    try std.testing.expectEqual(ExecutionKind.elementwise_generic, info.kind);
    try std.testing.expect(info.planner_hint != null);
    try std.testing.expectEqual(kernel_capability.InputLayoutDecision.accept, info.planner_hint.?.input_layout_decision);
}

test "binary inference records pack_to_dense planner hint for negative strides" {
    const allocator = std.testing.allocator;

    var lhs_shape = try Shape.initCopy(allocator, &.{ 3, 2 });
    defer lhs_shape.deinit();
    var lhs_layout = try Layout.initCopy(allocator, &.{ -2, 1 }, 4);
    defer lhs_layout.deinit();
    const lhs = ValueSpec{
        .shape = lhs_shape,
        .dtype = .f32,
        .layout = lhs_layout,
        .device = .cpu,
    };

    var rhs_shape = try Shape.initCopy(allocator, &.{ 3, 2 });
    defer rhs_shape.deinit();
    var rhs_layout = try Layout.initContiguous(allocator, rhs_shape);
    defer rhs_layout.deinit();
    const rhs = ValueSpec{
        .shape = rhs_shape,
        .dtype = .f32,
        .layout = rhs_layout,
        .device = .cpu,
    };

    var info = try inferFromSpecs(allocator, .add, &.{ lhs, rhs }, .{ .binary = .{} });
    defer info.deinit();

    try std.testing.expectEqual(InputRequirement.require_storage, info.input_requirement);
    try std.testing.expect(info.planner_hint != null);
    try std.testing.expectEqual(kernel_capability.InputLayoutDecision.pack_to_dense, info.planner_hint.?.input_layout_decision);
}

test "where inference right-aligns broadcast dims" {
    const allocator = std.testing.allocator;
    const cond = try Value.fromSliceI64(allocator, &.{ 3, 1 }, &.{ 1, 0, 1 });
    defer cond.deinit();
    const on_true = try Value.fromSliceF32(allocator, &.{ 2, 3, 4 }, &.{
        1,  2,  3,  4,
        5,  6,  7,  8,
        9,  10, 11, 12,
        13, 14, 15, 16,
        17, 18, 19, 20,
        21, 22, 23, 24,
    });
    defer on_true.deinit();
    const on_false = try Value.fromSliceF32(allocator, &.{4}, &.{ 0, 0, 0, 0 });
    defer on_false.deinit();

    const op = try Op.init(.where, &.{ cond, on_true, on_false }, .{ .none = {} });
    var info = try infer(allocator, op);
    defer info.deinit();

    try std.testing.expectEqualSlices(usize, &.{ 2, 3, 4 }, info.shape.dims);
}

test "masked_fill inference right-aligns broadcast dims" {
    const allocator = std.testing.allocator;
    const input = try Value.fromSliceF32(allocator, &.{ 2, 3, 4 }, &.{
        1,  2,  3,  4,
        5,  6,  7,  8,
        9,  10, 11, 12,
        13, 14, 15, 16,
        17, 18, 19, 20,
        21, 22, 23, 24,
    });
    defer input.deinit();
    const mask = try Value.fromSliceI64(allocator, &.{ 3, 1 }, &.{ 1, 0, 1 });
    defer mask.deinit();

    const op = try Op.init(.masked_fill, &.{ input, mask }, .{ .masked_fill = .{ .value = 0.0 } });
    var info = try infer(allocator, op);
    defer info.deinit();

    try std.testing.expectEqualSlices(usize, &.{ 2, 3, 4 }, info.shape.dims);
    try std.testing.expectEqual(DType.f32, info.dtype);
}

test "gather inference follows index shape on the selected axis contract" {
    const allocator = std.testing.allocator;
    const input = try Value.fromSliceF32(allocator, &.{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer input.deinit();
    const index = try Value.fromSliceI64(allocator, &.{ 2, 4 }, &.{ 0, 2, 1, 0, 1, 1, 2, 0 });
    defer index.deinit();

    const op = try Op.init(.gather, &.{ input, index }, .{ .gather = .{ .axis = 1 } });
    var info = try infer(allocator, op);
    defer info.deinit();

    try std.testing.expectEqualSlices(usize, &.{ 2, 4 }, info.shape.dims);
    try std.testing.expectEqual(DType.f32, info.dtype);
    try std.testing.expectEqual(ExecutionKind.index, info.kind);
    try std.testing.expect(info.planner_hint != null);
    try std.testing.expectEqual(kernel_capability.InputLayoutDecision.accept, info.planner_hint.?.input_layout_decision);
}

test "index_select inference replaces the selected axis extent" {
    const allocator = std.testing.allocator;
    const input = try Value.fromSliceF64(allocator, &.{ 2, 3, 4 }, &.{
        1,  2,  3,  4,
        5,  6,  7,  8,
        9,  10, 11, 12,
        13, 14, 15, 16,
        17, 18, 19, 20,
        21, 22, 23, 24,
    });
    defer input.deinit();
    const index = try Value.fromSliceI64(allocator, &.{2}, &.{ 2, 0 });
    defer index.deinit();

    const op = try Op.init(.index_select, &.{ input, index }, .{ .index_select = .{ .axis = 1 } });
    var info = try infer(allocator, op);
    defer info.deinit();

    try std.testing.expectEqualSlices(usize, &.{ 2, 2, 4 }, info.shape.dims);
    try std.testing.expectEqual(DType.f64, info.dtype);
    try std.testing.expectEqual(ExecutionKind.index, info.kind);
    try std.testing.expect(info.planner_hint != null);
    try std.testing.expectEqual(kernel_capability.InputLayoutDecision.accept, info.planner_hint.?.input_layout_decision);
}

test "one_hot inference appends class axis and returns f32 output" {
    const allocator = std.testing.allocator;
    const indices = try Value.fromSliceI64(allocator, &.{ 2, 3 }, &.{ 0, 1, 2, 1, 2, 0 });
    defer indices.deinit();

    const op = try Op.init(.one_hot, &.{indices}, .{ .one_hot = .{ .num_classes = 4 } });
    var info = try infer(allocator, op);
    defer info.deinit();

    try std.testing.expectEqualSlices(usize, &.{ 2, 3, 4 }, info.shape.dims);
    try std.testing.expectEqual(DType.f32, info.dtype);
    try std.testing.expectEqual(ExecutionKind.index, info.kind);
    try std.testing.expect(info.planner_hint != null);
    try std.testing.expectEqual(kernel_capability.InputLayoutDecision.accept, info.planner_hint.?.input_layout_decision);
}

test "embedding inference appends embedding dims to index shape" {
    const allocator = std.testing.allocator;
    const table = try Value.fromSliceF32(allocator, &.{ 6, 4 }, &.{
        0,  1,  2,  3,
        4,  5,  6,  7,
        8,  9,  10, 11,
        12, 13, 14, 15,
        16, 17, 18, 19,
        20, 21, 22, 23,
    });
    defer table.deinit();
    const index = try Value.fromSliceI64(allocator, &.{ 2, 3 }, &.{ 0, 1, 2, 3, 4, 5 });
    defer index.deinit();

    const op = try Op.init(.embedding, &.{ table, index }, .{ .none = {} });
    var info = try infer(allocator, op);
    defer info.deinit();

    try std.testing.expectEqualSlices(usize, &.{ 2, 3, 4 }, info.shape.dims);
    try std.testing.expectEqual(DType.f32, info.dtype);
    try std.testing.expectEqual(ExecutionKind.index, info.kind);
    try std.testing.expect(info.planner_hint != null);
    try std.testing.expectEqual(kernel_capability.InputLayoutDecision.accept, info.planner_hint.?.input_layout_decision);
}

test "gather inference records pack_to_dense planner hint for negative strides" {
    const allocator = std.testing.allocator;

    var input_shape = try Shape.initCopy(allocator, &.{ 2, 3 });
    defer input_shape.deinit();
    var input_layout = try Layout.initCopy(allocator, &.{ -3, 1 }, 3);
    defer input_layout.deinit();
    const input = ValueSpec{
        .shape = input_shape,
        .dtype = .f32,
        .layout = input_layout,
        .device = .cpu,
    };

    var index_shape = try Shape.initCopy(allocator, &.{ 2, 2 });
    defer index_shape.deinit();
    var index_layout = try Layout.initContiguous(allocator, index_shape);
    defer index_layout.deinit();
    const index = ValueSpec{
        .shape = index_shape,
        .dtype = .i64,
        .layout = index_layout,
        .device = .cpu,
    };

    var info = try inferFromSpecs(allocator, .gather, &.{ input, index }, .{ .gather = .{ .axis = 1 } });
    defer info.deinit();

    try std.testing.expect(info.planner_hint != null);
    try std.testing.expectEqual(kernel_capability.InputLayoutDecision.pack_to_dense, info.planner_hint.?.input_layout_decision);
}

test "log_softmax_nll inference returns scalar-like loss output" {
    const allocator = std.testing.allocator;
    const logits = try Value.fromSliceF32(allocator, &.{ 2, 3 }, &.{ 0.1, 0.2, 0.3, 1.0, 0.0, -1.0 });
    defer logits.deinit();
    const targets = try Value.fromSliceF32(allocator, &.{ 2, 3 }, &.{ 0.0, 0.0, 1.0, 1.0, 0.0, 0.0 });
    defer targets.deinit();

    const op = try Op.init(.log_softmax_nll, &.{ logits, targets }, .{ .log_softmax_nll = .{ .axis = 1 } });
    var info = try infer(allocator, op);
    defer info.deinit();

    try std.testing.expectEqual(ExecutionKind.reduction_all, info.kind);
    try std.testing.expectEqual(DType.f32, info.dtype);
    try std.testing.expectEqualSlices(usize, &.{1}, info.shape.dims);
    try std.testing.expect(info.planner_hint != null);
    try std.testing.expectEqual(kernel_capability.InputLayoutDecision.accept, info.planner_hint.?.input_layout_decision);
}

test "cross_entropy inference returns scalar-like loss output" {
    const allocator = std.testing.allocator;
    const logits = try Value.fromSliceF32(allocator, &.{ 2, 3 }, &.{ 0.1, 0.2, 0.3, 1.0, 0.0, -1.0 });
    defer logits.deinit();
    const targets = try Value.fromSliceF32(allocator, &.{ 2, 3 }, &.{ 0.0, 0.0, 1.0, 1.0, 0.0, 0.0 });
    defer targets.deinit();

    const op = try Op.init(.cross_entropy, &.{ logits, targets }, .{ .cross_entropy = .{ .axis = 1 } });
    var info = try infer(allocator, op);
    defer info.deinit();

    try std.testing.expectEqual(ExecutionKind.reduction_all, info.kind);
    try std.testing.expectEqual(DType.f32, info.dtype);
    try std.testing.expectEqualSlices(usize, &.{1}, info.shape.dims);
    try std.testing.expect(info.planner_hint != null);
    try std.testing.expectEqual(kernel_capability.InputLayoutDecision.accept, info.planner_hint.?.input_layout_decision);
}

test "cross_entropy_indexed inference returns scalar-like loss output" {
    const allocator = std.testing.allocator;
    const logits = try Value.fromSliceF32(allocator, &.{ 2, 3 }, &.{ 0.1, 0.2, 0.3, 1.0, 0.0, -1.0 });
    defer logits.deinit();
    const targets = try Value.fromSliceI64(allocator, &.{2}, &.{ 2, 0 });
    defer targets.deinit();

    const op = try Op.init(.cross_entropy_indexed, &.{ logits, targets }, .{ .cross_entropy_indexed = .{ .axis = 1 } });
    var info = try infer(allocator, op);
    defer info.deinit();

    try std.testing.expectEqual(ExecutionKind.reduction_all, info.kind);
    try std.testing.expectEqual(DType.f32, info.dtype);
    try std.testing.expectEqualSlices(usize, &.{1}, info.shape.dims);
    try std.testing.expect(info.planner_hint != null);
    try std.testing.expectEqual(kernel_capability.InputLayoutDecision.accept, info.planner_hint.?.input_layout_decision);
}

test "cross_entropy_indexed_backward inference returns logits-shaped output" {
    const allocator = std.testing.allocator;
    const logits = try Value.fromSliceF32(allocator, &.{ 2, 3 }, &.{ 0.1, 0.2, 0.3, 1.0, 0.0, -1.0 });
    defer logits.deinit();
    const targets = try Value.fromSliceI64(allocator, &.{2}, &.{ 2, 0 });
    defer targets.deinit();
    const grad_out = try Value.fromSliceF32(allocator, &.{1}, &.{1.0});
    defer grad_out.deinit();

    const op = try Op.init(.cross_entropy_indexed_backward, &.{ logits, targets, grad_out }, .{ .cross_entropy_indexed_backward = .{ .axis = 1 } });
    var info = try infer(allocator, op);
    defer info.deinit();

    try std.testing.expectEqual(ExecutionKind.elementwise_generic, info.kind);
    try std.testing.expectEqual(DType.f32, info.dtype);
    try std.testing.expectEqualSlices(usize, &.{ 2, 3 }, info.shape.dims);
    try std.testing.expect(info.planner_hint != null);
    try std.testing.expectEqual(kernel_capability.InputLayoutDecision.accept, info.planner_hint.?.input_layout_decision);
}

test "loss inference records pack_to_dense planner hint for negative strides" {
    const allocator = std.testing.allocator;

    var logits_shape = try Shape.initCopy(allocator, &.{ 2, 3 });
    defer logits_shape.deinit();
    var logits_layout = try Layout.initCopy(allocator, &.{ -3, 1 }, 3);
    defer logits_layout.deinit();
    const logits = ValueSpec{
        .shape = logits_shape,
        .dtype = .f32,
        .layout = logits_layout,
        .device = .cpu,
    };

    var targets_shape = try Shape.initCopy(allocator, &.{ 2, 3 });
    defer targets_shape.deinit();
    var targets_layout = try Layout.initCopy(allocator, &.{ -3, 1 }, 3);
    defer targets_layout.deinit();
    const targets = ValueSpec{
        .shape = targets_shape,
        .dtype = .f32,
        .layout = targets_layout,
        .device = .cpu,
    };

    var info = try inferFromSpecs(allocator, .log_softmax_nll, &.{ logits, targets }, .{ .log_softmax_nll = .{ .axis = 1 } });
    defer info.deinit();

    try std.testing.expect(info.planner_hint != null);
    try std.testing.expectEqual(kernel_capability.InputLayoutDecision.pack_to_dense, info.planner_hint.?.input_layout_decision);
}

test "topk inference emits values and indices output contracts" {
    const allocator = std.testing.allocator;
    const input = try Value.fromSliceF32(allocator, &.{ 2, 5 }, &.{
        1,  2, 3, 4, 5,
        10, 9, 8, 7, 6,
    });
    defer input.deinit();

    const op = try Op.init(.topk, &.{input}, .{ .topk = .{ .k = 3, .axis = 1 } });
    var info = try infer(allocator, op);
    defer info.deinit();

    try std.testing.expectEqualSlices(usize, &.{ 2, 3 }, info.shape.dims);
    try std.testing.expectEqual(DType.f32, info.dtype);
    try std.testing.expectEqual(ExecutionKind.index, info.kind);
    try std.testing.expect(info.secondary_output != null);
    try std.testing.expectEqual(DType.i64, info.secondary_output.?.dtype);
    try std.testing.expectEqualSlices(usize, &.{ 2, 3 }, info.secondary_output.?.shape.dims);
}

test "softmax inference preserves shape and validates axis contract" {
    const allocator = std.testing.allocator;
    const input = try Value.fromSliceF32(allocator, &.{ 2, 3, 4 }, &.{
        1,  2,  3,  4,
        5,  6,  7,  8,
        9,  10, 11, 12,
        13, 14, 15, 16,
        17, 18, 19, 20,
        21, 22, 23, 24,
    });
    defer input.deinit();

    const op = try Op.init(.softmax, &.{input}, .{ .softmax = .{ .axis = 2 } });
    var info = try infer(allocator, op);
    defer info.deinit();

    try std.testing.expectEqualSlices(usize, &.{ 2, 3, 4 }, info.shape.dims);
    try std.testing.expectEqual(DType.f32, info.dtype);
    try std.testing.expectEqual(ExecutionKind.reduction, info.kind);
    try std.testing.expect(info.planner_hint != null);
    try std.testing.expectEqual(kernel_capability.InputLayoutDecision.accept, info.planner_hint.?.input_layout_decision);
}

test "softmax inference accepts positive-stride metal view layout" {
    const allocator = std.testing.allocator;
    var shape = try Shape.initCopy(allocator, &.{ 2, 2 });
    defer shape.deinit();
    var layout = try Layout.initCopy(allocator, &.{ 1, 2 }, 0);
    defer layout.deinit();
    const input = ValueSpec{
        .shape = shape,
        .dtype = .f32,
        .layout = layout,
        .device = .metal,
    };

    var info = try inferFromSpecs(allocator, .softmax, &.{input}, .{ .softmax = .{ .axis = 1 } });
    defer info.deinit();

    try std.testing.expect(info.planner_hint != null);
    try std.testing.expectEqual(kernel_capability.InputLayoutDecision.accept, info.planner_hint.?.input_layout_decision);
}

test "layer_norm inference preserves shape and validates axis/eps contract" {
    const allocator = std.testing.allocator;
    const input = try Value.fromSliceF32(allocator, &.{ 2, 3, 4 }, &.{
        1,  2,  3,  4,
        5,  6,  7,  8,
        9,  10, 11, 12,
        13, 14, 15, 16,
        17, 18, 19, 20,
        21, 22, 23, 24,
    });
    defer input.deinit();

    const op = try Op.init(.layer_norm, &.{input}, .{ .layer_norm = .{ .axis = 2, .eps = 1e-5 } });
    var info = try infer(allocator, op);
    defer info.deinit();

    try std.testing.expectEqualSlices(usize, &.{ 2, 3, 4 }, info.shape.dims);
    try std.testing.expectEqual(DType.f32, info.dtype);
    try std.testing.expectEqual(ExecutionKind.elementwise_generic, info.kind);
    try std.testing.expect(info.planner_hint != null);
    try std.testing.expectEqual(kernel_capability.InputLayoutDecision.accept, info.planner_hint.?.input_layout_decision);
}

test "layer_norm inference records pack_to_dense planner hint for negative strides" {
    const allocator = std.testing.allocator;

    var input_shape = try Shape.initCopy(allocator, &.{ 3, 2 });
    defer input_shape.deinit();
    var input_layout = try Layout.initCopy(allocator, &.{ -2, 1 }, 4);
    defer input_layout.deinit();
    const input = ValueSpec{
        .shape = input_shape,
        .dtype = .f32,
        .layout = input_layout,
        .device = .cpu,
    };

    var info = try inferFromSpecs(allocator, .layer_norm, &.{input}, .{ .layer_norm = .{ .axis = 1, .eps = 1e-5 } });
    defer info.deinit();

    try std.testing.expect(info.planner_hint != null);
    try std.testing.expectEqual(kernel_capability.InputLayoutDecision.pack_to_dense, info.planner_hint.?.input_layout_decision);
}

test "dot inference requires 1d inputs and returns scalar-like shape" {
    const allocator = std.testing.allocator;
    const lhs = try Value.fromSliceF64(allocator, &.{3}, &.{ 1, 2, 3 });
    defer lhs.deinit();
    const rhs = try Value.fromSliceF64(allocator, &.{3}, &.{ 4, 5, 6 });
    defer rhs.deinit();

    const op = try Op.init(.dot, &.{ lhs, rhs }, .{ .none = {} });
    var info = try infer(allocator, op);
    defer info.deinit();

    try std.testing.expectEqual(@as(usize, 0), info.shape.rank());
    try std.testing.expectEqual(DType.f64, info.dtype);
    try std.testing.expectEqual(ExecutionKind.reduction, info.kind);
    try std.testing.expect(info.planner_hint != null);
    try std.testing.expectEqual(kernel_capability.InputLayoutDecision.accept, info.planner_hint.?.input_layout_decision);
}

test "matmul inference broadcasts batch dims and preserves matrix contract" {
    const allocator = std.testing.allocator;
    const lhs = try Value.fromSliceF32(allocator, &.{ 2, 3, 4 }, &.{
        1,  2,  3,  4,
        5,  6,  7,  8,
        9,  10, 11, 12,
        13, 14, 15, 16,
        17, 18, 19, 20,
        21, 22, 23, 24,
    });
    defer lhs.deinit();
    const rhs = try Value.fromSliceF32(allocator, &.{ 1, 4, 5 }, &.{
        1,  2,  3,  4,  5,
        6,  7,  8,  9,  10,
        11, 12, 13, 14, 15,
        16, 17, 18, 19, 20,
    });
    defer rhs.deinit();

    const op = try Op.init(.matmul, &.{ lhs, rhs }, .{ .none = {} });
    var info = try infer(allocator, op);
    defer info.deinit();

    try std.testing.expectEqualSlices(usize, &.{ 2, 3, 5 }, info.shape.dims);
    try std.testing.expectEqual(DType.f32, info.dtype);
    try std.testing.expectEqual(ExecutionKind.reduction, info.kind);
    try std.testing.expectEqual(InputRequirement.preserve, info.input_requirement);
    try std.testing.expect(info.planner_hint != null);
    try std.testing.expectEqual(kernel_capability.InputLayoutDecision.accept, info.planner_hint.?.input_layout_decision);
}

test "reduce_to_shape inference right-aligns target dims" {
    const allocator = std.testing.allocator;
    const input = try Value.fromSliceF32(allocator, &.{ 2, 2, 3 }, &.{
        1,  2,  3,
        4,  5,  6,
        7,  8,  9,
        10, 11, 12,
    });
    defer input.deinit();

    const op = try Op.init(.reduce_to_shape, &.{input}, .{ .reduce_to_shape = .{ .shape = &.{ 1, 3 } } });
    var info = try infer(allocator, op);
    defer info.deinit();

    try std.testing.expectEqualSlices(usize, &.{ 1, 3 }, info.shape.dims);
    try std.testing.expect(info.planner_hint != null);
    try std.testing.expectEqual(kernel_capability.InputLayoutDecision.accept, info.planner_hint.?.input_layout_decision);
}

test "reduction inference records pack_to_dense planner hint for negative strides" {
    const allocator = std.testing.allocator;

    var input_shape = try Shape.initCopy(allocator, &.{ 3, 2 });
    defer input_shape.deinit();
    var input_layout = try Layout.initCopy(allocator, &.{ -2, 1 }, 4);
    defer input_layout.deinit();
    const input = ValueSpec{
        .shape = input_shape,
        .dtype = .f32,
        .layout = input_layout,
        .device = .cpu,
    };

    var info = try inferFromSpecs(allocator, .sum_axis, &.{input}, .{ .reduce_axis = .{ .axis = 1, .keepdim = false } });
    defer info.deinit();

    try std.testing.expectEqual(InputRequirement.require_storage, info.input_requirement);
    try std.testing.expect(info.planner_hint != null);
    try std.testing.expectEqual(kernel_capability.InputLayoutDecision.pack_to_dense, info.planner_hint.?.input_layout_decision);
}

test "matmul inference supports vector @ matrix and matrix @ vector patterns" {
    const allocator = std.testing.allocator;

    const v = try Value.fromSliceF32(allocator, &.{4}, &.{ 1, 2, 3, 4 });
    defer v.deinit();
    const m = try Value.fromSliceF32(allocator, &.{ 4, 3 }, &.{
        1,  2,  3,
        4,  5,  6,
        7,  8,  9,
        10, 11, 12,
    });
    defer m.deinit();

    const vm = try Op.init(.matmul, &.{ v, m }, .{ .none = {} });
    var vm_info = try infer(allocator, vm);
    defer vm_info.deinit();
    try std.testing.expectEqualSlices(usize, &.{3}, vm_info.shape.dims);
    try std.testing.expectEqual(InputRequirement.preserve, vm_info.input_requirement);
    try std.testing.expect(vm_info.planner_hint != null);
    try std.testing.expectEqual(kernel_capability.InputLayoutDecision.accept, vm_info.planner_hint.?.input_layout_decision);

    const m2 = try Value.fromSliceF32(allocator, &.{ 2, 4 }, &.{
        1, 2, 3, 4,
        5, 6, 7, 8,
    });
    defer m2.deinit();
    const v2 = try Value.fromSliceF32(allocator, &.{4}, &.{ 1, 2, 3, 4 });
    defer v2.deinit();

    const mv = try Op.init(.matmul, &.{ m2, v2 }, .{ .none = {} });
    var mv_info = try infer(allocator, mv);
    defer mv_info.deinit();
    try std.testing.expectEqualSlices(usize, &.{2}, mv_info.shape.dims);
    try std.testing.expectEqual(InputRequirement.preserve, mv_info.input_requirement);
    try std.testing.expect(mv_info.planner_hint != null);
    try std.testing.expectEqual(kernel_capability.InputLayoutDecision.accept, mv_info.planner_hint.?.input_layout_decision);
}

test "matmul inference records pack_to_dense planner hint for negative strides" {
    const allocator = std.testing.allocator;

    var lhs_shape = try Shape.initCopy(allocator, &.{ 3, 2 });
    defer lhs_shape.deinit();
    var lhs_layout = try Layout.initCopy(allocator, &.{ -2, 1 }, 4);
    defer lhs_layout.deinit();
    const lhs = ValueSpec{
        .shape = lhs_shape,
        .dtype = .f32,
        .layout = lhs_layout,
        .device = .cpu,
    };

    var rhs_shape = try Shape.initCopy(allocator, &.{ 2, 2 });
    defer rhs_shape.deinit();
    var rhs_layout = try Layout.initContiguous(allocator, rhs_shape);
    defer rhs_layout.deinit();
    const rhs = ValueSpec{
        .shape = rhs_shape,
        .dtype = .f32,
        .layout = rhs_layout,
        .device = .cpu,
    };

    var info = try inferFromSpecs(allocator, .matmul, &.{ lhs, rhs }, .{ .none = {} });
    defer info.deinit();

    try std.testing.expectEqual(InputRequirement.preserve, info.input_requirement);
    try std.testing.expect(info.planner_hint != null);
    try std.testing.expectEqual(kernel_capability.InputLayoutDecision.pack_to_dense, info.planner_hint.?.input_layout_decision);
}
