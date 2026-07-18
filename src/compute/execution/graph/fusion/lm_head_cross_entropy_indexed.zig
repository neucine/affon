const std = @import("std");
const compat = @import("../../../../support/compat.zig");
const Graph = @import("../../../types/ir/index.zig").Graph;
const Node = @import("../../../types/ir/index.zig").Node;
const Step = @import("../../../types/ir/plan.zig").Step;
const OpTag = @import("../../../types/operation/tag.zig").OpTag;
const Tensor = @import("../../../types/tensor/tensor.zig").Tensor;
const kernel_dispatch = @import("../../../backend/dispatch.zig");
const value_helpers = @import("value_helpers.zig");
const metal_common = @import("../../../backend/metal/common.zig");
const common = @import("common.zig");

fn containsNode(group: []const Step, node_id: u32) bool {
    for (group) |step| {
        if (step.node_id == node_id) return true;
    }
    return false;
}

fn producerNode(graph: *const Graph, value_id: u32) ?Node {
    if (value_id >= graph.values.items.len) return null;
    const producer_id = graph.values.items[value_id].producer;
    if (producer_id >= graph.nodes.items.len) return null;
    return graph.nodes.items[producer_id];
}

fn requireProducerTag(
    graph: *const Graph,
    value_id: u32,
    tag: OpTag,
) ?Node {
    const node = producerNode(graph, value_id) orelse return null;
    return switch (node.kind) {
        .op => |actual| if (actual == tag) node else null,
        else => null,
    };
}

fn requireProducerTagInGroup(
    graph: *const Graph,
    group: []const Step,
    value_id: u32,
    tag: OpTag,
) ?Node {
    const node = requireProducerTag(graph, value_id, tag) orelse return null;
    if (!containsNode(group, node.id)) return null;
    return node;
}

fn matchesTokenSlice(node: Node) bool {
    const slice = switch (node.options) {
        .slice => |s| s,
        else => return false,
    };
    if (slice.ranges.len != 2) return false;
    const r0 = slice.ranges[0];
    const r1 = slice.ranges[1];
    return r0.start == 0 and r0.step == 1 and r1.start == 1 and r1.step == 1;
}

fn matchTargetIndexBranch(
    graph: *const Graph,
    group: []const Step,
    target_index_id: u32,
) bool {
    const target_id = if (requireProducerTagInGroup(graph, group, target_index_id, .cast)) |cast_node| blk: {
        if (cast_node.inputs.len != 1) return false;
        break :blk cast_node.inputs[0];
    } else target_index_id;

    const reshape_node = requireProducerTagInGroup(graph, group, target_id, .reshape) orelse return false;
    if (reshape_node.inputs.len != 1) return false;

    const contiguous_input_id = if (requireProducerTagInGroup(graph, group, reshape_node.inputs[0], .contiguous)) |contiguous_node| blk: {
        if (contiguous_node.inputs.len != 1) return false;
        break :blk contiguous_node.inputs[0];
    } else reshape_node.inputs[0];

    const slice_input_id = if (requireProducerTagInGroup(graph, group, contiguous_input_id, .reshape)) |shape_restore_node| blk: {
        if (shape_restore_node.inputs.len != 1) return false;
        break :blk shape_restore_node.inputs[0];
    } else contiguous_input_id;

    const contiguous_node = requireProducerTagInGroup(graph, group, slice_input_id, .contiguous) orelse null;
    const slice_value_id = if (contiguous_node) |node| blk: {
        if (node.inputs.len != 1) return false;
        break :blk node.inputs[0];
    } else slice_input_id;

    const slice_node = requireProducerTagInGroup(graph, group, slice_value_id, .slice) orelse return false;
    return slice_node.inputs.len == 1 and matchesTokenSlice(slice_node);
}

fn requireTransposeLikeProducerInGroup(
    graph: *const Graph,
    group: []const Step,
    value_id: u32,
) ?Node {
    if (requireProducerTagInGroup(graph, group, value_id, .transpose)) |node| return node;
    if (requireProducerTagInGroup(graph, group, value_id, .permute)) |node| return node;
    return null;
}

fn requireLogits2dSource(
    graph: *const Graph,
    group: []const Step,
    logits2d_id: u32,
) ?Node {
    const logits2d_reshape = requireProducerTagInGroup(graph, group, logits2d_id, .reshape) orelse return null;
    if (logits2d_reshape.inputs.len != 1) return null;

    var source_id = logits2d_reshape.inputs[0];
    if (requireProducerTagInGroup(graph, group, source_id, .reshape)) |reshape_node| {
        if (reshape_node.inputs.len != 1) return null;
        source_id = reshape_node.inputs[0];
    }

    const dense_flat = requireProducerTagInGroup(graph, group, source_id, .contiguous) orelse return null;
    if (dense_flat.inputs.len != 1) return null;

    const transpose_node = requireTransposeLikeProducerInGroup(graph, group, dense_flat.inputs[0]) orelse return null;
    if (transpose_node.inputs.len != 1) return null;
    return transpose_node;
}

fn matchLogitsBranch(
    graph: *const Graph,
    group: []const Step,
    logits2d_id: u32,
) bool {
    const transpose_node = requireLogits2dSource(graph, group, logits2d_id) orelse return false;
    const matmul_node = requireProducerTag(graph, transpose_node.inputs[0], .matmul) orelse return false;
    return matmul_node.inputs.len == 2;
}

pub fn shouldSkipPreparatoryStep(
    graph: *const Graph,
    group: []const Step,
    step: Step,
) bool {
    if (!matchesGroup(graph, group)) return false;
    const loss_node = findLossNode(graph, group) orelse return false;
    if (loss_node.inputs.len != 2) return false;
    return isLogitsPreprocessingNode(graph, group, loss_node.inputs[0], step.node_id);
}

fn isLogitsPreprocessingNode(
    graph: *const Graph,
    group: []const Step,
    logits2d_id: u32,
    node_id: u32,
) bool {
    const logits2d_reshape = requireProducerTagInGroup(graph, group, logits2d_id, .reshape) orelse return false;
    if (node_id == logits2d_reshape.id) return true;
    if (logits2d_reshape.inputs.len != 1) return false;

    var source_id = logits2d_reshape.inputs[0];
    if (requireProducerTagInGroup(graph, group, source_id, .reshape)) |reshape_node| {
        if (node_id == reshape_node.id) return true;
        if (reshape_node.inputs.len != 1) return false;
        source_id = reshape_node.inputs[0];
    }

    const dense_flat = requireProducerTagInGroup(graph, group, source_id, .contiguous) orelse return false;
    if (node_id == dense_flat.id) return true;
    if (dense_flat.inputs.len != 1) return false;

    const transpose_node = requireTransposeLikeProducerInGroup(graph, group, dense_flat.inputs[0]) orelse return false;
    return node_id == transpose_node.id;
}

fn findLossNode(graph: *const Graph, group: []const Step) ?Node {
    if (group.len == 0) return null;
    const node = graph.nodes.items[group[group.len - 1].node_id];
    return switch (node.kind) {
        .op => |tag| if (tag == .cross_entropy_indexed) node else null,
        else => null,
    };
}

fn findRegionContainingProducer(plan: *const @import("../../../types/ir/plan.zig").GraphPlan, producer_id: u32) ?@import("../../../types/ir/plan.zig").Region {
    for (plan.regions.items) |region| {
        for (plan.steps.items[region.step_start..region.step_end]) |step| {
            if (step.node_id == producer_id) return region;
        }
    }
    return null;
}

pub fn matchesGroup(graph: *const Graph, group: []const Step) bool {
    if (group.len < 4) return false;

    const loss_step = group[group.len - 1];
    const loss_node = graph.nodes.items[loss_step.node_id];
    const loss_tag = switch (loss_node.kind) {
        .op => |tag| tag,
        else => return false,
    };
    if (loss_tag != .cross_entropy_indexed) return false;
    if (loss_node.inputs.len != 2 or loss_node.outputs.len != 1) return false;

    const axis = switch (loss_node.options) {
        .cross_entropy_indexed => |o| o.axis,
        .none => @as(usize, 1),
        else => return false,
    };
    if (axis != 1) return false;

    const logits2d_id = loss_node.inputs[0];
    const target_index_id = loss_node.inputs[1];

    const logits2d_spec = graph.values.items[logits2d_id].spec;
    const target_spec = graph.values.items[target_index_id].spec;
    if (logits2d_spec.shape.rank() != 2) return false;
    if (target_spec.shape.rank() != 1) return false;
    if (logits2d_spec.shape.dims[0] != target_spec.shape.dims[0]) return false;

    if (!matchTargetIndexBranch(graph, group, target_index_id)) return false;

    if (!matchLogitsBranch(graph, group, logits2d_id)) return false;
    return true;
}

pub fn tryExecute(
    allocator: std.mem.Allocator,
    graph: *const Graph,
    group: []const Step,
    values: []?*Tensor,
    owned: []bool,
) !bool {
    if (!matchesGroup(graph, group)) return false;

    const loss_node = findLossNode(graph, group) orelse return false;
    const logits2d_id = loss_node.inputs[0];
    const target_index_id = loss_node.inputs[1];
    const out_id = loss_node.outputs[0];

    const transpose_node = requireLogits2dSource(graph, group, logits2d_id) orelse return false;
    const vocab_by_token_id = transpose_node.inputs[0];

    const logits = values[vocab_by_token_id] orelse return error.UnboundGraphValue;
    const targets = values[target_index_id] orelse return error.UnboundGraphValue;
    const device = logits.device() orelse return error.InputNotMaterialized;
    if ((targets.device() orelse return error.InputNotMaterialized) != device) return false;
    if (logits.dtype != .f32 or targets.dtype != .i64) return false;
    if (logits.shape.rank() != 2 or targets.shape.rank() != 1) return false;
    if (!common.rawStorageInputsArePackedDense(&.{ logits, targets })) return false;

    const classes = logits.shape.dims[0];
    const rows = logits.shape.dims[1];
    if (targets.shape.dims[0] != rows) return false;

    const out_spec = graph.values.items[out_id].spec;
    const out = try Tensor.createContiguousWithSource(allocator, out_spec.shape.dims, out_spec.dtype, out_spec.device, false, .graph);
    errdefer out.deinit();

    try kernel_dispatch.crossEntropyIndexedTransposed(
        device,
        out_spec.dtype,
        logits.storage orelse return error.InputNotMaterialized,
        targets.storage orelse return error.InputNotMaterialized,
        out.storage orelse return error.InputNotMaterialized,
        rows,
        classes,
    );

    values[out_id] = out;
    owned[out_id] = true;
    return true;
}

test "lm head cross entropy indexed matcher recognizes tied-head indexed loss shape" {
    const allocator = std.testing.allocator;
    const TensorSpec = @import("../../../types/tensor/tensor_spec.zig").TensorSpec;
    const Layout = @import("../../../types/tensor/layout.zig").Layout;
    const Shape = @import("../../../types/tensor/shape.zig").Shape;
    const SliceRange = @import("../../../types/operation/options.zig").SliceRange;

    var weight_shape = try Shape.initCopy(allocator, &.{ 3, 2 });
    defer weight_shape.deinit();
    var weight_layout = try Layout.initContiguous(allocator, weight_shape);
    defer weight_layout.deinit();
    const weight_spec = TensorSpec{ .shape = weight_shape, .dtype = .f32, .layout = weight_layout, .device = .cpu };

    var hidden_t_shape = try Shape.initCopy(allocator, &.{ 2, 4 });
    defer hidden_t_shape.deinit();
    var hidden_t_layout = try Layout.initContiguous(allocator, hidden_t_shape);
    defer hidden_t_layout.deinit();
    const hidden_t_spec = TensorSpec{ .shape = hidden_t_shape, .dtype = .f32, .layout = hidden_t_layout, .device = .cpu };

    var vocab_by_token_shape = try Shape.initCopy(allocator, &.{ 3, 4 });
    defer vocab_by_token_shape.deinit();
    var vocab_by_token_layout = try Layout.initContiguous(allocator, vocab_by_token_shape);
    defer vocab_by_token_layout.deinit();
    const vocab_by_token_spec = TensorSpec{ .shape = vocab_by_token_shape, .dtype = .f32, .layout = vocab_by_token_layout, .device = .cpu };

    var flat_logits_shape = try Shape.initCopy(allocator, &.{ 4, 3 });
    defer flat_logits_shape.deinit();
    var flat_logits_view_layout = try Layout.initCopy(allocator, &.{ 1, 4 }, 0);
    defer flat_logits_view_layout.deinit();
    const flat_logits_view_spec = TensorSpec{ .shape = flat_logits_shape, .dtype = .f32, .layout = flat_logits_view_layout, .device = .cpu };

    var flat_logits_layout = try Layout.initContiguous(allocator, flat_logits_shape);
    defer flat_logits_layout.deinit();
    const flat_logits_spec = TensorSpec{ .shape = flat_logits_shape, .dtype = .f32, .layout = flat_logits_layout, .device = .cpu };

    var logits3_shape = try Shape.initCopy(allocator, &.{ 2, 2, 3 });
    defer logits3_shape.deinit();
    var logits3_layout = try Layout.initContiguous(allocator, logits3_shape);
    defer logits3_layout.deinit();
    const logits3_spec = TensorSpec{ .shape = logits3_shape, .dtype = .f32, .layout = logits3_layout, .device = .cpu };

    var token_shape = try Shape.initCopy(allocator, &.{ 2, 3 });
    defer token_shape.deinit();
    var token_layout = try Layout.initContiguous(allocator, token_shape);
    defer token_layout.deinit();
    const token_spec = TensorSpec{ .shape = token_shape, .dtype = .i64, .layout = token_layout, .device = .cpu };

    var shifted_shape = try Shape.initCopy(allocator, &.{ 2, 2 });
    defer shifted_shape.deinit();
    var shifted_layout = try Layout.initCopy(allocator, &.{ 3, 1 }, 1);
    defer shifted_layout.deinit();
    const shifted_spec = TensorSpec{ .shape = shifted_shape, .dtype = .i64, .layout = shifted_layout, .device = .cpu };

    var shifted_dense_layout = try Layout.initContiguous(allocator, shifted_shape);
    defer shifted_dense_layout.deinit();
    const shifted_dense_spec = TensorSpec{ .shape = shifted_shape, .dtype = .i64, .layout = shifted_dense_layout, .device = .cpu };

    var flat_targets_shape = try Shape.initCopy(allocator, &.{4});
    defer flat_targets_shape.deinit();
    var flat_targets_layout = try Layout.initContiguous(allocator, flat_targets_shape);
    defer flat_targets_layout.deinit();
    const flat_targets_spec = TensorSpec{ .shape = flat_targets_shape, .dtype = .i64, .layout = flat_targets_layout, .device = .cpu };

    var scalar_shape = try Shape.initCopy(allocator, &.{1});
    defer scalar_shape.deinit();
    var scalar_layout = try Layout.initContiguous(allocator, scalar_shape);
    defer scalar_layout.deinit();
    const scalar_spec = TensorSpec{ .shape = scalar_shape, .dtype = .f32, .layout = scalar_layout, .device = .cpu };

    var graph = Graph.init(allocator);
    defer graph.deinit();

    const weight = try graph.addInput(weight_spec);
    const hidden_t = try graph.addInput(hidden_t_spec);
    const token_ids = try graph.addInput(token_spec);

    const vocab_by_token = try graph.addOp(.matmul, &.{ weight, hidden_t }, .{ .none = {} }, vocab_by_token_spec);
    const flat_logits_view = try graph.addOp(.transpose, &.{vocab_by_token}, .{ .transpose = .{ .permutation = &.{ 1, 0 } } }, flat_logits_view_spec);
    const flat_logits_dense = try graph.addOp(.contiguous, &.{flat_logits_view}, .{ .none = {} }, flat_logits_spec);
    const logits3 = try graph.addOp(.reshape, &.{flat_logits_dense}, .{ .reshape = .{ .shape = &.{ 2, 2, 3 } } }, logits3_spec);
    const flat_logits = try graph.addOp(.reshape, &.{logits3}, .{ .reshape = .{ .shape = &.{ 4, 3 } } }, flat_logits_spec);

    const ranges = [_]SliceRange{
        .{ .start = 0, .stop = 2, .step = 1 },
        .{ .start = 1, .stop = 3, .step = 1 },
    };
    const shifted = try graph.addOp(.slice, &.{token_ids}, .{ .slice = .{ .ranges = &ranges } }, shifted_spec);
    const shifted_dense = try graph.addOp(.contiguous, &.{shifted}, .{ .none = {} }, shifted_dense_spec);
    const flat_targets = try graph.addOp(.reshape, &.{shifted_dense}, .{ .reshape = .{ .shape = &.{4} } }, flat_targets_spec);

    const loss = try graph.addOp(.cross_entropy_indexed, &.{ flat_logits, flat_targets }, .{ .cross_entropy_indexed = .{ .axis = 1 } }, scalar_spec);
    try graph.setOutputs(&.{loss});

    var graph_plan = try @import("../../../plan/graph.zig").create(allocator, &graph);
    defer graph_plan.deinit();

    const region = findRegionContainingProducer(&graph_plan, graph.values.items[loss].producer) orelse return error.TestUnexpectedResult;
    try std.testing.expect(region.step_end > region.step_start);
    const group = graph_plan.steps.items[region.step_start..region.step_end];
    try std.testing.expect(matchesGroup(&graph, group));
    try std.testing.expect(shouldSkipPreparatoryStep(&graph, group, group[0]));
    try std.testing.expect(shouldSkipPreparatoryStep(&graph, group, group[1]));
    try std.testing.expect(shouldSkipPreparatoryStep(&graph, group, group[2]));
    try std.testing.expect(shouldSkipPreparatoryStep(&graph, group, group[3]));
    try std.testing.expect(!shouldSkipPreparatoryStep(&graph, group, group[4]));
    try std.testing.expect(!shouldSkipPreparatoryStep(&graph, group, group[group.len - 1]));
}

test "lm head cross entropy indexed fused execution matches dense indexed loss" {
    const allocator = std.testing.allocator;
    const TensorSpec = @import("../../../types/tensor/tensor_spec.zig").TensorSpec;
    const Layout = @import("../../../types/tensor/layout.zig").Layout;
    const Shape = @import("../../../types/tensor/shape.zig").Shape;
    const SliceRange = @import("../../../types/operation/options.zig").SliceRange;

    var weight_shape = try Shape.initCopy(allocator, &.{ 3, 2 });
    defer weight_shape.deinit();
    var weight_layout = try Layout.initContiguous(allocator, weight_shape);
    defer weight_layout.deinit();
    const weight_spec = TensorSpec{ .shape = weight_shape, .dtype = .f32, .layout = weight_layout, .device = .cpu };

    var hidden_t_shape = try Shape.initCopy(allocator, &.{ 2, 4 });
    defer hidden_t_shape.deinit();
    var hidden_t_layout = try Layout.initContiguous(allocator, hidden_t_shape);
    defer hidden_t_layout.deinit();
    const hidden_t_spec = TensorSpec{ .shape = hidden_t_shape, .dtype = .f32, .layout = hidden_t_layout, .device = .cpu };

    var vocab_by_token_shape = try Shape.initCopy(allocator, &.{ 3, 4 });
    defer vocab_by_token_shape.deinit();
    var vocab_by_token_layout = try Layout.initContiguous(allocator, vocab_by_token_shape);
    defer vocab_by_token_layout.deinit();
    const vocab_by_token_spec = TensorSpec{ .shape = vocab_by_token_shape, .dtype = .f32, .layout = vocab_by_token_layout, .device = .cpu };

    var flat_logits_shape = try Shape.initCopy(allocator, &.{ 4, 3 });
    defer flat_logits_shape.deinit();
    var flat_logits_view_layout = try Layout.initCopy(allocator, &.{ 1, 4 }, 0);
    defer flat_logits_view_layout.deinit();
    const flat_logits_view_spec = TensorSpec{ .shape = flat_logits_shape, .dtype = .f32, .layout = flat_logits_view_layout, .device = .cpu };

    var flat_logits_layout = try Layout.initContiguous(allocator, flat_logits_shape);
    defer flat_logits_layout.deinit();
    const flat_logits_spec = TensorSpec{ .shape = flat_logits_shape, .dtype = .f32, .layout = flat_logits_layout, .device = .cpu };

    var logits3_shape = try Shape.initCopy(allocator, &.{ 2, 2, 3 });
    defer logits3_shape.deinit();
    var logits3_layout = try Layout.initContiguous(allocator, logits3_shape);
    defer logits3_layout.deinit();
    const logits3_spec = TensorSpec{ .shape = logits3_shape, .dtype = .f32, .layout = logits3_layout, .device = .cpu };

    var token_shape = try Shape.initCopy(allocator, &.{ 2, 3 });
    defer token_shape.deinit();
    var token_layout = try Layout.initContiguous(allocator, token_shape);
    defer token_layout.deinit();
    const token_spec = TensorSpec{ .shape = token_shape, .dtype = .i64, .layout = token_layout, .device = .cpu };

    var shifted_shape = try Shape.initCopy(allocator, &.{ 2, 2 });
    defer shifted_shape.deinit();
    var shifted_layout = try Layout.initCopy(allocator, &.{ 3, 1 }, 1);
    defer shifted_layout.deinit();
    const shifted_spec = TensorSpec{ .shape = shifted_shape, .dtype = .i64, .layout = shifted_layout, .device = .cpu };

    var shifted_dense_layout = try Layout.initContiguous(allocator, shifted_shape);
    defer shifted_dense_layout.deinit();
    const shifted_dense_spec = TensorSpec{ .shape = shifted_shape, .dtype = .i64, .layout = shifted_dense_layout, .device = .cpu };

    var flat_targets_shape = try Shape.initCopy(allocator, &.{4});
    defer flat_targets_shape.deinit();
    var flat_targets_layout = try Layout.initContiguous(allocator, flat_targets_shape);
    defer flat_targets_layout.deinit();
    const flat_targets_spec = TensorSpec{ .shape = flat_targets_shape, .dtype = .i64, .layout = flat_targets_layout, .device = .cpu };

    var scalar_shape = try Shape.initCopy(allocator, &.{1});
    defer scalar_shape.deinit();
    var scalar_layout = try Layout.initContiguous(allocator, scalar_shape);
    defer scalar_layout.deinit();
    const scalar_spec = TensorSpec{ .shape = scalar_shape, .dtype = .f32, .layout = scalar_layout, .device = .cpu };

    var graph = Graph.init(allocator);
    defer graph.deinit();

    const weight = try graph.addInput(weight_spec);
    const hidden_t = try graph.addInput(hidden_t_spec);
    const token_ids = try graph.addInput(token_spec);

    const vocab_by_token = try graph.addOp(.matmul, &.{ weight, hidden_t }, .{ .none = {} }, vocab_by_token_spec);
    const flat_logits_view = try graph.addOp(.transpose, &.{vocab_by_token}, .{ .transpose = .{ .permutation = &.{ 1, 0 } } }, flat_logits_view_spec);
    const flat_logits_dense = try graph.addOp(.contiguous, &.{flat_logits_view}, .{ .none = {} }, flat_logits_spec);
    const logits3 = try graph.addOp(.reshape, &.{flat_logits_dense}, .{ .reshape = .{ .shape = &.{ 2, 2, 3 } } }, logits3_spec);
    const flat_logits = try graph.addOp(.reshape, &.{logits3}, .{ .reshape = .{ .shape = &.{ 4, 3 } } }, flat_logits_spec);

    const ranges = [_]SliceRange{
        .{ .start = 0, .stop = 2, .step = 1 },
        .{ .start = 1, .stop = 3, .step = 1 },
    };
    const shifted = try graph.addOp(.slice, &.{token_ids}, .{ .slice = .{ .ranges = &ranges } }, shifted_spec);
    const shifted_dense = try graph.addOp(.contiguous, &.{shifted}, .{ .none = {} }, shifted_dense_spec);
    const flat_targets = try graph.addOp(.reshape, &.{shifted_dense}, .{ .reshape = .{ .shape = &.{4} } }, flat_targets_spec);
    const loss = try graph.addOp(.cross_entropy_indexed, &.{ flat_logits, flat_targets }, .{ .cross_entropy_indexed = .{ .axis = 1 } }, scalar_spec);
    try graph.setOutputs(&.{loss});

    var graph_plan = try @import("../../../plan/graph.zig").create(allocator, &graph);
    defer graph_plan.deinit();

    const region = findRegionContainingProducer(&graph_plan, graph.values.items[loss].producer) orelse return error.TestUnexpectedResult;
    try std.testing.expect(region.step_end > region.step_start);

    const vocab_by_token_value = try Tensor.fromSliceF32(allocator, &.{ 3, 4 }, &.{
        1.0, 0.0, 2.0, 1.0,
        2.0, 1.0, 0.0, -1.0,
        3.0, 2.0, 1.0, 0.0,
    });
    defer vocab_by_token_value.deinit();
    const flat_targets_value = try Tensor.fromSliceI64(allocator, &.{4}, &.{ 2, 1, 0, 0 });
    defer flat_targets_value.deinit();

    const flat_logits_value = try Tensor.fromSliceF32(allocator, &.{ 4, 3 }, &.{
        1.0, 2.0,  3.0,
        0.0, 1.0,  2.0,
        2.0, 0.0,  1.0,
        1.0, -1.0, 0.0,
    });
    defer flat_logits_value.deinit();
    const expected = try Tensor.createContiguousWithSource(allocator, &.{1}, .f32, .cpu, false, .graph);
    defer expected.deinit();
    try kernel_dispatch.crossEntropyIndexed(.cpu, .f32, flat_logits_value.storage.?, flat_targets_value.storage.?, expected.storage.?, 4, 3);

    const values = try allocator.alloc(?*Tensor, graph.values.items.len);
    defer allocator.free(values);
    const owned = try allocator.alloc(bool, graph.values.items.len);
    defer allocator.free(owned);
    @memset(values, null);
    @memset(owned, false);

    values[vocab_by_token] = vocab_by_token_value;
    values[flat_targets] = flat_targets_value;

    try std.testing.expect(try tryExecute(allocator, &graph, graph_plan.steps.items[region.step_start..region.step_end], values, owned));
    defer {
        for (values, owned) |maybe_value, is_owned| {
            if (is_owned and maybe_value != null) maybe_value.?.deinit();
        }
    }

    const actual_bytes = try values[loss].?.storage.?.readableBytes();
    const expected_bytes = try expected.storage.?.readableBytes();
    const actual_vals = std.mem.bytesAsSlice(f32, actual_bytes);
    const expected_vals = std.mem.bytesAsSlice(f32, expected_bytes);
    try std.testing.expectApproxEqAbs(expected_vals[0], actual_vals[0], 1e-5);
}

fn fillBenchmarkLogits(flat: *Tensor, transposed: *Tensor, rows: usize, classes: usize) !void {
    const flat_vals = std.mem.bytesAsSlice(f32, try flat.storage.?.writableBytes());
    const transposed_vals = std.mem.bytesAsSlice(f32, try transposed.storage.?.writableBytes());
    for (0..rows) |row| {
        for (0..classes) |class| {
            const raw: f32 = @floatFromInt(@mod(row * 17 + class * 13, 97));
            const scaled = (raw - 48.0) / 17.0;
            flat_vals[row * classes + class] = scaled;
            transposed_vals[class * rows + row] = scaled;
        }
    }
}

fn fillBenchmarkTargets(targets: *Tensor, rows: usize, classes: usize) !void {
    const target_vals = std.mem.bytesAsSlice(i64, try targets.storage.?.writableBytes());
    for (0..rows) |row| target_vals[row] = @intCast((row * 29 + 7) % classes);
}

test "lm head cross entropy indexed transposed benchmark (opt-in)" {
    if (compat.getenv("AFFON_LM_HEAD_FUSION_BENCH") == null) return error.SkipZigTest;
    if (!metal_common.isAvailable()) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const rows: usize = 512;
    const classes: usize = 50_257;
    const iterations: usize = 20;

    const flat_cpu = try Tensor.createContiguous(allocator, &.{ rows, classes }, .f32, .cpu, false);
    defer flat_cpu.deinit();
    const transposed_cpu = try Tensor.createContiguous(allocator, &.{ classes, rows }, .f32, .cpu, false);
    defer transposed_cpu.deinit();
    try fillBenchmarkLogits(flat_cpu, transposed_cpu, rows, classes);

    const targets_cpu = try Tensor.createContiguous(allocator, &.{rows}, .i64, .cpu, false);
    defer targets_cpu.deinit();
    try fillBenchmarkTargets(targets_cpu, rows, classes);

    const flat = try value_helpers.moveToDevice(allocator, flat_cpu, .metal);
    defer flat.deinit();
    const transposed = try value_helpers.moveToDevice(allocator, transposed_cpu, .metal);
    defer transposed.deinit();
    const targets = try value_helpers.moveToDevice(allocator, targets_cpu, .metal);
    defer targets.deinit();

    const out_dense = try Tensor.createContiguousWithSource(allocator, &.{1}, .f32, .metal, false, .graph);
    defer out_dense.deinit();
    const out_transposed = try Tensor.createContiguousWithSource(allocator, &.{1}, .f32, .metal, false, .graph);
    defer out_transposed.deinit();
    const out_materialized = try Tensor.createContiguousWithSource(allocator, &.{1}, .f32, .metal, false, .graph);
    defer out_materialized.deinit();

    const transposed_storage = transposed.storage orelse return error.InputNotMaterialized;
    transposed_storage.retain();
    defer transposed_storage.release();
    const flat_view_shape = try @import("../../../types/tensor/shape.zig").Shape.initCopy(allocator, &.{ rows, classes });
    const flat_view_layout = try @import("../../../types/tensor/layout.zig").Layout.initCopy(allocator, &.{ 1, @as(isize, @intCast(rows)) }, 0);
    const flat_view = try allocator.create(Tensor);
    defer {
        flat_view.storage = null;
        flat_view.deinit();
    }
    flat_view.* = .{
        .allocator = allocator,
        .shape = flat_view_shape,
        .dtype = .f32,
        .layout = flat_view_layout,
        .storage = transposed_storage,
        .axes = null,
    };

    try kernel_dispatch.crossEntropyIndexed(.metal, .f32, flat.storage.?, targets.storage.?, out_dense.storage.?, rows, classes);
    try kernel_dispatch.crossEntropyIndexedTransposed(.metal, .f32, transposed.storage.?, targets.storage.?, out_transposed.storage.?, rows, classes);
    {
        const materialized = try value_helpers.cloneValue(allocator, flat_view);
        defer materialized.deinit();
        try kernel_dispatch.crossEntropyIndexed(.metal, .f32, materialized.storage.?, targets.storage.?, out_materialized.storage.?, rows, classes);
    }

    const dense_host = try value_helpers.moveToDevice(allocator, out_dense, .cpu);
    defer dense_host.deinit();
    const transposed_host = try value_helpers.moveToDevice(allocator, out_transposed, .cpu);
    defer transposed_host.deinit();
    const materialized_host = try value_helpers.moveToDevice(allocator, out_materialized, .cpu);
    defer materialized_host.deinit();
    const dense_vals = std.mem.bytesAsSlice(f32, try dense_host.storage.?.readableBytes());
    const transposed_vals = std.mem.bytesAsSlice(f32, try transposed_host.storage.?.readableBytes());
    const materialized_vals = std.mem.bytesAsSlice(f32, try materialized_host.storage.?.readableBytes());
    try std.testing.expectApproxEqAbs(dense_vals[0], transposed_vals[0], 1e-5);
    try std.testing.expectApproxEqAbs(dense_vals[0], materialized_vals[0], 1e-5);

    const dense_start_ns = compat.nanoTimestamp();
    for (0..iterations) |_| {
        try kernel_dispatch.crossEntropyIndexed(.metal, .f32, flat.storage.?, targets.storage.?, out_dense.storage.?, rows, classes);
    }
    const dense_ns: u64 = @intCast(@max(compat.nanoTimestamp() - dense_start_ns, 0));

    const transposed_start_ns = compat.nanoTimestamp();
    for (0..iterations) |_| {
        try kernel_dispatch.crossEntropyIndexedTransposed(.metal, .f32, transposed.storage.?, targets.storage.?, out_transposed.storage.?, rows, classes);
    }
    const transposed_ns: u64 = @intCast(@max(compat.nanoTimestamp() - transposed_start_ns, 0));

    const materialized_start_ns = compat.nanoTimestamp();
    for (0..iterations) |_| {
        const materialized = try value_helpers.cloneValue(allocator, flat_view);
        defer materialized.deinit();
        try kernel_dispatch.crossEntropyIndexed(.metal, .f32, materialized.storage.?, targets.storage.?, out_materialized.storage.?, rows, classes);
    }
    const materialized_ns: u64 = @intCast(@max(compat.nanoTimestamp() - materialized_start_ns, 0));

    std.debug.print(
        "lm_head_cross_entropy_indexed bench rows={d} classes={d} iters={d} dense_ms_per_iter={d:.3} transposed_ms_per_iter={d:.3} materialize_plus_dense_ms_per_iter={d:.3}\n",
        .{
            rows,
            classes,
            iterations,
            (@as(f64, @floatFromInt(dense_ns)) / 1_000_000.0) / @as(f64, @floatFromInt(iterations)),
            (@as(f64, @floatFromInt(transposed_ns)) / 1_000_000.0) / @as(f64, @floatFromInt(iterations)),
            (@as(f64, @floatFromInt(materialized_ns)) / 1_000_000.0) / @as(f64, @floatFromInt(iterations)),
        },
    );
}
