const std = @import("std");
const graph_mod = @import("../types/ir/index.zig");
const NodeId = graph_mod.NodeId;
const ValueId = graph_mod.ValueId;
const Graph = graph_mod.Graph;
const tensor = @import("../types/tensor/index.zig");
const ValueSpec = tensor.ValueSpec;
const DType = tensor.DType;
const Device = tensor.Device;
const Shape = tensor.Shape;
const Layout = tensor.Layout;
const semantic = @import("../sema/index.zig");
const ExecutionKind = execution_spec.ExecutionKind;
const SliceRange = @import("../types/operation/options.zig").SliceRange;
const execution_layout = @import("../execution/layout.zig");
const execution_spec = @import("../execution/spec.zig");
const pir = @import("../types/ir/pir/index.zig");

pub const StepKind = pir.graph.StepKind;
pub const Step = pir.graph.Step;
pub const RegionKind = pir.graph.RegionKind;
pub const MatmulEpilogueActivation = pir.graph.MatmulEpilogueActivation;
pub const Region = pir.graph.Region;
pub const GraphPlan = pir.GraphPlan;

pub fn lower(allocator: std.mem.Allocator, graph: *const Graph) !GraphPlan {
    var plan = GraphPlan.init(allocator);
    errdefer plan.deinit();

    for (graph.nodes.items) |node| {
        switch (node.kind) {
            .input, .constant => {},
            .op => |tag| {
                const input_specs = try allocator.alloc(ValueSpec, node.inputs.len);
                defer allocator.free(input_specs);
                for (node.inputs, 0..) |input_id, i| {
                    if (input_id >= graph.values.items.len) return error.InvalidGraphPlan;
                    input_specs[i] = graph.values.items[input_id].spec;
                }

                var inferred = try semantic.inferFromSpecs(allocator, tag, input_specs, node.options);
                errdefer inferred.deinit();
                try validateNodeOutputs(graph, node.id, inferred);
                try plan.steps.append(allocator, .{
                    .node_id = node.id,
                    .kind = .single_op,
                    .semantic_spec = inferred,
                    .execution = executionDecisionsFromSemantic(inferred),
                });
            },
        }
    }
    try discoverFusionRegions(&plan, graph);
    try plan.outputs.appendSlice(allocator, graph.outputs.items);
    return plan;
}

fn executionDecisionsFromSemantic(spec: semantic.OpSpec) pir.graph.Decisions {
    return .{
        .kind = execution_spec.executionKindFromSemantic(spec.kind),
        .allocation = execution_spec.allocationIntentFromSemantic(spec.allocation),
        .input_requirement = execution_spec.inputRequirementFromSemantic(spec.input_requirement),
        .input_layout_decision = execution_layout.inputDecisionForHint(spec.planner_hint),
        .broadcast = spec.broadcast,
        .reduce_to_shape = spec.reduce_to_shape,
    };
}

fn validateNodeOutputs(graph: *const Graph, node_id: NodeId, inferred: semantic.OpSpec) !void {
    const node = graph.nodes.items[node_id];
    for (node.inputs) |input_id| {
        if (input_id >= graph.values.items.len) return error.InvalidGraphPlan;
    }
    for (node.outputs) |output_id| {
        if (output_id >= graph.values.items.len) return error.InvalidGraphPlan;
    }

    const primary = graph.values.items[node.outputs[0]].spec;
    try expectSpecEqual(primary, inferred.shape.dims, inferred.dtype, inferred.layout.strides, inferred.layout.offset, inferred.device);

    if (inferred.secondary_output) |secondary| {
        if (node.outputs.len != 2) return error.InvalidGraphPlan;
        const secondary_value = graph.values.items[node.outputs[1]].spec;
        try expectSpecEqual(secondary_value, secondary.shape.dims, secondary.dtype, secondary.layout.strides, secondary.layout.offset, inferred.device);
    } else if (node.outputs.len != 1) {
        return error.InvalidGraphPlan;
    }
}

fn expectSpecEqual(
    actual: ValueSpec,
    expected_shape: []const usize,
    expected_dtype: DType,
    expected_strides: []const isize,
    expected_offset: usize,
    expected_device: Device,
) !void {
    if (!std.mem.eql(usize, actual.shape.dims, expected_shape)) return error.ShapeMismatch;
    if (actual.dtype != expected_dtype) return error.DTypeMismatch;
    if (!std.mem.eql(isize, actual.layout.strides, expected_strides)) return error.ShapeMismatch;
    if (actual.layout.offset != expected_offset) return error.ShapeMismatch;
    if (actual.device != expected_device) return error.DeviceMismatch;
}

fn validateSliceRange(range: SliceRange, dim_size: usize) !void {
    if (range.step == 0) return error.InvalidSliceStep;
    if (range.step < 0) return error.NegativeSliceStepNotYetSupported;
    if (range.start > dim_size) return error.InvalidSliceBound;
    if (range.stop > dim_size) return error.InvalidSliceBound;
    if (range.start > range.stop) return error.InvalidSliceBound;
}

fn discoverFusionRegions(plan: *GraphPlan, graph: *const Graph) !void {
    var i: usize = 0;
    while (i < plan.steps.items.len) : (i += 1) {
        if (classifyGatherLogsumexpLossRegion(plan, graph, i)) |region| {
            try plan.regions.append(plan.allocator, region);
            i = region.step_end - 1;
            continue;
        }
        if (classifyCausalGatherLogsumexpLossRegion(plan, graph, i)) |region| {
            try plan.regions.append(plan.allocator, region);
            i = region.step_end - 1;
            continue;
        }
        if (classifyLogsumexpLossRegion(plan, graph, i)) |region| {
            try plan.regions.append(plan.allocator, region);
            i = region.step_end - 1;
            continue;
        }
        if (classifyMatmulEpilogueRegion(plan, graph, i)) |region| {
            try plan.regions.append(plan.allocator, region);
            i = region.step_end - 1;
            continue;
        }
        if (classifyLmHeadCrossEntropyIndexedRegion(plan, graph, i)) |region| {
            try plan.regions.append(plan.allocator, region);
            i = region.step_end - 1;
            continue;
        }

        if (!isCoarselyFusableKind(plan.steps.items[i].execution.kind)) continue;
        const run_start = i;
        var run_end = i + 1;
        while (run_end < plan.steps.items.len) : (run_end += 1) {
            if (classifyGatherLogsumexpLossRegion(plan, graph, run_end) != null or
                classifyCausalGatherLogsumexpLossRegion(plan, graph, run_end) != null) break;
            if (classifyLogsumexpLossRegion(plan, graph, run_end) != null) break;
            if (classifyMatmulEpilogueRegion(plan, graph, run_end) != null) break;
            if (classifyLmHeadCrossEntropyIndexedRegion(plan, graph, run_end) != null) break;
            if (!isCoarselyFusableKind(plan.steps.items[run_end].execution.kind)) break;
        }
        if (run_end - run_start >= 2) {
            try plan.regions.append(plan.allocator, .{
                .kind = .fusable_run,
                .step_start = run_start,
                .step_end = run_end,
            });
        }
        i = run_end - 1;
    }
}

fn classifyGatherLogsumexpLossRegion(plan: *const GraphPlan, graph: *const Graph, start: usize) ?Region {
    const tags = &.{ .gather, .max_axis, .sub, .exp, .sum_axis, .log, .add, .sub, .mean_all };
    if (!matchesStepTags(graph, plan.steps.items, start, tags)) return null;
    return .{ .kind = .fusable_run, .step_start = start, .step_end = start + tags.len };
}

fn classifyCausalGatherLogsumexpLossRegion(plan: *const GraphPlan, graph: *const Graph, start: usize) ?Region {
    const tags = &.{ .slice, .reshape, .reshape, .gather, .max_axis, .sub, .exp, .sum_axis, .log, .add, .sub, .mean_all };
    if (!matchesStepTags(graph, plan.steps.items, start, tags)) return null;
    return .{ .kind = .fusable_run, .step_start = start, .step_end = start + tags.len };
}

fn classifyLogsumexpLossRegion(plan: *const GraphPlan, graph: *const Graph, start: usize) ?Region {
    const tags = &.{ .max_axis, .sub, .exp, .sum_axis, .log, .sub, .mul, .sum_axis, .neg, .mean_all };
    if (!matchesStepTags(graph, plan.steps.items, start, tags)) return null;
    return .{ .kind = .fusable_run, .step_start = start, .step_end = start + tags.len };
}

fn isCoarselyFusableKind(kind: ExecutionKind) bool {
    return kind == .elementwise_binary or
        kind == .elementwise_unary or
        kind == .elementwise_generic or
        kind == .reduction or
        kind == .reduction_all or
        kind == .index or
        kind == .view;
}

fn classifyLmHeadCrossEntropyIndexedRegion(plan: *const GraphPlan, graph: *const Graph, start: usize) ?Region {
    const steps = plan.steps.items;
    if (matchesStepTags(graph, steps, start, &.{ .permute, .contiguous, .reshape, .slice, .reshape, .contiguous, .reshape, .cast, .cross_entropy_indexed }) or
        matchesStepTags(graph, steps, start, &.{ .transpose, .contiguous, .reshape, .slice, .reshape, .contiguous, .reshape, .cast, .cross_entropy_indexed }) or
        matchesStepTags(graph, steps, start, &.{ .transpose, .contiguous, .reshape, .reshape, .slice, .contiguous, .reshape, .cross_entropy_indexed }))
    {
        var end = start + 1;
        while (end < steps.len) : (end += 1) {
            const tag = nodeOpTag(graph.nodes.items[steps[end - 1].node_id]) orelse return null;
            if (tag == .cross_entropy_indexed) break;
        }
        return .{
            .kind = .fusable_run,
            .step_start = start,
            .step_end = end,
        };
    }
    return null;
}

fn matchesStepTags(
    graph: *const Graph,
    steps: []const Step,
    start: usize,
    comptime tags: []const @import("../types/operation/tag.zig").OpTag,
) bool {
    if (start + tags.len > steps.len) return false;
    for (tags, 0..) |expected, offset| {
        const step = steps[start + offset];
        if (step.node_id >= graph.nodes.items.len) return false;
        if ((nodeOpTag(graph.nodes.items[step.node_id]) orelse return false) != expected) return false;
    }
    return true;
}

fn classifyMatmulEpilogueRegion(plan: *const GraphPlan, graph: *const Graph, start: usize) ?Region {
    if (start + 1 >= plan.steps.items.len) return null;
    const steps = plan.steps.items;
    const mm_step = steps[start];
    const add_step = steps[start + 1];
    if (mm_step.execution.kind != .reduction or add_step.execution.kind != .elementwise_binary) return null;

    if (mm_step.node_id >= graph.nodes.items.len or add_step.node_id >= graph.nodes.items.len) return null;
    const mm_node = graph.nodes.items[mm_step.node_id];
    const add_node = graph.nodes.items[add_step.node_id];
    const mm_tag = nodeOpTag(mm_node) orelse return null;
    const add_tag = nodeOpTag(add_node) orelse return null;
    if (mm_tag != .matmul or add_tag != .add) return null;
    if (mm_node.inputs.len != 2 or mm_node.outputs.len != 1) return null;
    if (add_node.inputs.len != 2 or add_node.outputs.len != 1) return null;

    const mm_out_id = mm_node.outputs[0];
    if (add_node.inputs[0] != mm_out_id and add_node.inputs[1] != mm_out_id) return null;
    if (mm_out_id >= graph.values.items.len or add_node.outputs[0] >= graph.values.items.len) return null;
    const bias_id = if (add_node.inputs[0] == mm_out_id) add_node.inputs[1] else add_node.inputs[0];
    if (!isInputOrConstantValue(graph, bias_id)) return null;
    const add_out_spec = graph.values.items[add_node.outputs[0]].spec;
    const mm_out_spec = graph.values.items[mm_out_id].spec;
    if (!std.mem.eql(usize, add_out_spec.shape.dims, mm_out_spec.shape.dims)) return null;

    if (!hasOnlyExpectedConsumer(graph, mm_out_id, add_node.id)) return null;

    if (start + 2 < steps.len) {
        const activation_step = steps[start + 2];
        if (activation_step.execution.kind == .elementwise_unary) {
            if (activation_step.node_id >= graph.nodes.items.len) return null;
            const activation_node = graph.nodes.items[activation_step.node_id];
            const activation_tag = nodeOpTag(activation_node) orelse return null;
            const activation = switch (activation_tag) {
                .gelu => MatmulEpilogueActivation.gelu,
                .relu => MatmulEpilogueActivation.relu,
                .sigmoid => MatmulEpilogueActivation.sigmoid,
                else => null,
            };
            if (activation) |matched_activation| {
                if (activation_node.inputs.len != 1 or activation_node.outputs.len != 1) return null;
                const add_out_id = add_node.outputs[0];
                if (activation_node.inputs[0] != add_out_id) return null;
                if (activation_node.outputs[0] >= graph.values.items.len or add_out_id >= graph.values.items.len) return null;
                const activation_out_spec = graph.values.items[activation_node.outputs[0]].spec;
                const add_out_spec2 = graph.values.items[add_out_id].spec;
                if (!std.mem.eql(usize, activation_out_spec.shape.dims, add_out_spec2.shape.dims)) return null;
                if (!hasOnlyExpectedConsumer(graph, add_out_id, activation_node.id)) return null;
                if (graphOutputContains(graph, mm_out_id) or graphOutputContains(graph, add_out_id)) return null;
                return .{
                    .kind = .matmul_epilogue,
                    .step_start = start,
                    .step_end = start + 3,
                    .matmul_epilogue_activation = matched_activation,
                };
            }
        }
    }
    if (graphOutputContains(graph, mm_out_id)) return null;

    return .{
        .kind = .matmul_epilogue,
        .step_start = start,
        .step_end = start + 2,
        .matmul_epilogue_activation = .none,
    };
}

fn nodeOpTag(node: graph_mod.Node) ?@import("../types/operation/tag.zig").OpTag {
    return switch (node.kind) {
        .op => |tag| tag,
        else => null,
    };
}

fn hasOnlyExpectedConsumer(graph: *const Graph, value_id: ValueId, expected_node_id: NodeId) bool {
    var consumer_count: usize = 0;
    var expected_is_consumer = false;
    for (graph.nodes.items) |node| {
        for (node.inputs) |input_id| {
            if (input_id == value_id) {
                consumer_count += 1;
                if (node.id == expected_node_id) expected_is_consumer = true;
            }
        }
    }
    return consumer_count == 1 and expected_is_consumer;
}

fn isInputOrConstantValue(graph: *const Graph, value_id: ValueId) bool {
    if (value_id >= graph.values.items.len) return false;
    const producer_id = graph.values.items[value_id].producer;
    if (producer_id >= graph.nodes.items.len) return false;
    return switch (graph.nodes.items[producer_id].kind) {
        .input, .constant => true,
        .op => false,
    };
}

fn graphOutputContains(graph: *const Graph, value_id: ValueId) bool {
    for (graph.outputs.items) |output_id| {
        if (output_id == value_id) return true;
    }
    return false;
}

test "graph plan lowers op nodes into single-op steps" {
    const allocator = std.testing.allocator;
    var graph = Graph.init(allocator);
    defer graph.deinit();

    var shape = try Shape.initCopy(allocator, &.{3});
    defer shape.deinit();
    var layout = try Layout.initContiguous(allocator, shape);
    defer layout.deinit();
    const spec = ValueSpec{
        .shape = shape,
        .dtype = .f32,
        .layout = layout,
        .device = .cpu,
    };

    const a = try graph.addInput(spec);
    const b = try graph.addInput(spec);
    const c = try graph.addOp(.add, &.{ a, b }, .{ .none = {} }, spec);
    try graph.setOutputs(&.{c});

    var plan = try lower(allocator, &graph);
    defer plan.deinit();

    try std.testing.expectEqual(@as(usize, 1), plan.steps.items.len);
    try std.testing.expectEqual(StepKind.single_op, plan.steps.items[0].kind);
    try std.testing.expectEqual(ExecutionKind.elementwise_binary, plan.steps.items[0].execution.kind);
    try std.testing.expectEqual(execution_layout.InputLayoutDecision.accept, plan.steps.items[0].execution.input_layout_decision);
    try std.testing.expectEqual(@as(usize, 1), plan.outputs.items.len);
}

test "graph plan records matmul transposed input as accepted layout" {
    const allocator = std.testing.allocator;
    var graph = Graph.init(allocator);
    defer graph.deinit();

    var lhs_shape = try Shape.initCopy(allocator, &.{ 3, 2 });
    defer lhs_shape.deinit();
    var lhs_layout = try Layout.initCopy(allocator, &.{ 1, 3 }, 0);
    defer lhs_layout.deinit();
    const lhs_spec = ValueSpec{
        .shape = lhs_shape,
        .dtype = .f32,
        .layout = lhs_layout,
        .device = .cpu,
    };

    var rhs_shape = try Shape.initCopy(allocator, &.{ 2, 4 });
    defer rhs_shape.deinit();
    var rhs_layout = try Layout.initContiguous(allocator, rhs_shape);
    defer rhs_layout.deinit();
    const rhs_spec = ValueSpec{
        .shape = rhs_shape,
        .dtype = .f32,
        .layout = rhs_layout,
        .device = .cpu,
    };

    var out_shape = try Shape.initCopy(allocator, &.{ 3, 4 });
    defer out_shape.deinit();
    var out_layout = try Layout.initContiguous(allocator, out_shape);
    defer out_layout.deinit();
    const out_spec = ValueSpec{
        .shape = out_shape,
        .dtype = .f32,
        .layout = out_layout,
        .device = .cpu,
    };

    const lhs = try graph.addInput(lhs_spec);
    const rhs = try graph.addInput(rhs_spec);
    const out = try graph.addOp(.matmul, &.{ lhs, rhs }, .{ .none = {} }, out_spec);
    try graph.setOutputs(&.{out});

    var plan = try lower(allocator, &graph);
    defer plan.deinit();

    try std.testing.expectEqual(@as(usize, 1), plan.steps.items.len);
    try std.testing.expectEqual(execution_layout.InputLayoutDecision.accept, plan.steps.items[0].execution.input_layout_decision);
}

test "graph plan records reduction pack_to_dense input hint" {
    const allocator = std.testing.allocator;
    var graph = Graph.init(allocator);
    defer graph.deinit();

    var input_shape = try Shape.initCopy(allocator, &.{ 3, 2 });
    defer input_shape.deinit();
    var input_layout = try Layout.initCopy(allocator, &.{ -2, 1 }, 4);
    defer input_layout.deinit();
    const input_spec = ValueSpec{
        .shape = input_shape,
        .dtype = .f32,
        .layout = input_layout,
        .device = .cpu,
    };

    var out_shape = try Shape.initCopy(allocator, &.{3});
    defer out_shape.deinit();
    var out_layout = try Layout.initContiguous(allocator, out_shape);
    defer out_layout.deinit();
    const out_spec = ValueSpec{
        .shape = out_shape,
        .dtype = .f32,
        .layout = out_layout,
        .device = .cpu,
    };

    const input = try graph.addInput(input_spec);
    const out = try graph.addOp(.sum_axis, &.{input}, .{ .reduce_axis = .{ .axis = 1, .keepdim = false } }, out_spec);
    try graph.setOutputs(&.{out});

    var plan = try lower(allocator, &graph);
    defer plan.deinit();
    try std.testing.expectEqual(execution_layout.InputLayoutDecision.pack_to_dense, plan.steps.items[0].execution.input_layout_decision);
}

test "graph plan marks fusion groups for consecutive elementwise steps" {
    const allocator = std.testing.allocator;
    var graph = Graph.init(allocator);
    defer graph.deinit();

    var shape = try Shape.initCopy(allocator, &.{3});
    defer shape.deinit();
    var layout = try Layout.initContiguous(allocator, shape);
    defer layout.deinit();
    const spec = ValueSpec{
        .shape = shape,
        .dtype = .f32,
        .layout = layout,
        .device = .cpu,
    };

    const a = try graph.addInput(spec);
    const b = try graph.addInput(spec);
    const add_out = try graph.addOp(.add, &.{ a, b }, .{ .none = {} }, spec);
    const relu_out = try graph.addOp(.relu, &.{add_out}, .{ .none = {} }, spec);
    var scalar_shape = try Shape.initCopy(allocator, &.{1});
    defer scalar_shape.deinit();
    var scalar_layout = try Layout.initContiguous(allocator, scalar_shape);
    defer scalar_layout.deinit();
    const scalar_spec = ValueSpec{
        .shape = scalar_shape,
        .dtype = .f32,
        .layout = scalar_layout,
        .device = .cpu,
    };
    const sum_out = try graph.addOp(.sum_all, &.{relu_out}, .{ .none = {} }, scalar_spec);
    try graph.setOutputs(&.{sum_out});

    var plan = try lower(allocator, &graph);
    defer plan.deinit();
    try std.testing.expectEqual(@as(usize, 1), plan.regions.items.len);
    try std.testing.expectEqual(RegionKind.fusable_run, plan.regions.items[0].kind);
    try std.testing.expectEqual(@as(usize, 0), plan.regions.items[0].step_start);
    try std.testing.expectEqual(@as(usize, 3), plan.regions.items[0].step_end);
}

test "graph plan recognizes matmul epilogue region with gelu activation" {
    const allocator = std.testing.allocator;
    var graph = Graph.init(allocator);
    defer graph.deinit();

    var a_shape = try Shape.initCopy(allocator, &.{ 2, 3 });
    defer a_shape.deinit();
    var a_layout = try Layout.initContiguous(allocator, a_shape);
    defer a_layout.deinit();
    const a_spec = ValueSpec{ .shape = a_shape, .dtype = .f32, .layout = a_layout, .device = .cpu };

    var b_shape = try Shape.initCopy(allocator, &.{ 3, 2 });
    defer b_shape.deinit();
    var b_layout = try Layout.initContiguous(allocator, b_shape);
    defer b_layout.deinit();
    const b_spec = ValueSpec{ .shape = b_shape, .dtype = .f32, .layout = b_layout, .device = .cpu };

    var out_shape = try Shape.initCopy(allocator, &.{ 2, 2 });
    defer out_shape.deinit();
    var out_layout = try Layout.initContiguous(allocator, out_shape);
    defer out_layout.deinit();
    const out_spec = ValueSpec{ .shape = out_shape, .dtype = .f32, .layout = out_layout, .device = .cpu };

    const a = try graph.addInput(a_spec);
    const b = try graph.addInput(b_spec);
    const bias = try graph.addInput(out_spec);
    const mm = try graph.addOp(.matmul, &.{ a, b }, .{ .none = {} }, out_spec);
    const add = try graph.addOp(.add, &.{ mm, bias }, .{ .none = {} }, out_spec);
    const gelu = try graph.addOp(.gelu, &.{add}, .{ .none = {} }, out_spec);
    try graph.setOutputs(&.{gelu});

    var plan = try lower(allocator, &graph);
    defer plan.deinit();

    try std.testing.expectEqual(@as(usize, 1), plan.regions.items.len);
    try std.testing.expectEqual(RegionKind.matmul_epilogue, plan.regions.items[0].kind);
    try std.testing.expectEqual(@as(usize, 0), plan.regions.items[0].step_start);
    try std.testing.expectEqual(@as(usize, 3), plan.regions.items[0].step_end);
    try std.testing.expectEqual(MatmulEpilogueActivation.gelu, plan.regions.items[0].matmul_epilogue_activation.?);
}

test "graph plan recognizes matmul epilogue region with broadcast bias" {
    const allocator = std.testing.allocator;
    var graph = Graph.init(allocator);
    defer graph.deinit();

    var a_shape = try Shape.initCopy(allocator, &.{ 2, 3 });
    defer a_shape.deinit();
    var a_layout = try Layout.initContiguous(allocator, a_shape);
    defer a_layout.deinit();
    const a_spec = ValueSpec{ .shape = a_shape, .dtype = .f32, .layout = a_layout, .device = .cpu };

    var b_shape = try Shape.initCopy(allocator, &.{ 3, 2 });
    defer b_shape.deinit();
    var b_layout = try Layout.initContiguous(allocator, b_shape);
    defer b_layout.deinit();
    const b_spec = ValueSpec{ .shape = b_shape, .dtype = .f32, .layout = b_layout, .device = .cpu };

    var out_shape = try Shape.initCopy(allocator, &.{ 2, 2 });
    defer out_shape.deinit();
    var out_layout = try Layout.initContiguous(allocator, out_shape);
    defer out_layout.deinit();
    const out_spec = ValueSpec{ .shape = out_shape, .dtype = .f32, .layout = out_layout, .device = .cpu };

    var bias_shape = try Shape.initCopy(allocator, &.{ 1, 2 });
    defer bias_shape.deinit();
    var bias_layout = try Layout.initContiguous(allocator, bias_shape);
    defer bias_layout.deinit();
    const bias_spec = ValueSpec{ .shape = bias_shape, .dtype = .f32, .layout = bias_layout, .device = .cpu };

    const a = try graph.addInput(a_spec);
    const b = try graph.addInput(b_spec);
    const bias = try graph.addInput(bias_spec);
    const mm = try graph.addOp(.matmul, &.{ a, b }, .{ .none = {} }, out_spec);
    const add = try graph.addOp(.add, &.{ mm, bias }, .{ .none = {} }, out_spec);
    try graph.setOutputs(&.{add});

    var plan = try lower(allocator, &graph);
    defer plan.deinit();

    try std.testing.expectEqual(@as(usize, 1), plan.regions.items.len);
    try std.testing.expectEqual(RegionKind.matmul_epilogue, plan.regions.items[0].kind);
    try std.testing.expectEqual(@as(usize, 0), plan.regions.items[0].step_start);
    try std.testing.expectEqual(@as(usize, 2), plan.regions.items[0].step_end);
    try std.testing.expectEqual(MatmulEpilogueActivation.none, plan.regions.items[0].matmul_epilogue_activation.?);
}

test "graph plan recognizes matmul epilogue when add operands are canonicalized" {
    const allocator = std.testing.allocator;
    var graph = Graph.init(allocator);
    defer graph.deinit();

    var a_shape = try Shape.initCopy(allocator, &.{ 2, 3 });
    defer a_shape.deinit();
    var a_layout = try Layout.initContiguous(allocator, a_shape);
    defer a_layout.deinit();
    const a_spec = ValueSpec{ .shape = a_shape, .dtype = .f32, .layout = a_layout, .device = .cpu };

    var b_shape = try Shape.initCopy(allocator, &.{ 3, 2 });
    defer b_shape.deinit();
    var b_layout = try Layout.initContiguous(allocator, b_shape);
    defer b_layout.deinit();
    const b_spec = ValueSpec{ .shape = b_shape, .dtype = .f32, .layout = b_layout, .device = .cpu };

    var out_shape = try Shape.initCopy(allocator, &.{ 2, 2 });
    defer out_shape.deinit();
    var out_layout = try Layout.initContiguous(allocator, out_shape);
    defer out_layout.deinit();
    const out_spec = ValueSpec{ .shape = out_shape, .dtype = .f32, .layout = out_layout, .device = .cpu };

    const a = try graph.addInput(a_spec);
    const b = try graph.addInput(b_spec);
    const bias = try graph.addInput(out_spec);
    const mm = try graph.addOp(.matmul, &.{ a, b }, .{ .none = {} }, out_spec);
    const add = try graph.addOp(.add, &.{ bias, mm }, .{ .none = {} }, out_spec);
    const gelu = try graph.addOp(.gelu, &.{add}, .{ .none = {} }, out_spec);
    try graph.setOutputs(&.{gelu});

    var plan = try lower(allocator, &graph);
    defer plan.deinit();

    try std.testing.expectEqual(@as(usize, 1), plan.regions.items.len);
    try std.testing.expectEqual(RegionKind.matmul_epilogue, plan.regions.items[0].kind);
    try std.testing.expectEqual(@as(usize, 0), plan.regions.items[0].step_start);
    try std.testing.expectEqual(@as(usize, 3), plan.regions.items[0].step_end);
    try std.testing.expectEqual(MatmulEpilogueActivation.gelu, plan.regions.items[0].matmul_epilogue_activation.?);
}

test "graph plan keeps harmless view steps inside fusion groups" {
    const allocator = std.testing.allocator;
    var graph = Graph.init(allocator);
    defer graph.deinit();

    var input_shape = try Shape.initCopy(allocator, &.{ 2, 3 });
    defer input_shape.deinit();
    var input_layout = try Layout.initContiguous(allocator, input_shape);
    defer input_layout.deinit();
    const input_spec = ValueSpec{
        .shape = input_shape,
        .dtype = .f32,
        .layout = input_layout,
        .device = .cpu,
    };

    var rhs_shape = try Shape.initCopy(allocator, &.{ 3, 1 });
    defer rhs_shape.deinit();
    var rhs_layout = try Layout.initContiguous(allocator, rhs_shape);
    defer rhs_layout.deinit();
    const rhs_spec = ValueSpec{
        .shape = rhs_shape,
        .dtype = .f32,
        .layout = rhs_layout,
        .device = .cpu,
    };

    var reshaped_shape = try Shape.initCopy(allocator, &.{ 3, 2 });
    defer reshaped_shape.deinit();
    var reshaped_layout = try Layout.initContiguous(allocator, reshaped_shape);
    defer reshaped_layout.deinit();
    const reshaped_spec = ValueSpec{
        .shape = reshaped_shape,
        .dtype = .f32,
        .layout = reshaped_layout,
        .device = .cpu,
    };

    var out_shape = try Shape.initCopy(allocator, &.{ 3, 1 });
    defer out_shape.deinit();
    var out_layout = try Layout.initContiguous(allocator, out_shape);
    defer out_layout.deinit();
    const out_spec = ValueSpec{
        .shape = out_shape,
        .dtype = .f32,
        .layout = out_layout,
        .device = .cpu,
    };

    const input = try graph.addInput(input_spec);
    const rhs = try graph.addInput(rhs_spec);
    const reshaped = try graph.addOp(.reshape, &.{input}, .{ .reshape = .{ .shape = &.{ 3, 2 } } }, reshaped_spec);
    const reduced = try graph.addOp(.sum_axis, &.{reshaped}, .{ .reduce_axis = .{ .axis = 1, .keepdim = true } }, out_spec);
    const logits = try graph.addOp(.sub, &.{ reduced, rhs }, .{ .none = {} }, out_spec);
    try graph.setOutputs(&.{logits});

    var plan = try lower(allocator, &graph);
    defer plan.deinit();

    try std.testing.expectEqual(@as(usize, 3), plan.steps.items.len);
    try std.testing.expectEqual(@as(usize, 1), plan.regions.items.len);
    try std.testing.expectEqual(@as(usize, 0), plan.regions.items[0].step_start);
    try std.testing.expectEqual(@as(usize, 3), plan.regions.items[0].step_end);
}

test "graph plan validates topk requires two outputs" {
    const allocator = std.testing.allocator;
    var graph = Graph.init(allocator);
    defer graph.deinit();

    var shape = try Shape.initCopy(allocator, &.{ 2, 3 });
    defer shape.deinit();
    var layout = try Layout.initContiguous(allocator, shape);
    defer layout.deinit();
    const in_spec = ValueSpec{
        .shape = shape,
        .dtype = .f32,
        .layout = layout,
        .device = .cpu,
    };
    var out_shape = try Shape.initCopy(allocator, &.{ 2, 2 });
    defer out_shape.deinit();
    var out_layout = try Layout.initContiguous(allocator, out_shape);
    defer out_layout.deinit();
    const out_spec = ValueSpec{
        .shape = out_shape,
        .dtype = .f32,
        .layout = out_layout,
        .device = .cpu,
    };

    const x = try graph.addInput(in_spec);
    _ = try graph.addOp(.topk, &.{x}, .{ .topk = .{ .k = 2, .axis = 1 } }, out_spec);
    try graph.setOutputs(&.{1});

    try std.testing.expectError(error.InvalidGraphPlan, lower(allocator, &graph));
}

test "graph plan accepts topk with two outputs" {
    const allocator = std.testing.allocator;
    var graph = Graph.init(allocator);
    defer graph.deinit();

    var shape = try Shape.initCopy(allocator, &.{ 2, 3 });
    defer shape.deinit();
    var layout = try Layout.initContiguous(allocator, shape);
    defer layout.deinit();
    const in_spec = ValueSpec{
        .shape = shape,
        .dtype = .f32,
        .layout = layout,
        .device = .cpu,
    };
    var out_shape = try Shape.initCopy(allocator, &.{ 2, 2 });
    defer out_shape.deinit();
    var out_layout = try Layout.initContiguous(allocator, out_shape);
    defer out_layout.deinit();
    const values_spec = ValueSpec{
        .shape = out_shape,
        .dtype = .f32,
        .layout = out_layout,
        .device = .cpu,
    };
    const indices_spec = ValueSpec{
        .shape = out_shape,
        .dtype = .i64,
        .layout = out_layout,
        .device = .cpu,
    };

    const x = try graph.addInput(in_spec);
    const ids = try graph.addOpMulti(.topk, &.{x}, .{ .topk = .{ .k = 2, .axis = 1 } }, &.{ values_spec, indices_spec });
    defer allocator.free(ids);
    try graph.setOutputs(ids);

    var plan = try lower(allocator, &graph);
    defer plan.deinit();
    try std.testing.expectEqual(@as(usize, 1), plan.steps.items.len);
}

test "graph plan rejects softmax axis out of bounds" {
    const allocator = std.testing.allocator;
    var graph = Graph.init(allocator);
    defer graph.deinit();

    var shape = try Shape.initCopy(allocator, &.{ 2, 2 });
    defer shape.deinit();
    var layout = try Layout.initContiguous(allocator, shape);
    defer layout.deinit();
    const spec = ValueSpec{
        .shape = shape,
        .dtype = .f32,
        .layout = layout,
        .device = .cpu,
    };

    const x = try graph.addInput(spec);
    _ = try graph.addOp(.softmax, &.{x}, .{ .softmax = .{ .axis = 2 } }, spec);
    try graph.setOutputs(&.{1});
    try std.testing.expectError(error.InvalidAxis, lower(allocator, &graph));
}

test "graph plan rejects topk k greater than axis dim" {
    const allocator = std.testing.allocator;
    var graph = Graph.init(allocator);
    defer graph.deinit();

    var shape = try Shape.initCopy(allocator, &.{ 2, 3 });
    defer shape.deinit();
    var layout = try Layout.initContiguous(allocator, shape);
    defer layout.deinit();
    const in_spec = ValueSpec{
        .shape = shape,
        .dtype = .f32,
        .layout = layout,
        .device = .cpu,
    };
    var out_shape = try Shape.initCopy(allocator, &.{ 2, 4 });
    defer out_shape.deinit();
    var out_layout = try Layout.initContiguous(allocator, out_shape);
    defer out_layout.deinit();
    const values_spec = ValueSpec{
        .shape = out_shape,
        .dtype = .f32,
        .layout = out_layout,
        .device = .cpu,
    };
    const indices_spec = ValueSpec{
        .shape = out_shape,
        .dtype = .i64,
        .layout = out_layout,
        .device = .cpu,
    };

    const x = try graph.addInput(in_spec);
    const ids = try graph.addOpMulti(.topk, &.{x}, .{ .topk = .{ .k = 4, .axis = 1 } }, &.{ values_spec, indices_spec });
    defer allocator.free(ids);
    try graph.setOutputs(ids);
    try std.testing.expectError(error.InvalidTopK, lower(allocator, &graph));
}

test "graph plan rejects slice with invalid bounds" {
    const allocator = std.testing.allocator;
    var graph = Graph.init(allocator);
    defer graph.deinit();

    var in_shape = try Shape.initCopy(allocator, &.{ 2, 3 });
    defer in_shape.deinit();
    var in_layout = try Layout.initContiguous(allocator, in_shape);
    defer in_layout.deinit();
    const in_spec = ValueSpec{
        .shape = in_shape,
        .dtype = .f32,
        .layout = in_layout,
        .device = .cpu,
    };
    var out_shape = try Shape.initCopy(allocator, &.{ 2, 3 });
    defer out_shape.deinit();
    var out_layout = try Layout.initContiguous(allocator, out_shape);
    defer out_layout.deinit();
    const out_spec = ValueSpec{
        .shape = out_shape,
        .dtype = .f32,
        .layout = out_layout,
        .device = .cpu,
    };

    const x = try graph.addInput(in_spec);
    const ranges = [_]SliceRange{.{ .start = 0, .stop = 4, .step = 1 }};
    _ = try graph.addOp(.slice, &.{x}, .{ .slice = .{ .ranges = &ranges } }, out_spec);
    try graph.setOutputs(&.{1});
    try std.testing.expectError(error.InvalidSlice, lower(allocator, &graph));
}
