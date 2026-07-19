const std = @import("std");
const Tensor = @import("../../types/tensor/tensor.zig").Tensor;
const Graph = @import("../../types/ir/graph.zig").Graph;
const GraphPlan = @import("../../types/ir/plan.zig").GraphPlan;
const Op = @import("../../types/operation/op.zig").Op;
const eager = @import("../eager/index.zig");
const fusion = @import("fusion/index.zig");
const telemetry = @import("../../telemetry.zig");
const autograd = @import("../autograd.zig");
const grad_mode = @import("../../grad_mode.zig");

fn fusionMetric(name: []const u8) void {
    telemetry.addCounter(.execution, name, 1);
}

fn emitFusionHit(scope: telemetry.Scope, hit: fusion.Hit) void {
    if (fusion.metricName(hit)) |name| fusionMetric(name);
    if (fusion.traceEventName(hit)) |name| {
        scope.addEventNow(name, &.{});
    }
}

fn emitFusionMiss(scope: telemetry.Scope, miss: fusion.Miss) void {
    if (fusion.missMetricName(miss)) |name| fusionMetric(name);
    if (fusion.missTraceEventName(miss)) |name| {
        scope.addEventNow(name, &.{});
    }
}

fn emitRegionMetric(scope: telemetry.Scope, kind: @import("../../types/ir/plan.zig").RegionKind, prefix: []const u8) void {
    const suffix = switch (kind) {
        .fusable_run => "fusable_run",
        .matmul_epilogue => "matmul_epilogue",
    };
    var name_buffer: [64]u8 = undefined;
    const name = std.fmt.bufPrint(&name_buffer, "{s}_region_{s}", .{ prefix, suffix }) catch return;
    fusionMetric(name);
    var event_buffer: [64]u8 = undefined;
    const event = std.fmt.bufPrint(&event_buffer, "fusion_{s}_region_{s}", .{ prefix, suffix }) catch return;
    scope.addEventNow(event, &.{});
}

pub const Result = struct {
    allocator: std.mem.Allocator,
    values: []?*Tensor,
    owned: []bool,
    outputs: []*Tensor,

    pub fn deinit(self: *Result) void {
        for (self.values, self.owned) |value, is_owned| if (is_owned) if (value) |owned_value| owned_value.deinit();
        self.allocator.free(self.outputs);
        self.allocator.free(self.owned);
        self.allocator.free(self.values);
        self.* = undefined;
    }
};

pub fn execute(
    allocator: std.mem.Allocator,
    graph: *const Graph,
    plan: *const GraphPlan,
    inputs: []const *Tensor,
) !Result {
    if (inputs.len != graph.inputs.items.len) return error.InputCountMismatch;

    var graph_scope = telemetry.beginTrace(.execution, telemetry.traces.run);
    var graph_succeeded = false;
    defer if (!graph_succeeded) graph_scope.endError();

    const values = try allocator.alloc(?*Tensor, graph.values.items.len);
    var result_initialized = false;
    errdefer if (!result_initialized) allocator.free(values);
    @memset(values, null);

    const owned = try allocator.alloc(bool, graph.values.items.len);
    errdefer if (!result_initialized) allocator.free(owned);
    @memset(owned, false);

    const remaining_uses = try allocator.alloc(usize, graph.values.items.len);
    defer allocator.free(remaining_uses);
    @memset(remaining_uses, 0);
    const is_graph_output = try allocator.alloc(bool, graph.values.items.len);
    defer allocator.free(is_graph_output);
    @memset(is_graph_output, false);
    for (graph.nodes.items) |node| {
        for (node.inputs) |input_id| remaining_uses[input_id] += 1;
    }
    for (graph.outputs.items) |output_id| is_graph_output[output_id] = true;

    for (graph.inputs.items, inputs) |value_id, input| values[value_id] = input;

    var result = Result{
        .allocator = allocator,
        .values = values,
        .owned = owned,
        .outputs = try allocator.alloc(*Tensor, 0),
    };
    result_initialized = true;
    errdefer result.deinit();

    var step_index: usize = 0;
    var region_index: usize = 0;
    while (step_index < plan.steps.items.len) {
        if (!grad_mode.isEnabled() and region_index < plan.regions.items.len and plan.regions.items[region_index].step_start == step_index) {
            const region = plan.regions.items[region_index];
            const region_steps = plan.steps.items[region.step_start..region.step_end];
            region_index += 1;
            fusionMetric("groups_eligible");
            var region_scope = graph_scope.child(telemetry.traces.region, .internal, &.{});
            emitRegionMetric(region_scope, region.kind, "eligible");
            try prepareFusionPrefix(allocator, graph, region_scope, region, region_steps, values, owned);
            const outcome = try fusion.execute(allocator, graph, region, region_steps, values, owned);
            if (outcome.hit != .none) {
                emitFusionHit(region_scope, outcome.hit);
                region_scope.end();
                consumeInputs(graph, region_steps, values, owned, remaining_uses, is_graph_output);
                step_index = region.step_end;
                continue;
            }
            emitFusionMiss(region_scope, outcome.miss);
            fusionMetric("fallback");
            emitRegionMetric(region_scope, region.kind, "fallback");
            var fallback_scope = region_scope.child(telemetry.traces.fallback, .internal, &.{});
            fallback_scope.end();
            region_scope.end();
        }

        const step = plan.steps.items[step_index];
        const node = graph.nodes.items[step.node_id];
        if (stepOutputsMaterialized(graph, step, values)) {
            consumeInputs(graph, &.{step}, values, owned, remaining_uses, is_graph_output);
            step_index += 1;
            continue;
        }
        const tag = switch (node.kind) {
            .op => |op_tag| op_tag,
            else => return error.InvalidGraphStep,
        };
        try executeStep(allocator, graph, graph_scope, step, tag, values, owned);
        consumeInputs(graph, &.{step}, values, owned, remaining_uses, is_graph_output);
        step_index += 1;
    }

    result.outputs = try allocator.alloc(*Tensor, plan.outputs.items.len);
    for (plan.outputs.items, result.outputs) |value_id, *output| {
        output.* = values[value_id] orelse return error.MissingGraphValue;
    }
    graph_succeeded = true;
    graph_scope.end();
    return result;
}

fn isFusionPreparatoryTag(tag: @import("../../types/operation/tag.zig").OpTag) bool {
    return switch (tag) {
        .slice, .cast, .contiguous, .reshape, .permute, .transpose, .squeeze, .unsqueeze => true,
        else => false,
    };
}

fn stepOutputsMaterialized(graph: *const Graph, step: @import("../../types/ir/plan.zig").Step, values: []?*Tensor) bool {
    for (graph.nodes.items[step.node_id].outputs) |output_id| {
        if (values[output_id] == null) return false;
    }
    return true;
}

fn stepInputsBound(graph: *const Graph, step: @import("../../types/ir/plan.zig").Step, values: []?*Tensor) bool {
    for (graph.nodes.items[step.node_id].inputs) |input_id| {
        if (values[input_id] == null) return false;
    }
    return true;
}

fn prepareFusionPrefix(
    allocator: std.mem.Allocator,
    graph: *const Graph,
    parent: telemetry.Scope,
    region: @import("../../types/ir/plan.zig").Region,
    steps: []const @import("../../types/ir/plan.zig").Step,
    values: []?*Tensor,
    owned: []bool,
) !void {
    for (steps) |step| {
        const tag = switch (graph.nodes.items[step.node_id].kind) {
            .op => |value| value,
            else => return error.InvalidGraphStep,
        };
        if (!isFusionPreparatoryTag(tag)) break;
        if (fusion.shouldSkipPreparatoryStep(graph, region, steps, step)) continue;
        if (stepOutputsMaterialized(graph, step, values)) continue;
        if (!stepInputsBound(graph, step, values)) break;
        try executeStep(allocator, graph, parent, step, tag, values, owned);
    }
}

fn executeStep(
    allocator: std.mem.Allocator,
    graph: *const Graph,
    parent: telemetry.Scope,
    step: @import("../../types/ir/plan.zig").Step,
    tag: @import("../../types/operation/tag.zig").OpTag,
    values: []?*Tensor,
    owned: []bool,
) !void {
    const node = graph.nodes.items[step.node_id];
    const op_inputs = try allocator.alloc(*Tensor, node.inputs.len);
    defer allocator.free(op_inputs);
    for (node.inputs, op_inputs) |value_id, *input| {
        input.* = values[value_id] orelse return error.MissingGraphValue;
    }
    const op = try Op.initWithExecutionMetadata(tag, op_inputs, node.options, node.execution_metadata);
    var step_scope = parent.child(@tagName(tag), .internal, &.{});
    defer step_scope.end();
    var execution = try eager.executeAllWithPlan(allocator, op, &step.eager_plan);
    errdefer execution.deinit();
    if (node.outputs.len == 0 or node.outputs.len > 2) return error.InvalidOutputCount;
    if (node.outputs.len == 1 and execution.secondary != null) return error.MultiOutputRequiresExecuteAll;
    if (node.outputs.len == 2 and execution.secondary == null) return error.MissingGraphValue;
    try autograd.recordOperation(allocator, execution.primary, op);
    values[node.outputs[0]] = execution.primary;
    owned[node.outputs[0]] = true;
    execution.primary = undefined;
    if (node.outputs.len == 2) {
        values[node.outputs[1]] = execution.secondary.?;
        owned[node.outputs[1]] = true;
        execution.secondary = null;
    }
    execution = undefined;
}

fn consumeInputs(
    graph: *const Graph,
    steps: []const @import("../../types/ir/plan.zig").Step,
    values: []?*Tensor,
    owned: []bool,
    remaining_uses: []usize,
    is_graph_output: []const bool,
) void {
    for (steps) |step| {
        const node = graph.nodes.items[step.node_id];
        for (node.inputs) |value_id| {
            if (remaining_uses[value_id] > 0) remaining_uses[value_id] -= 1;
            if (remaining_uses[value_id] != 0 or is_graph_output[value_id] or !owned[value_id]) continue;
            if (values[value_id]) |value| autograd.releaseOwnedTensor(value);
            values[value_id] = null;
            owned[value_id] = false;
        }
    }
}
