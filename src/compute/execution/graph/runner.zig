const std = @import("std");
const Value = @import("../../types/tensor/value.zig").Value;
const Graph = @import("../../types/ir/sir.zig").Graph;
const ValueId = @import("../../types/ir/sir.zig").ValueId;
const Op = @import("../../types/operation/op.zig").Op;
const eager = @import("../eager/index.zig");
const plan_graph = @import("../../plan/graph.zig");
const graph_lower = @import("lower.zig");
const fusion = @import("fusion/index.zig");

pub const GraphExecutionResult = struct {
    allocator: std.mem.Allocator,
    values: []?*Value,
    owned: []bool,
    outputs: []*Value,

    pub fn deinit(self: *GraphExecutionResult) void {
        for (self.values, self.owned) |value, is_owned| {
            if (is_owned) if (value) |owned_value| owned_value.deinit();
        }
        self.allocator.free(self.outputs);
        self.allocator.free(self.owned);
        self.allocator.free(self.values);
        self.* = undefined;
    }
};

pub fn execute(
    allocator: std.mem.Allocator,
    graph: *const Graph,
    inputs: []const *Value,
) !GraphExecutionResult {
    if (inputs.len != graph.inputs.items.len) return error.InputCountMismatch;

    var graph_plan = try plan_graph.lower(allocator, graph);
    defer graph_plan.deinit();
    var program = try graph_lower.lower(allocator, &graph_plan);
    defer program.deinit();

    const values = try allocator.alloc(?*Value, graph.values.items.len);
    errdefer allocator.free(values);
    @memset(values, null);

    const owned = try allocator.alloc(bool, graph.values.items.len);
    errdefer allocator.free(owned);
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

    var result = GraphExecutionResult{
        .allocator = allocator,
        .values = values,
        .owned = owned,
        .outputs = try allocator.alloc(*Value, 0),
    };
    errdefer result.deinit();

    var step_index: usize = 0;
    var region_index: usize = 0;
    while (step_index < program.steps.items.len) {
        if (region_index < program.regions.items.len and program.regions.items[region_index].step_start == step_index) {
            const region = program.regions.items[region_index];
            const region_steps = program.steps.items[region.step_start..region.step_end];
            region_index += 1;
            if (try fusion.execute(allocator, graph, region, region_steps, values, owned)) {
                consumeInputs(graph, region_steps, values, owned, remaining_uses, is_graph_output);
                step_index = region.step_end;
                continue;
            }
        }

        const step = program.steps.items[step_index];
        const node = graph.nodes.items[step.node_id];
        const tag = switch (node.kind) {
            .op => |op_tag| op_tag,
            else => return error.InvalidGraphStep,
        };

        const op_inputs = try allocator.alloc(*Value, node.inputs.len);
        defer allocator.free(op_inputs);
        for (node.inputs, op_inputs) |value_id, *input| {
            input.* = values[value_id] orelse return error.MissingGraphValue;
        }

        const op = try Op.initWithExecutionMetadata(
            tag,
            op_inputs,
            node.options,
            node.execution_metadata,
        );
        var execution = try eager.executeAll(allocator, op);
        errdefer execution.deinit();

        if (node.outputs.len == 0 or node.outputs.len > 2) return error.InvalidOutputCount;
        if (node.outputs.len == 1 and execution.secondary != null) {
            return error.MultiOutputRequiresExecuteAll;
        }
        if (node.outputs.len == 2 and execution.secondary == null) return error.MissingGraphValue;

        const primary = execution.primary;
        const secondary = execution.secondary;
        execution.primary = undefined;
        execution.secondary = null;
        execution = undefined;

        values[node.outputs[0]] = primary;
        owned[node.outputs[0]] = true;
        if (node.outputs.len == 2) {
            values[node.outputs[1]] = secondary.?;
            owned[node.outputs[1]] = true;
        }
        consumeInputs(graph, &.{step}, values, owned, remaining_uses, is_graph_output);
        step_index += 1;
    }

    result.outputs = try allocator.alloc(*Value, program.outputs.items.len);
    for (program.outputs.items, result.outputs) |value_id, *output| {
        output.* = values[value_id] orelse return error.MissingGraphValue;
    }
    return result;
}

fn consumeInputs(
    graph: *const Graph,
    steps: []const @import("../../types/ir/eir/graph.zig").Step,
    values: []?*Value,
    owned: []bool,
    remaining_uses: []usize,
    is_graph_output: []const bool,
) void {
    for (steps) |step| {
        const node = graph.nodes.items[step.node_id];
        for (node.inputs) |value_id| {
            if (remaining_uses[value_id] > 0) remaining_uses[value_id] -= 1;
            if (remaining_uses[value_id] != 0 or is_graph_output[value_id] or !owned[value_id]) continue;
            if (values[value_id]) |value| value.deinit();
            values[value_id] = null;
            owned[value_id] = false;
        }
    }
}
