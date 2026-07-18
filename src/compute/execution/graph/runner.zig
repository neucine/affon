const std = @import("std");
const Value = @import("../../types/tensor/value.zig").Value;
const Graph = @import("../../types/ir/sir.zig").Graph;
const ValueId = @import("../../types/ir/sir.zig").ValueId;
const Op = @import("../../types/operation/op.zig").Op;
const eager = @import("../eager/index.zig");
const plan_graph = @import("../../plan/graph.zig");
const graph_lower = @import("lower.zig");

pub const GraphExecutionResult = struct {
    allocator: std.mem.Allocator,
    values: []?*Value,
    owned: []bool,
    outputs: []*Value,

    pub fn deinit(self: *GraphExecutionResult) void {
        for (self.values, self.owned) |value, is_owned| {
            if (is_owned) value.?.deinit();
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

    for (graph.inputs.items, inputs) |value_id, input| values[value_id] = input;

    var result = GraphExecutionResult{
        .allocator = allocator,
        .values = values,
        .owned = owned,
        .outputs = try allocator.alloc(*Value, 0),
    };
    errdefer result.deinit();

    for (program.steps.items) |step| {
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
    }

    result.outputs = try allocator.alloc(*Value, program.outputs.items.len);
    for (program.outputs.items, result.outputs) |value_id, *output| {
        output.* = values[value_id] orelse return error.MissingGraphValue;
    }
    return result;
}
