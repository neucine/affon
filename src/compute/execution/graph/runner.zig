const std = @import("std");
const Value = @import("../../types/tensor/value.zig").Value;
const Graph = @import("../../types/ir/sir.zig").Graph;
const ValueId = @import("../../types/ir/sir.zig").ValueId;
const Op = @import("../../types/operation/op.zig").Op;
const Shape = @import("../../types/tensor/shape.zig").Shape;
const Device = @import("../../types/tensor/device.zig").Device;
const Region = @import("../../types/ir/pir/graph.zig").Region;
const Step = @import("../../types/ir/eir/graph.zig").Step;
const backend_dispatch = @import("../../backend/dispatch.zig");
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

    var step_index: usize = 0;
    var region_index: usize = 0;
    while (step_index < program.steps.items.len) {
        if (region_index < program.regions.items.len and program.regions.items[region_index].step_start == step_index) {
            const region = program.regions.items[region_index];
            const region_steps = program.steps.items[region.step_start..region.step_end];
            region_index += 1;
            if (try executeFusableRegion(allocator, graph, region, region_steps, values, owned)) {
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
        step_index += 1;
    }

    result.outputs = try allocator.alloc(*Value, program.outputs.items.len);
    for (program.outputs.items, result.outputs) |value_id, *output| {
        output.* = values[value_id] orelse return error.MissingGraphValue;
    }
    return result;
}

fn executeFusableRegion(
    allocator: std.mem.Allocator,
    graph: *const Graph,
    region: Region,
    steps: []const Step,
    values: []?*Value,
    owned: []bool,
) !bool {
    if (region.kind == .matmul_epilogue) {
        return if (region.matmul_epilogue_activation == .none)
            executeMatmulAddRegion(allocator, graph, steps, values, owned)
        else
            false;
    }
    if (region.kind != .fusable_run or steps.len < 2) return false;

    const first_node = graph.nodes.items[steps[0].node_id];
    const binary_tag = switch (first_node.kind) {
        .op => |tag| tag,
        else => return false,
    };
    const is_binary = switch (binary_tag) {
        .add, .sub, .mul, .div => first_node.inputs.len == 2,
        else => false,
    };
    const is_unary = switch (binary_tag) {
        .abs, .neg, .relu, .sign, .clamp, .exp, .log, .sqrt => first_node.inputs.len == 1,
        else => false,
    };
    if (!is_binary and !is_unary) return false;

    const input_id = first_node.inputs[0];
    const lhs = values[input_id] orelse return error.MissingGraphValue;
    const device = lhs.device() orelse return error.InputNotMaterialized;
    if (lhs.storage == null or lhs.layout.offset != 0 or !lhs.layout.isContiguous(lhs.shape)) return false;

    var stages = try allocator.alloc(backend_dispatch.UnaryStage, if (is_binary) steps.len - 1 else steps.len);
    defer allocator.free(stages);

    if (is_binary) {
        const rhs = values[first_node.inputs[1]] orelse return error.MissingGraphValue;
        if ((rhs.device() orelse return error.InputNotMaterialized) != device or rhs.dtype != lhs.dtype) return false;
        if (rhs.storage == null or rhs.layout.offset != 0 or !rhs.layout.isContiguous(rhs.shape)) return false;
        if (!Shape.eql(rhs.shape, lhs.shape)) return false;
        for (steps[1..], 0..) |step, i| {
            const node = graph.nodes.items[step.node_id];
            if (node.inputs.len != 1 or node.inputs[0] != graph.nodes.items[steps[i].node_id].outputs[0]) return false;
            stages[i] = try unaryStage(node);
        }
        return try dispatchRegion(allocator, graph, steps, lhs, rhs, device, binary_tag, stages, values, owned);
    }

    for (steps, 0..) |step, i| {
        const node = graph.nodes.items[step.node_id];
        if (node.inputs.len != 1) return false;
        if (i > 0 and node.inputs[0] != graph.nodes.items[steps[i - 1].node_id].outputs[0]) return false;
        stages[i] = try unaryStage(node);
    }
    return try dispatchRegion(allocator, graph, steps, lhs, null, device, undefined, stages, values, owned);
}

fn executeMatmulAddRegion(
    allocator: std.mem.Allocator,
    graph: *const Graph,
    steps: []const Step,
    values: []?*Value,
    owned: []bool,
) !bool {
    if (steps.len != 2) return false;
    const matmul_node = graph.nodes.items[steps[0].node_id];
    const add_node = graph.nodes.items[steps[1].node_id];
    if (matmul_node.inputs.len != 2 or matmul_node.outputs.len != 1) return false;
    if (add_node.inputs.len != 2 or add_node.outputs.len != 1) return false;

    const matmul_output_id = matmul_node.outputs[0];
    const bias_id = if (add_node.inputs[0] == matmul_output_id)
        add_node.inputs[1]
    else if (add_node.inputs[1] == matmul_output_id)
        add_node.inputs[0]
    else
        return false;
    const lhs = values[matmul_node.inputs[0]] orelse return error.MissingGraphValue;
    const rhs = values[matmul_node.inputs[1]] orelse return error.MissingGraphValue;
    const bias = values[bias_id] orelse return error.MissingGraphValue;
    const device = lhs.device() orelse return error.InputNotMaterialized;
    if ((rhs.device() orelse return error.InputNotMaterialized) != device or
        (bias.device() orelse return error.InputNotMaterialized) != device) return false;
    if (lhs.dtype != rhs.dtype or lhs.dtype != bias.dtype) return false;
    if (!lhs.layout.isContiguous(lhs.shape) or lhs.layout.offset != 0 or
        !rhs.layout.isContiguous(rhs.shape) or rhs.layout.offset != 0 or
        !bias.layout.isContiguous(bias.shape) or bias.layout.offset != 0) return false;

    const output_spec = graph.values.items[add_node.outputs[0]].spec;
    const matmul_spec = graph.values.items[matmul_output_id].spec;
    if (!Shape.eql(output_spec.shape, matmul_spec.shape) or !Shape.eql(output_spec.shape, bias.shape)) return false;
    const output = try Value.createContiguousWithSource(allocator, output_spec.shape.dims, output_spec.dtype, device, false, .graph);
    errdefer output.deinit();
    try backend_dispatch.matmulAdd(
        allocator,
        device,
        lhs.dtype,
        lhs.storage orelse return error.InputNotMaterialized,
        rhs.storage orelse return error.InputNotMaterialized,
        bias.storage orelse return error.InputNotMaterialized,
        output.storage.?,
        lhs.shape.dims,
        rhs.shape.dims,
        output_spec.shape.dims,
    );
    values[add_node.outputs[0]] = output;
    owned[add_node.outputs[0]] = true;
    return true;
}

fn unaryStage(node: anytype) !backend_dispatch.UnaryStage {
    const tag = switch (node.kind) {
        .op => |value| value,
        else => return error.InvalidGraphStep,
    };
    return switch (tag) {
        .abs, .neg, .relu, .sign, .exp, .log, .sqrt => .{ .tag = tag },
        .clamp => switch (node.options) {
            .clamp => |options| .{ .tag = tag, .clamp_min = options.min, .clamp_max = options.max },
            else => error.InvalidGraphPlan,
        },
        else => error.InvalidGraphPlan,
    };
}

fn dispatchRegion(
    allocator: std.mem.Allocator,
    graph: *const Graph,
    steps: []const Step,
    lhs: *Value,
    rhs: ?*Value,
    device: Device,
    binary_tag: @import("../../types/operation/tag.zig").OpTag,
    stages: []const backend_dispatch.UnaryStage,
    values: []?*Value,
    owned: []bool,
) !bool {
    const last_node = graph.nodes.items[steps[steps.len - 1].node_id];
    if (last_node.outputs.len != 1) return false;
    const spec = graph.values.items[last_node.outputs[0]].spec;
    if (spec.dtype != lhs.dtype or spec.device != device or !Shape.eql(spec.shape, lhs.shape)) return false;
    const output = try Value.createContiguousWithSource(allocator, spec.shape.dims, spec.dtype, spec.device, false, .graph);
    errdefer output.deinit();
    if (rhs) |right| {
        try backend_dispatch.binaryThenUnaryChain(allocator, device, lhs.dtype, binary_tag, lhs.storage.?, right.storage.?, output.storage.?, spec.shape.dims, stages);
    } else {
        try backend_dispatch.unaryChain(allocator, device, lhs.dtype, lhs.storage.?, output.storage.?, spec.shape.dims, stages);
    }
    values[last_node.outputs[0]] = output;
    owned[last_node.outputs[0]] = true;
    return true;
}
