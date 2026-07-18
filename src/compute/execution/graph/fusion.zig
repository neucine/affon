const std = @import("std");
const Value = @import("../../types/tensor/value.zig").Value;
const Shape = @import("../../types/tensor/shape.zig").Shape;
const Device = @import("../../types/tensor/device.zig").Device;
const Graph = @import("../../types/ir/sir.zig").Graph;
const Region = @import("../../types/ir/pir/graph.zig").Region;
const Step = @import("../../types/ir/eir/graph.zig").Step;
const OpTag = @import("../../types/operation/tag.zig").OpTag;
const backend_dispatch = @import("../../backend/dispatch.zig");

pub fn execute(
    allocator: std.mem.Allocator,
    graph: *const Graph,
    region: Region,
    steps: []const Step,
    values: []?*Value,
    owned: []bool,
) !bool {
    if (region.kind == .matmul_epilogue) {
        return if (region.matmul_epilogue_activation == .none or region.matmul_epilogue_activation == .gelu)
            executeMatmulAddRegion(allocator, graph, region, steps, values, owned)
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

    const lhs = values[first_node.inputs[0]] orelse return error.MissingGraphValue;
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
        return dispatchRegion(allocator, graph, steps, lhs, rhs, device, binary_tag, stages, values, owned);
    }

    for (steps, 0..) |step, i| {
        const node = graph.nodes.items[step.node_id];
        if (node.inputs.len != 1) return false;
        if (i > 0 and node.inputs[0] != graph.nodes.items[steps[i - 1].node_id].outputs[0]) return false;
        stages[i] = try unaryStage(node);
    }
    return dispatchRegion(allocator, graph, steps, lhs, null, device, undefined, stages, values, owned);
}

fn executeMatmulAddRegion(
    allocator: std.mem.Allocator,
    graph: *const Graph,
    region: Region,
    steps: []const Step,
    values: []?*Value,
    owned: []bool,
) !bool {
    const has_gelu = region.matmul_epilogue_activation == .gelu;
    const expected_len: usize = if (has_gelu) 3 else 2;
    if (steps.len != expected_len) return false;
    const matmul_node = graph.nodes.items[steps[0].node_id];
    const add_node = graph.nodes.items[steps[1].node_id];
    if (matmul_node.inputs.len != 2 or matmul_node.outputs.len != 1) return false;
    if (add_node.inputs.len != 2 or add_node.outputs.len != 1) return false;
    if (has_gelu) {
        const activation_node = graph.nodes.items[steps[2].node_id];
        if (activation_node.inputs.len != 1 or activation_node.outputs.len != 1) return false;
        if (activation_node.inputs[0] != add_node.outputs[0]) return false;
        switch (activation_node.kind) {
            .op => |tag| if (tag != .gelu) return false,
            else => return false,
        }
    }

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

    const output_id = if (has_gelu) graph.nodes.items[steps[2].node_id].outputs[0] else add_node.outputs[0];
    const output_spec = graph.values.items[output_id].spec;
    const matmul_spec = graph.values.items[matmul_output_id].spec;
    if (!Shape.eql(output_spec.shape, matmul_spec.shape) or !Shape.eql(output_spec.shape, bias.shape)) return false;
    const output = try Value.createContiguousWithSource(allocator, output_spec.shape.dims, output_spec.dtype, device, false, .graph);
    errdefer output.deinit();
    if (has_gelu) {
        try backend_dispatch.matmulAddGelu(allocator, device, lhs.dtype, lhs.storage orelse return error.InputNotMaterialized, rhs.storage orelse return error.InputNotMaterialized, bias.storage orelse return error.InputNotMaterialized, output.storage.?, lhs.shape.dims, rhs.shape.dims, output_spec.shape.dims);
    } else {
        try backend_dispatch.matmulAdd(allocator, device, lhs.dtype, lhs.storage orelse return error.InputNotMaterialized, rhs.storage orelse return error.InputNotMaterialized, bias.storage orelse return error.InputNotMaterialized, output.storage.?, lhs.shape.dims, rhs.shape.dims, output_spec.shape.dims);
    }
    values[output_id] = output;
    owned[output_id] = true;
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
    binary_tag: OpTag,
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
