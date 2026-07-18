const std = @import("std");
const Value = @import("../../../types/tensor/value.zig").Value;
const Shape = @import("../../../types/tensor/shape.zig").Shape;
const Device = @import("../../../types/tensor/device.zig").Device;
const Graph = @import("../../../types/ir/sir.zig").Graph;
const Region = @import("../../../types/ir/pir/graph.zig").Region;
const Step = @import("../../../types/ir/eir/graph.zig").Step;
const OpTag = @import("../../../types/operation/tag.zig").OpTag;
const backend_dispatch = @import("../../../backend/dispatch.zig");
const logsumexp_loss = @import("logsumexp_loss.zig");
const gather_logsumexp_loss = @import("gather_logsumexp_loss.zig");
const causal_shift_gather_logsumexp_loss = @import("causal_shift_gather_logsumexp_loss.zig");
const lm_head_cross_entropy_indexed = @import("lm_head_cross_entropy_indexed.zig");

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
    if (try logsumexp_loss.tryExecute(allocator, graph, steps, values, owned)) return true;
    if (try gather_logsumexp_loss.tryExecute(allocator, graph, steps, values, owned)) return true;
    if (try causal_shift_gather_logsumexp_loss.tryExecute(allocator, graph, steps, values, owned)) return true;
    if (try lm_head_cross_entropy_indexed.tryExecute(allocator, graph, steps, values, owned)) return true;
    if (try executeAttentionScores(allocator, graph, steps, values, owned)) return true;
    if (try executeAddLayerNorm(allocator, graph, steps, values, owned)) return true;

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

fn executeAttentionScores(
    allocator: std.mem.Allocator,
    graph: *const Graph,
    steps: []const Step,
    values: []?*Value,
    owned: []bool,
) !bool {
    if (steps.len != 4) return false;
    const matmul_node = graph.nodes.items[steps[0].node_id];
    const scale_node = graph.nodes.items[steps[1].node_id];
    const mask_node = graph.nodes.items[steps[2].node_id];
    const softmax_node = graph.nodes.items[steps[3].node_id];
    const matmul_tag = switch (matmul_node.kind) { .op => |tag| tag, else => return false };
    const scale_tag = switch (scale_node.kind) { .op => |tag| tag, else => return false };
    const mask_tag = switch (mask_node.kind) { .op => |tag| tag, else => return false };
    const softmax_tag = switch (softmax_node.kind) { .op => |tag| tag, else => return false };
    if (matmul_tag != .matmul or scale_tag != .mul or mask_tag != .masked_fill or softmax_tag != .softmax) return false;
    if (matmul_node.inputs.len != 2 or matmul_node.outputs.len != 1 or
        scale_node.inputs.len != 2 or scale_node.outputs.len != 1 or
        mask_node.inputs.len != 2 or mask_node.outputs.len != 1 or
        softmax_node.inputs.len != 1 or softmax_node.outputs.len != 1) return false;
    if (scale_node.inputs[0] != matmul_node.outputs[0] or mask_node.inputs[0] != scale_node.outputs[0] or
        softmax_node.inputs[0] != mask_node.outputs[0]) return false;

    const q = values[matmul_node.inputs[0]] orelse return error.MissingGraphValue;
    const k_t = values[matmul_node.inputs[1]] orelse return error.MissingGraphValue;
    const scale = values[scale_node.inputs[1]] orelse return error.MissingGraphValue;
    const mask = values[mask_node.inputs[1]] orelse return error.MissingGraphValue;
    const device = q.device() orelse return error.InputNotMaterialized;
    if ((k_t.device() orelse return error.InputNotMaterialized) != device or
        (scale.device() orelse return error.InputNotMaterialized) != device or
        (mask.device() orelse return error.InputNotMaterialized) != device) return false;
    if (q.dtype != k_t.dtype or q.dtype != scale.dtype) return false;
    if (q.storage == null or k_t.storage == null or scale.storage == null or mask.storage == null) return false;
    if (q.layout.offset != 0 or k_t.layout.offset != 0 or scale.layout.offset != 0 or mask.layout.offset != 0 or
        !q.layout.isContiguous(q.shape) or !k_t.layout.isContiguous(k_t.shape) or
        !scale.layout.isContiguous(scale.shape) or !mask.layout.isContiguous(mask.shape)) return false;

    const output_spec = graph.values.items[softmax_node.outputs[0]].spec;
    const matmul_spec = graph.values.items[matmul_node.outputs[0]].spec;
    if (output_spec.dtype != q.dtype or output_spec.device != device or
        !Shape.eql(matmul_spec.shape, output_spec.shape) or !Shape.eql(scale.shape, matmul_spec.shape) or
        !Shape.eql(mask.shape, matmul_spec.shape)) return false;
    const softmax_options = switch (softmax_node.options) {
        .softmax => |options| options,
        else => return false,
    };
    const mask_options = switch (mask_node.options) {
        .masked_fill => |options| options,
        else => return false,
    };
    if (softmax_options.axis + 1 != output_spec.shape.rank()) return false;

    const output = try Value.createContiguousWithSource(allocator, output_spec.shape.dims, output_spec.dtype, device, false, .graph);
    errdefer output.deinit();
    try backend_dispatch.attentionScores(allocator, device, q.dtype, q.storage.?, k_t.storage.?, scale.storage.?, mask.storage.?, output.storage.?, q.shape.dims, k_t.shape.dims, output_spec.shape.dims, softmax_options.axis, mask_options.value);
    values[softmax_node.outputs[0]] = output;
    owned[softmax_node.outputs[0]] = true;
    return true;
}

fn executeAddLayerNorm(
    allocator: std.mem.Allocator,
    graph: *const Graph,
    steps: []const Step,
    values: []?*Value,
    owned: []bool,
) !bool {
    if (steps.len != 2) return false;
    const add_node = graph.nodes.items[steps[0].node_id];
    const norm_node = graph.nodes.items[steps[1].node_id];
    const add_tag = switch (add_node.kind) {
        .op => |tag| tag,
        else => return false,
    };
    const norm_tag = switch (norm_node.kind) {
        .op => |tag| tag,
        else => return false,
    };
    if (add_tag != .add or norm_tag != .layer_norm) return false;
    if (add_node.inputs.len != 2 or add_node.outputs.len != 1) return false;
    if (norm_node.inputs.len != 1 or norm_node.outputs.len != 1) return false;
    if (norm_node.inputs[0] != add_node.outputs[0]) return false;

    const lhs = values[add_node.inputs[0]] orelse return error.MissingGraphValue;
    const rhs = values[add_node.inputs[1]] orelse return error.MissingGraphValue;
    const device = lhs.device() orelse return error.InputNotMaterialized;
    if ((rhs.device() orelse return error.InputNotMaterialized) != device or rhs.dtype != lhs.dtype) return false;
    if (!Shape.eql(lhs.shape, rhs.shape)) return false;
    if (lhs.storage == null or rhs.storage == null or lhs.layout.offset != 0 or rhs.layout.offset != 0 or
        !lhs.layout.isContiguous(lhs.shape) or !rhs.layout.isContiguous(rhs.shape)) return false;

    const options = switch (norm_node.options) {
        .layer_norm => |value| value,
        else => return false,
    };
    const output_spec = graph.values.items[norm_node.outputs[0]].spec;
    if (output_spec.device != device or output_spec.dtype != lhs.dtype or !Shape.eql(output_spec.shape, lhs.shape)) return false;
    const output = try Value.createContiguousWithSource(allocator, output_spec.shape.dims, output_spec.dtype, device, false, .graph);
    errdefer output.deinit();
    try backend_dispatch.addLayerNorm(device, lhs.dtype, lhs.storage.?, rhs.storage.?, output.storage.?, output_spec.shape.dims, options.axis, options.eps);
    values[norm_node.outputs[0]] = output;
    owned[norm_node.outputs[0]] = true;
    return true;
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
