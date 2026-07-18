const std = @import("std");
const kernel_dispatch = @import("../../../backend/dispatch.zig");
const Tensor = @import("../../../types/tensor/tensor.zig").Tensor;
const Step = @import("../../../types/ir/plan.zig").Step;
const Graph = @import("../../../types/ir/index.zig").Graph;
const OpTag = @import("../../../types/operation/tag.zig").OpTag;
const OpOptions = @import("../../../types/operation/options.zig").OpOptions;
const prepared_execution = @import("../../prepared.zig");

pub fn clampOptionsForTag(tag: OpTag, options: OpOptions) !?@import("../../../types/operation/options.zig").ClampOptions {
    return switch (tag) {
        .clamp => switch (options) {
            .clamp => |c| c,
            else => return error.InvalidGraphPlan,
        },
        else => null,
    };
}

pub fn materializeBroadcastBiasForAdd(
    allocator: std.mem.Allocator,
    graph: *const Graph,
    add_step: Step,
    matmul_output_id: u32,
    bias_input_index: usize,
    bias: *Tensor,
) !?*Tensor {
    const add_node = graph.nodes.items[add_step.node_id];
    if (add_node.inputs.len != 2 or add_node.outputs.len != 1) return error.InvalidGraphPlan;
    if (bias_input_index >= add_node.inputs.len) return error.InvalidGraphPlan;
    const add_output_id = add_node.outputs[0];
    const matmul_spec = graph.values.items[matmul_output_id].spec;
    const add_spec = graph.values.items[add_output_id].spec;
    if (std.mem.eql(usize, bias.shape.dims, matmul_spec.shape.dims)) return null;

    const broadcast = add_step.execution.broadcast orelse return error.InvalidGraphPlan;
    const binary = switch (broadcast) {
        .binary => |spec| spec,
        else => return error.InvalidGraphPlan,
    };
    const bias_strides = if (bias_input_index == 0)
        binary.lhs_strides[0..binary.rank]
    else
        binary.rhs_strides[0..binary.rank];

    const expanded = try Tensor.createContiguousWithSource(
        allocator,
        add_spec.shape.dims,
        add_spec.dtype,
        add_spec.device,
        false,
        .graph,
    );
    errdefer expanded.deinit();

    try kernel_dispatch.binaryBroadcast(
        add_spec.device,
        .sub,
        add_spec.dtype,
        bias.storage orelse return error.InputNotMaterialized,
        bias.storage orelse return error.InputNotMaterialized,
        expanded.storage.?,
        add_spec.shape.dims,
        bias_strides,
        bias_strides,
        bias.layout.offset,
        bias.layout.offset,
    );

    try kernel_dispatch.binaryBroadcast(
        add_spec.device,
        .add,
        add_spec.dtype,
        expanded.storage.?,
        bias.storage orelse return error.InputNotMaterialized,
        expanded.storage.?,
        add_spec.shape.dims,
        expanded.layout.strides,
        bias_strides,
        expanded.layout.offset,
        bias.layout.offset,
    );

    return expanded;
}

pub fn isInputOrConstantValue(graph: *const Graph, value_id: u32) bool {
    if (value_id >= graph.values.items.len) return false;
    const producer_id = graph.values.items[value_id].producer;
    if (producer_id >= graph.nodes.items.len) return false;
    return switch (graph.nodes.items[producer_id].kind) {
        .input, .constant => true,
        .op => false,
    };
}

pub fn rawStorageInputsArePackedDense(values: []const *Tensor) bool {
    for (values) |value| {
        if (!prepared_execution.isPackedDenseInput(value)) return false;
    }
    return true;
}
