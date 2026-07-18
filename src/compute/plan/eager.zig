const std = @import("std");
const tensor = @import("../types/tensor/index.zig");
const Layout = tensor.Layout;
const Shape = tensor.Shape;
const Op = @import("../types/operation/op.zig").Op;
const semantic = @import("sema/index.zig");
const ir_plan = @import("../types/ir/plan.zig");
const execution_layout = @import("layout.zig");
const execution_spec = @import("spec.zig");
const matmul_planning = @import("matmul.zig");
const ExecutionMetadata = @import("../types/operation/execution_metadata.zig").ExecutionMetadata;

pub const Plan = ir_plan.EagerPlan;

pub fn create(allocator: std.mem.Allocator, op: Op, info: semantic.OpSpec) !Plan {
    const input_specs = try allocator.alloc(tensor.TensorSpec, op.inputs.len);
    defer allocator.free(input_specs);
    for (op.inputs, 0..) |input, i| input_specs[i] = try input.spec();
    return createFromSpecs(allocator, op.tag, input_specs, info, op.execution_metadata);
}

pub fn createFromSpecs(
    allocator: std.mem.Allocator,
    op_tag: @import("../types/operation/tag.zig").OpTag,
    input_specs: []const tensor.TensorSpec,
    info: semantic.OpSpec,
    metadata: ExecutionMetadata,
) !Plan {
    const inputs = try allocator.alloc(Plan.Input, input_specs.len);
    errdefer allocator.free(inputs);
    for (input_specs, 0..) |input_spec, i| {
        var input_shape = try Shape.initCopy(allocator, input_spec.shape.dims);
        errdefer input_shape.deinit();
        inputs[i] = .{
            .dtype = input_spec.dtype,
            .device = input_spec.device,
            .shape = input_shape,
        };
    }

    var primary_shape = try Shape.initCopy(allocator, info.shape.dims);
    errdefer primary_shape.deinit();
    var primary_layout = try Layout.initCopy(allocator, info.layout.strides, info.layout.offset);
    errdefer primary_layout.deinit();
    const primary_axes = try tensor.cloneAxes(allocator, info.axes);
    errdefer tensor.deinitAxes(allocator, primary_axes);

    var secondary_output: ?Plan.Output = null;
    if (info.secondary_output) |secondary| {
        var secondary_shape = try Shape.initCopy(allocator, secondary.shape.dims);
        errdefer secondary_shape.deinit();
        var secondary_layout = try Layout.initCopy(allocator, secondary.layout.strides, secondary.layout.offset);
        errdefer secondary_layout.deinit();
        const secondary_axes = try tensor.cloneAxes(allocator, secondary.axes);
        errdefer tensor.deinitAxes(allocator, secondary_axes);
        secondary_output = .{
            .dtype = secondary.dtype,
            .shape = secondary_shape,
            .layout = secondary_layout,
            .axes = secondary_axes,
            .bytes = secondary.shape.numel() * secondary.dtype.size(),
        };
    }

    const input_layout_decision = execution_layout.inputDecisionForHint(info.planner_hint);
    const matmul = try createMatmulDecision(op_tag, input_specs, info, metadata, input_layout_decision);

    return .{
        .op_tag = op_tag,
        .allocator = allocator,
        .device = info.device,
        .kind = execution_spec.executionKindFromSemantic(info.kind),
        .broadcast = info.broadcast,
        .reduce_to_shape = info.reduce_to_shape,
        .input_requirement = execution_spec.inputRequirementFromSemantic(info.input_requirement),
        .input_layout_decision = input_layout_decision,
        .inputs = inputs,
        .matmul = matmul,
        .primary_output = .{
            .dtype = info.dtype,
            .shape = primary_shape,
            .layout = primary_layout,
            .axes = primary_axes,
            .bytes = if (info.allocation == .view_only) 0 else info.shape.numel() * info.dtype.size(),
        },
        .secondary_output = secondary_output,
    };
}

fn createMatmulDecision(
    op_tag: @import("../types/operation/tag.zig").OpTag,
    input_specs: []const tensor.TensorSpec,
    info: semantic.OpSpec,
    metadata: ExecutionMetadata,
    input_layout_decision: execution_layout.InputLayoutDecision,
) !?ir_plan.MatmulDecision {
    if (op_tag != .matmul or input_specs.len != 2) return null;
    const output = tensor.TensorSpec{
        .shape = info.shape,
        .dtype = info.dtype,
        .layout = info.layout,
        .device = info.device,
    };
    const classification = matmul_planning.classifyFromSpecs(input_specs[0], input_specs[1], output, .{
        .hint = metadata.matmul_hint,
        .hint_source = metadata.hint_source,
    });
    return .{
        .family = switch (classification.family) {
            .gemm_2d => .gemm_2d,
            .gemm_batched => .gemm_batched,
            .gemm_projection => .gemm_projection,
            .gemm_attention_scores => .gemm_attention_scores,
            .gemm_attention_values => .gemm_attention_values,
            .gemm_generic_unresolved => .gemm_generic_unresolved,
        },
        .projection = classification.family == .gemm_projection and
            classification.flattenable_leading_batch and
            input_specs[1].shape.rank() == 2 and
            (input_specs[0].layout.offset == 0 and input_specs[0].layout.isContiguous(input_specs[0].shape) or
                input_layout_decision == .pack_to_dense),
    };
}
