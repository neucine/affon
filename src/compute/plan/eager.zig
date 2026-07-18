const std = @import("std");
const tensor = @import("../types/tensor/index.zig");
const Layout = tensor.Layout;
const Shape = tensor.Shape;
const Op = @import("../types/operation/op.zig").Op;
const semantic = @import("../sema/index.zig");
const pir = @import("../types/ir/pir/index.zig");
const execution_layout = @import("../execution/layout.zig");
const execution_spec = @import("../execution/spec.zig");

pub const Plan = pir.EagerPlan;

pub fn lower(allocator: std.mem.Allocator, op: Op, info: semantic.OpSpec) !Plan {
    const inputs = try allocator.alloc(Plan.Input, op.inputs.len);
    errdefer allocator.free(inputs);
    for (op.inputs, 0..) |input, i| {
        const input_spec = try input.spec();
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

    return .{
        .op_tag = op.tag,
        .allocator = allocator,
        .device = info.device,
        .kind = execution_spec.executionKindFromSemantic(info.kind),
        .broadcast = info.broadcast,
        .reduce_to_shape = info.reduce_to_shape,
        .input_requirement = execution_spec.inputRequirementFromSemantic(info.input_requirement),
        .input_layout_decision = execution_layout.inputDecisionForHint(info.planner_hint),
        .inputs = inputs,
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
