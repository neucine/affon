const std = @import("std");
const pir = @import("../../types/ir/pir/index.zig");
const eir = @import("../../types/ir/eir/index.zig");
const semantic = @import("../../sema/index.zig");
const execution_layout = @import("../layout.zig");
const execution_spec = @import("../spec.zig");

pub fn lower(allocator: std.mem.Allocator, plan: *const pir.GraphPlan) !eir.GraphProgram {
    var program = eir.GraphProgram.init(allocator);
    errdefer program.deinit();

    for (plan.steps.items) |step| {
        try program.steps.append(allocator, .{
            .node_id = step.node_id,
            .kind = step.kind,
            .semantic_spec = try semantic.OpSpec.clone(allocator, step.semantic_spec),
            .execution = executionDecisionsFromSemantic(step.semantic_spec),
        });
    }
    try program.regions.appendSlice(allocator, plan.regions.items);
    try program.outputs.appendSlice(allocator, plan.outputs.items);
    return program;
}

fn executionDecisionsFromSemantic(spec: semantic.OpSpec) eir.graph.Decisions {
    return .{
        .kind = execution_spec.executionKindFromSemantic(spec.kind),
        .allocation = execution_spec.allocationIntentFromSemantic(spec.allocation),
        .input_requirement = execution_spec.inputRequirementFromSemantic(spec.input_requirement),
        .input_layout_decision = execution_layout.inputDecisionForHint(spec.planner_hint),
        .broadcast = spec.broadcast,
        .reduce_to_shape = spec.reduce_to_shape,
    };
}

test "graph execution lowering derives runtime decisions from PIR semantic facts" {
    const allocator = std.testing.allocator;
    var graph = @import("../../types/ir/sir.zig").Graph.init(allocator);
    defer graph.deinit();

    var shape = try @import("../../types/tensor/shape.zig").Shape.initCopy(allocator, &.{3});
    defer shape.deinit();
    var layout = try @import("../../types/tensor/layout.zig").Layout.initContiguous(allocator, shape);
    defer layout.deinit();
    const spec = @import("../../types/tensor/value_spec.zig").ValueSpec{
        .shape = shape,
        .dtype = .f32,
        .layout = layout,
        .device = .cpu,
    };

    const a = try graph.addInput(spec);
    const b = try graph.addInput(spec);
    const c = try graph.addOp(.add, &.{ a, b }, .{ .none = {} }, spec);
    try graph.setOutputs(&.{c});

    var plan = try @import("../../plan/graph.zig").lower(allocator, &graph);
    defer plan.deinit();

    var program = try lower(allocator, &plan);
    defer program.deinit();

    try std.testing.expectEqual(@as(usize, 1), program.steps.items.len);
    try std.testing.expectEqual(execution_spec.ExecutionKind.elementwise_binary, program.steps.items[0].execution.kind);
    try std.testing.expectEqual(@as(usize, 1), program.outputs.items.len);
}
