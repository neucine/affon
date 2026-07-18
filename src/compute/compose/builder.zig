const std = @import("std");
const Graph = @import("../types/ir/index.zig").Graph;
const TensorId = @import("../types/ir/index.zig").TensorId;
const Tensor = @import("../types/tensor/tensor.zig").Tensor;
const TensorSpec = @import("../types/tensor/tensor_spec.zig").TensorSpec;
const Shape = @import("../types/tensor/shape.zig").Shape;
const Layout = @import("../types/tensor/layout.zig").Layout;
const Op = @import("../types/operation/op.zig").Op;
const semantic = @import("../plan/sema/index.zig");

pub fn addOpFromOp(
    allocator: std.mem.Allocator,
    graph: *Graph,
    op: Op,
    input_ids: []const TensorId,
) ![]TensorId {
    if (op.inputs.len != input_ids.len) return error.InputCountMismatch;
    var info = try semantic.infer(allocator, op);
    defer info.deinit();

    const primary = TensorSpec{
        .shape = info.shape,
        .dtype = info.dtype,
        .layout = info.layout,
        .device = info.device,
    };

    if (info.secondary_output) |secondary| {
        const secondary_spec = TensorSpec{
            .shape = secondary.shape,
            .dtype = secondary.dtype,
            .layout = secondary.layout,
            .device = info.device,
        };
        return graph.addOpMultiWithExecutionMetadata(op.tag, input_ids, op.options, op.execution_metadata, &.{ primary, secondary_spec });
    }

    const out = try graph.addOpWithExecutionMetadata(op.tag, input_ids, op.options, op.execution_metadata, primary);
    const ids = try allocator.alloc(TensorId, 1);
    ids[0] = out;
    return ids;
}

pub fn addInputFromTensor(allocator: std.mem.Allocator, graph: *Graph, tensor: *const Tensor) !TensorId {
    const spec = try tensor.spec();
    const cloned = TensorSpec{
        .shape = try Shape.initCopy(allocator, spec.shape.dims),
        .dtype = spec.dtype,
        .layout = try Layout.initCopy(allocator, spec.layout.strides, spec.layout.offset),
        .device = spec.device,
    };
    defer {
        var s = cloned.shape;
        s.deinit();
        var l = cloned.layout;
        l.deinit();
    }
    return graph.addInput(cloned);
}

test "graph builder adds topk with inferred dual outputs" {
    const allocator = std.testing.allocator;
    var graph = Graph.init(allocator);
    defer graph.deinit();

    const input = try Tensor.fromSliceF32(allocator, &.{ 2, 3 }, &.{ 1, 9, 3, 7, 2, 8 });
    defer input.deinit();

    const input_id = try addInputFromTensor(allocator, &graph, input);
    const op = try Op.init(.topk, &.{input}, .{ .topk = .{ .k = 2, .axis = 1 } });
    const ids = try addOpFromOp(allocator, &graph, op, &.{input_id});
    defer allocator.free(ids);

    try std.testing.expectEqual(@as(usize, 2), ids.len);
    try std.testing.expectEqual(@as(usize, 2), graph.nodes.items[1].outputs.len);
    try std.testing.expectEqual(@as(usize, 3), graph.values.items.len);
}
