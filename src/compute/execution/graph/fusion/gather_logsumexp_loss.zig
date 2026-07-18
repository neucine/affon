const std = @import("std");
const Value = @import("../../../types/tensor/value.zig").Value;
const Shape = @import("../../../types/tensor/shape.zig").Shape;
const Layout = @import("../../../types/tensor/layout.zig").Layout;
const Graph = @import("../../../types/ir/index.zig").Graph;
const Step = @import("../../../types/ir/eir/graph.zig").Step;
const kernel_dispatch = @import("../../../backend/dispatch.zig");
const value_helpers = @import("value_helpers.zig");
const common = @import("common.zig");

fn axisFromReduceAxisNode(node: anytype) ?usize {
    return switch (node.options) {
        .reduce_axis => |r| r.axis,
        else => null,
    };
}

fn keepdimFromReduceAxisNode(node: anytype) ?bool {
    return switch (node.options) {
        .reduce_axis => |r| r.keepdim,
        else => null,
    };
}

fn hasSingleConsumer(graph: *const Graph, value_id: u32, expected_node_id: u32) bool {
    for (graph.outputs.items) |gid| if (gid == value_id) return false;
    var consumer_count: usize = 0;
    var expected_is_consumer = false;
    for (graph.nodes.items) |node| {
        for (node.inputs) |in_id| {
            if (in_id == value_id) {
                consumer_count += 1;
                if (node.id == expected_node_id) expected_is_consumer = true;
            }
        }
    }
    return consumer_count == 1 and expected_is_consumer;
}

fn hasExactlyConsumers(graph: *const Graph, value_id: u32, expected_node_ids: []const u32) bool {
    for (graph.outputs.items) |gid| if (gid == value_id) return false;
    var seen = [_]bool{false} ** 4;
    if (expected_node_ids.len > seen.len) return false;
    var consumer_count: usize = 0;
    for (graph.nodes.items) |node| {
        for (node.inputs) |in_id| {
            if (in_id == value_id) {
                consumer_count += 1;
                var matched = false;
                for (expected_node_ids, 0..) |exp_id, i| {
                    if (node.id == exp_id) {
                        seen[i] = true;
                        matched = true;
                        break;
                    }
                }
                if (!matched) return false;
            }
        }
    }
    if (consumer_count != expected_node_ids.len) return false;
    for (expected_node_ids, 0..) |_, i| if (!seen[i]) return false;
    return true;
}

fn makeLogitsExpandedView(allocator: std.mem.Allocator, logits: *Value) !*Value {
    const storage = logits.storage orelse return error.InputNotMaterialized;
    if (logits.shape.rank() != 2) return error.ShapeMismatch;
    const rows = logits.shape.dims[0];
    const vocab = logits.shape.dims[1];

    storage.retain();
    errdefer storage.release();
    const view = try allocator.create(Value);
    var shape3 = try Shape.initCopy(allocator, &.{ rows, 1, vocab });
    errdefer shape3.deinit();
    var layout3 = try Layout.initContiguous(allocator, shape3);
    errdefer layout3.deinit();
    view.* = .{
        .allocator = allocator,
        .shape = shape3,
        .dtype = logits.dtype,
        .layout = layout3,
        .storage = storage,
        .axes = null,
    };
    return view;
}

pub fn tryExecute(
    allocator: std.mem.Allocator,
    graph: *const Graph,
    group: []const Step,
    values: []?*Value,
    owned: []bool,
) !bool {
    if (group.len != 9) return false;
    const n0 = graph.nodes.items[group[0].node_id];
    const n1 = graph.nodes.items[group[1].node_id];
    const n2 = graph.nodes.items[group[2].node_id];
    const n3 = graph.nodes.items[group[3].node_id];
    const n4 = graph.nodes.items[group[4].node_id];
    const n5 = graph.nodes.items[group[5].node_id];
    const n6 = graph.nodes.items[group[6].node_id];
    const n7 = graph.nodes.items[group[7].node_id];
    const n8 = graph.nodes.items[group[8].node_id];

    const t0 = switch (n0.kind) {
        .op => |t| t,
        else => return false,
    };
    const t1 = switch (n1.kind) {
        .op => |t| t,
        else => return false,
    };
    const t2 = switch (n2.kind) {
        .op => |t| t,
        else => return false,
    };
    const t3 = switch (n3.kind) {
        .op => |t| t,
        else => return false,
    };
    const t4 = switch (n4.kind) {
        .op => |t| t,
        else => return false,
    };
    const t5 = switch (n5.kind) {
        .op => |t| t,
        else => return false,
    };
    const t6 = switch (n6.kind) {
        .op => |t| t,
        else => return false,
    };
    const t7 = switch (n7.kind) {
        .op => |t| t,
        else => return false,
    };
    const t8 = switch (n8.kind) {
        .op => |t| t,
        else => return false,
    };
    if (t0 != .gather or t1 != .max_axis or t2 != .sub or t3 != .exp or t4 != .sum_axis or t5 != .log or t6 != .add or t7 != .sub or t8 != .mean_all) return false;

    if (n0.inputs.len != 2 or n0.outputs.len != 1) return false;
    if (n1.inputs.len != 1 or n1.outputs.len != 1) return false;
    if (n2.inputs.len != 2 or n2.outputs.len != 1) return false;
    if (n3.inputs.len != 1 or n3.outputs.len != 1) return false;
    if (n4.inputs.len != 1 or n4.outputs.len != 1) return false;
    if (n5.inputs.len != 1 or n5.outputs.len != 1) return false;
    if (n6.inputs.len != 2 or n6.outputs.len != 1) return false;
    if (n7.inputs.len != 2 or n7.outputs.len != 1) return false;
    if (n8.inputs.len != 1 or n8.outputs.len != 1) return false;

    const logits_id = n0.inputs[0];
    const index_id = n0.inputs[1];
    const selected_id = n0.outputs[0];
    if (n1.inputs[0] != logits_id) return false;
    const max_id = n1.outputs[0];
    if (n2.inputs[0] != logits_id or n2.inputs[1] != max_id) return false;
    const shifted_id = n2.outputs[0];
    if (n3.inputs[0] != shifted_id) return false;
    const exp_id = n3.outputs[0];
    if (n4.inputs[0] != exp_id) return false;
    const sumexp_id = n4.outputs[0];
    if (n5.inputs[0] != sumexp_id) return false;
    const logsumexp_base_id = n5.outputs[0];
    if (n6.inputs[0] != logsumexp_base_id or n6.inputs[1] != max_id) return false;
    const logsumexp_id = n6.outputs[0];
    if (n7.inputs[0] != logsumexp_id or n7.inputs[1] != selected_id) return false;
    const per_token_id = n7.outputs[0];
    if (n8.inputs[0] != per_token_id) return false;
    const out_id = n8.outputs[0];

    const gather_axis = switch (n0.options) {
        .gather => |g| g.axis,
        else => return false,
    };
    if (gather_axis != 1) return false;
    const axis1 = axisFromReduceAxisNode(n1) orelse return false;
    const axis4 = axisFromReduceAxisNode(n4) orelse return false;
    if (axis1 != 1 or axis4 != 1) return false;
    if (!(keepdimFromReduceAxisNode(n1) orelse false)) return false;
    if (!(keepdimFromReduceAxisNode(n4) orelse false)) return false;

    if (!hasSingleConsumer(graph, selected_id, n7.id)) return false;
    if (!hasSingleConsumer(graph, per_token_id, n8.id)) return false;

    const logits = values[logits_id] orelse return error.UnboundGraphValue;
    const indices = values[index_id] orelse return error.UnboundGraphValue;
    const device = logits.device() orelse return error.InputNotMaterialized;
    if ((indices.device() orelse return error.InputNotMaterialized) != device) return false;
    if (logits.dtype != .f32) return false;
    if (indices.dtype != .i64) return false;
    if (logits.shape.rank() != 2) return false;
    if (indices.shape.rank() != 2) return false;
    if (indices.shape.dims[0] != logits.shape.dims[0] or indices.shape.dims[1] != 1) return false;
    if (!common.rawStorageInputsArePackedDense(&.{ logits, indices })) return false;

    const rows = logits.shape.dims[0];
    const vocab = logits.shape.dims[1];

    const logits3 = try makeLogitsExpandedView(allocator, logits);
    defer logits3.deinit();
    const dense_indices = try value_helpers.cloneValue(allocator, indices);
    defer dense_indices.deinit();

    const onehot = try Value.createContiguousWithSource(allocator, &.{ rows, 1, vocab }, .f32, device, false, .graph);
    defer onehot.deinit();
    try kernel_dispatch.oneHot(
        device,
        dense_indices.storage orelse return error.InputNotMaterialized,
        onehot.storage orelse return error.InputNotMaterialized,
        vocab,
    );

    const out_spec = graph.values.items[out_id].spec;
    const out = try Value.createContiguousWithSource(allocator, out_spec.shape.dims, out_spec.dtype, out_spec.device, false, .graph);
    errdefer out.deinit();
    try kernel_dispatch.logSoftmaxNll(
        device,
        .f32,
        logits3.storage orelse return error.InputNotMaterialized,
        onehot.storage orelse return error.InputNotMaterialized,
        out.storage orelse return error.InputNotMaterialized,
        logits3.shape.dims,
        2,
    );

    values[out_id] = out;
    owned[out_id] = true;
    return true;
}
