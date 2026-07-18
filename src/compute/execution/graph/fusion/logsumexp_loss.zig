const std = @import("std");
const Value = @import("../../../types/tensor/value.zig").Value;
const Shape = @import("../../../types/tensor/shape.zig").Shape;
const Graph = @import("../../../types/ir/index.zig").Graph;
const Step = @import("../../../types/ir/eir/graph.zig").Step;
const kernel_dispatch = @import("../../../backend/dispatch.zig");
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
    for (expected_node_ids, 0..) |_, i| {
        if (!seen[i]) return false;
    }
    return true;
}

pub fn tryExecute(
    allocator: std.mem.Allocator,
    graph: *const Graph,
    group: []const Step,
    values: []?*Value,
    owned: []bool,
) !bool {
    if (group.len != 10) return false;
    const n0 = graph.nodes.items[group[0].node_id];
    const n1 = graph.nodes.items[group[1].node_id];
    const n2 = graph.nodes.items[group[2].node_id];
    const n3 = graph.nodes.items[group[3].node_id];
    const n4 = graph.nodes.items[group[4].node_id];
    const n5 = graph.nodes.items[group[5].node_id];
    const n6 = graph.nodes.items[group[6].node_id];
    const n7 = graph.nodes.items[group[7].node_id];
    const n8 = graph.nodes.items[group[8].node_id];
    const n9 = graph.nodes.items[group[9].node_id];

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
    const t9 = switch (n9.kind) {
        .op => |t| t,
        else => return false,
    };
    if (t0 != .max_axis or t1 != .sub or t2 != .exp or t3 != .sum_axis or t4 != .log or t5 != .sub or t6 != .mul or t7 != .sum_axis or t8 != .neg or t9 != .mean_all) return false;

    if (n0.inputs.len != 1 or n0.outputs.len != 1) return false;
    if (n1.inputs.len != 2 or n1.outputs.len != 1) return false;
    if (n2.inputs.len != 1 or n2.outputs.len != 1) return false;
    if (n3.inputs.len != 1 or n3.outputs.len != 1) return false;
    if (n4.inputs.len != 1 or n4.outputs.len != 1) return false;
    if (n5.inputs.len != 2 or n5.outputs.len != 1) return false;
    if (n6.inputs.len != 2 or n6.outputs.len != 1) return false;
    if (n7.inputs.len != 1 or n7.outputs.len != 1) return false;
    if (n8.inputs.len != 1 or n8.outputs.len != 1) return false;
    if (n9.inputs.len != 1 or n9.outputs.len != 1) return false;

    const logits_id = n0.inputs[0];
    const max_id = n0.outputs[0];
    if (n1.inputs[0] != logits_id or n1.inputs[1] != max_id) return false;
    const shifted_id = n1.outputs[0];
    if (n2.inputs[0] != shifted_id) return false;
    const exp_id = n2.outputs[0];
    if (n3.inputs[0] != exp_id) return false;
    const sumexp_id = n3.outputs[0];
    if (n4.inputs[0] != sumexp_id) return false;
    const logsumexp_id = n4.outputs[0];
    if (n5.inputs[0] != shifted_id or n5.inputs[1] != logsumexp_id) return false;
    const logsm_id = n5.outputs[0];
    const targets_id = if (n6.inputs[0] == logsm_id) n6.inputs[1] else if (n6.inputs[1] == logsm_id) n6.inputs[0] else return false;
    const nll_elem_id = n6.outputs[0];
    if (n7.inputs[0] != nll_elem_id) return false;
    const per_example_id = n7.outputs[0];
    if (n8.inputs[0] != per_example_id) return false;
    const neg_id = n8.outputs[0];
    if (n9.inputs[0] != neg_id) return false;
    const out_id = n9.outputs[0];

    const axis0 = axisFromReduceAxisNode(n0) orelse return false;
    const axis3 = axisFromReduceAxisNode(n3) orelse return false;
    const axis7 = axisFromReduceAxisNode(n7) orelse return false;
    if (axis0 != axis3 or axis0 != axis7) return false;
    if (!(keepdimFromReduceAxisNode(n0) orelse false)) return false;
    if (!(keepdimFromReduceAxisNode(n3) orelse false)) return false;
    if ((keepdimFromReduceAxisNode(n7) orelse true)) return false;

    if (!hasSingleConsumer(graph, max_id, n1.id)) return false;
    if (!hasExactlyConsumers(graph, shifted_id, &.{ n2.id, n5.id })) return false;
    if (!hasSingleConsumer(graph, exp_id, n3.id)) return false;
    if (!hasSingleConsumer(graph, sumexp_id, n4.id)) return false;
    if (!hasSingleConsumer(graph, logsumexp_id, n5.id)) return false;
    if (!hasSingleConsumer(graph, logsm_id, n6.id)) return false;
    if (!hasSingleConsumer(graph, nll_elem_id, n7.id)) return false;
    if (!hasSingleConsumer(graph, per_example_id, n8.id)) return false;
    if (!hasSingleConsumer(graph, neg_id, n9.id)) return false;

    const logits = values[logits_id] orelse return error.UnboundGraphValue;
    const targets = values[targets_id] orelse return error.UnboundGraphValue;
    const device = logits.device() orelse return error.InputNotMaterialized;
    if ((targets.device() orelse return error.InputNotMaterialized) != device) return false;
    if (logits.dtype != targets.dtype) return false;
    if (!Shape.eql(logits.shape, targets.shape)) return false;
    if (logits.dtype == .i64) return false;
    if (axis0 >= logits.shape.rank()) return false;
    if (!common.rawStorageInputsArePackedDense(&.{ logits, targets })) return false;

    const out_spec = graph.values.items[out_id].spec;
    const out = try Value.createContiguousWithSource(allocator, out_spec.shape.dims, out_spec.dtype, out_spec.device, false, .graph);
    errdefer out.deinit();
    try kernel_dispatch.logSoftmaxNll(
        device,
        logits.dtype,
        logits.storage orelse return error.InputNotMaterialized,
        targets.storage orelse return error.InputNotMaterialized,
        out.storage orelse return error.InputNotMaterialized,
        logits.shape.dims,
        axis0,
    );

    values[out_id] = out;
    owned[out_id] = true;
    return true;
}
