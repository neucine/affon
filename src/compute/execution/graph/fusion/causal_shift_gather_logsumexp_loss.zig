const std = @import("std");
const Value = @import("../../../types/tensor/value.zig").Value;
const Shape = @import("../../../types/tensor/shape.zig").Shape;
const Layout = @import("../../../types/tensor/layout.zig").Layout;
const Graph = @import("../../../types/ir/index.zig").Graph;
const Step = @import("../../../types/ir/eir/graph.zig").Step;
const kernel_dispatch = @import("../../../backend/dispatch.zig");
const SliceRange = @import("../../../types/operation/options.zig").SliceRange;
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

fn matchesTokenSlice(node: anytype) bool {
    const slice = switch (node.options) {
        .slice => |s| s,
        else => return false,
    };
    if (slice.ranges.len != 2) return false;
    const r0 = slice.ranges[0];
    const r1 = slice.ranges[1];
    // [:, 1:...]
    return r0.start == 0 and r0.step == 1 and r1.start == 1 and r1.step == 1;
}

pub fn tryExecute(
    allocator: std.mem.Allocator,
    graph: *const Graph,
    group: []const Step,
    values: []?*Value,
    owned: []bool,
) !bool {
    if (group.len != 12) return false;
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
    const n10 = graph.nodes.items[group[10].node_id];
    const n11 = graph.nodes.items[group[11].node_id];

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
    const t10 = switch (n10.kind) {
        .op => |t| t,
        else => return false,
    };
    const t11 = switch (n11.kind) {
        .op => |t| t,
        else => return false,
    };
    if (t0 != .slice or t1 != .reshape or t2 != .reshape or t3 != .gather or t4 != .max_axis or t5 != .sub or t6 != .exp or t7 != .sum_axis or t8 != .log or t9 != .add or t10 != .sub or t11 != .mean_all) return false;
    if (!matchesTokenSlice(n0)) return false;

    if (n0.inputs.len != 1 or n0.outputs.len != 1) return false;
    if (n1.inputs.len != 1 or n1.outputs.len != 1) return false;
    if (n2.inputs.len != 1 or n2.outputs.len != 1) return false;
    if (n3.inputs.len != 2 or n3.outputs.len != 1) return false;
    if (n4.inputs.len != 1 or n4.outputs.len != 1) return false;
    if (n5.inputs.len != 2 or n5.outputs.len != 1) return false;
    if (n6.inputs.len != 1 or n6.outputs.len != 1) return false;
    if (n7.inputs.len != 1 or n7.outputs.len != 1) return false;
    if (n8.inputs.len != 1 or n8.outputs.len != 1) return false;
    if (n9.inputs.len != 2 or n9.outputs.len != 1) return false;
    if (n10.inputs.len != 2 or n10.outputs.len != 1) return false;
    if (n11.inputs.len != 1 or n11.outputs.len != 1) return false;

    const token_ids = n0.inputs[0];
    const shifted_ids = n0.outputs[0];
    if (n1.inputs[0] != shifted_ids) return false;
    const target_index = n1.outputs[0];

    const logits3d = n2.inputs[0];
    const logits2d = n2.outputs[0];
    if (n3.inputs[0] != logits2d or n3.inputs[1] != target_index) return false;
    const selected = n3.outputs[0];

    if (n4.inputs[0] != logits2d) return false;
    const maxv = n4.outputs[0];
    if (n5.inputs[0] != logits2d or n5.inputs[1] != maxv) return false;
    const shifted = n5.outputs[0];
    if (n6.inputs[0] != shifted) return false;
    const expv = n6.outputs[0];
    if (n7.inputs[0] != expv) return false;
    const sumexp = n7.outputs[0];
    if (n8.inputs[0] != sumexp) return false;
    const logsumexp_base = n8.outputs[0];
    if (n9.inputs[0] != logsumexp_base or n9.inputs[1] != maxv) return false;
    const logsumexp = n9.outputs[0];
    if (n10.inputs[0] != logsumexp or n10.inputs[1] != selected) return false;
    const per_token = n10.outputs[0];
    if (n11.inputs[0] != per_token) return false;
    const out_id = n11.outputs[0];

    const gather_axis = switch (n3.options) {
        .gather => |g| g.axis,
        else => return false,
    };
    if (gather_axis != 1) return false;
    if ((axisFromReduceAxisNode(n4) orelse 99) != 1) return false;
    if ((axisFromReduceAxisNode(n7) orelse 99) != 1) return false;
    if (!(keepdimFromReduceAxisNode(n4) orelse false)) return false;
    if (!(keepdimFromReduceAxisNode(n7) orelse false)) return false;

    const logits = values[logits2d] orelse return error.UnboundGraphValue;
    const indices = values[target_index] orelse return error.UnboundGraphValue;
    _ = values[token_ids] orelse return error.UnboundGraphValue;
    _ = values[logits3d] orelse return error.UnboundGraphValue;
    const device = logits.device() orelse return error.InputNotMaterialized;
    if ((indices.device() orelse return error.InputNotMaterialized) != device) return false;
    if (logits.dtype != .f32 or indices.dtype != .i64) return false;
    if (logits.shape.rank() != 2 or indices.shape.rank() != 2) return false;
    if (indices.shape.dims[0] != logits.shape.dims[0] or indices.shape.dims[1] != 1) return false;
    if (!common.rawStorageInputsArePackedDense(&.{ logits, indices })) return false;

    const rows = logits.shape.dims[0];
    const vocab = logits.shape.dims[1];
    const logits_expanded = try makeLogitsExpandedView(allocator, logits);
    defer logits_expanded.deinit();
    const onehot = try Value.createContiguousWithSource(allocator, &.{ rows, 1, vocab }, .f32, device, false, .graph);
    defer onehot.deinit();
    try kernel_dispatch.oneHot(
        device,
        indices.storage orelse return error.InputNotMaterialized,
        onehot.storage orelse return error.InputNotMaterialized,
        vocab,
    );

    const out_spec = graph.values.items[out_id].spec;
    const out = try Value.createContiguousWithSource(allocator, out_spec.shape.dims, out_spec.dtype, out_spec.device, false, .graph);
    errdefer out.deinit();
    try kernel_dispatch.logSoftmaxNll(
        device,
        .f32,
        logits_expanded.storage orelse return error.InputNotMaterialized,
        onehot.storage orelse return error.InputNotMaterialized,
        out.storage orelse return error.InputNotMaterialized,
        logits_expanded.shape.dims,
        2,
    );

    values[out_id] = out;
    owned[out_id] = true;
    return true;
}
