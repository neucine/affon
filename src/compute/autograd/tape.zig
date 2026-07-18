const std = @import("std");
const OpTag = @import("../types/operation/tag.zig").OpTag;
const SliceRange = @import("../types/operation/options.zig").SliceRange;
const Value = @import("../types/tensor/value.zig").Value;
const Shape = @import("../types/tensor/shape.zig").Shape;
const Layout = @import("../types/tensor/layout.zig").Layout;
const types = @import("types.zig");

pub const Node = types.Node;
pub const Parent = types.Parent;

fn aliasValue(allocator: std.mem.Allocator, value: *const Value) !*const Value {
    const storage = value.storage orelse return error.InputNotMaterialized;
    storage.retain();
    errdefer storage.release();
    const out = try allocator.create(Value);
    errdefer allocator.destroy(out);
    var shape = try Shape.initCopy(allocator, value.shape.dims);
    errdefer shape.deinit();
    var layout = try Layout.initCopy(allocator, value.layout.strides, value.layout.offset);
    errdefer layout.deinit();
    out.* = .{
        .allocator = allocator,
        .shape = shape,
        .dtype = value.dtype,
        .layout = layout,
        .storage = storage,
        .axes = null,
    };
    return out;
}

pub fn createNode(
    allocator: std.mem.Allocator,
    op_tag: OpTag,
    parents: []const Parent,
    saved_inputs: []const *const Value,
    saved_output: ?*const Value,
    saved_aux: ?*const Value,
    axis: ?usize,
    keepdim: ?bool,
    slice_ranges: ?[]const SliceRange,
    permute_axes: ?[]const usize,
    scalar_a: ?f64,
    scalar_b: ?f64,
) !*Node {
    const node = try allocator.create(Node);
    errdefer allocator.destroy(node);
    const parents_copy = try allocator.dupe(Parent, parents);
    errdefer allocator.free(parents_copy);
    const saved_copy = try allocator.alloc(*const Value, saved_inputs.len);
    errdefer allocator.free(saved_copy);
    var saved_count: usize = 0;
    errdefer for (saved_copy[0..saved_count]) |value| @constCast(value).deinit();
    for (saved_inputs, 0..) |value, i| {
        saved_copy[i] = try aliasValue(allocator, value);
        saved_count += 1;
    }
    const saved_output_copy = if (saved_output) |value| try aliasValue(allocator, value) else null;
    errdefer if (saved_output_copy) |value| @constCast(value).deinit();
    const saved_aux_copy = if (saved_aux) |value| try aliasValue(allocator, value) else null;
    errdefer if (saved_aux_copy) |value| @constCast(value).deinit();
    const ranges_copy = if (slice_ranges) |ranges| try allocator.dupe(SliceRange, ranges) else null;
    errdefer if (ranges_copy) |ranges| allocator.free(ranges);
    const axes_copy = if (permute_axes) |axes| try allocator.dupe(usize, axes) else null;
    errdefer if (axes_copy) |axes| allocator.free(axes);

    node.* = .{
        .op_tag = op_tag,
        .parents = parents_copy,
        .saved = .{ .inputs = saved_copy, .output = saved_output_copy, .aux = saved_aux_copy },
        .axis = axis,
        .keepdim = keepdim,
        .slice_ranges = ranges_copy,
        .permute_axes = axes_copy,
        .scalar_a = scalar_a,
        .scalar_b = scalar_b,
    };
    return node;
}

pub fn deinitNode(allocator: std.mem.Allocator, node: *Node) void {
    if (node.saved.aux) |value| @constCast(value).deinit();
    if (node.saved.output) |value| @constCast(value).deinit();
    for (node.saved.inputs) |value| @constCast(value).deinit();
    allocator.free(node.parents);
    allocator.free(node.saved.inputs);
    if (node.slice_ranges) |ranges| allocator.free(ranges);
    if (node.permute_axes) |axes| allocator.free(axes);
    allocator.destroy(node);
}
