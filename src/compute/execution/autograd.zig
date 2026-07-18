const std = @import("std");
const OpTag = @import("../types/operation/tag.zig").OpTag;
const SliceRange = @import("../types/operation/options.zig").SliceRange;
const Tensor = @import("../types/tensor/tensor.zig").Tensor;
const Shape = @import("../types/tensor/shape.zig").Shape;
const Layout = @import("../types/tensor/layout.zig").Layout;
const autograd_types = @import("../types/autograd.zig");
const State = @import("../types/autograd.zig").State;
const Op = @import("../types/operation/op.zig").Op;
const Engine = @import("../engine.zig").Engine;

pub const Node = autograd_types.Node;
pub const Parent = autograd_types.Parent;

fn aliasTensor(allocator: std.mem.Allocator, tensor: *const Tensor) !*const Tensor {
    const storage = tensor.storage orelse return error.InputNotMaterialized;
    storage.retain();
    errdefer storage.release();
    const out = try allocator.create(Tensor);
    errdefer allocator.destroy(out);
    var shape = try Shape.initCopy(allocator, tensor.shape.dims);
    errdefer shape.deinit();
    var layout = try Layout.initCopy(allocator, tensor.layout.strides, tensor.layout.offset);
    errdefer layout.deinit();
    out.* = .{
        .allocator = allocator,
        .shape = shape,
        .dtype = tensor.dtype,
        .layout = layout,
        .storage = storage,
        .axes = null,
        .autograd_state = null,
    };
    return out;
}

pub fn cloneTensor(allocator: std.mem.Allocator, tensor: *const Tensor) !*Tensor {
    return @constCast(try aliasTensor(allocator, tensor));
}

pub fn createScalarLikeTensor(allocator: std.mem.Allocator, like: *const Tensor, scalar: f64) !*Tensor {
    const device = like.device() orelse return error.InputNotMaterialized;
    const tensor = try Tensor.createContiguous(allocator, &.{}, like.dtype, device, false);
    errdefer tensor.deinit();
    switch (like.dtype) {
        .f32 => {
            const value: f32 = @floatCast(scalar);
            try tensor.storage.?.writeFromHost(std.mem.asBytes(&value));
        },
        .f64 => try tensor.storage.?.writeFromHost(std.mem.asBytes(&scalar)),
        .i64 => {
            const value: i64 = @intFromFloat(scalar);
            try tensor.storage.?.writeFromHost(std.mem.asBytes(&value));
        },
    }
    return tensor;
}

pub fn createNode(
    allocator: std.mem.Allocator,
    op_tag: OpTag,
    parents: []const Parent,
    saved_inputs: []const *const Tensor,
    saved_output: ?*const Tensor,
    saved_aux: ?*const Tensor,
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
    const saved_copy = try allocator.alloc(*const Tensor, saved_inputs.len);
    errdefer allocator.free(saved_copy);
    var saved_count: usize = 0;
    errdefer for (saved_copy[0..saved_count]) |tensor| @constCast(tensor).deinit();
    for (saved_inputs, 0..) |tensor, i| {
        saved_copy[i] = try aliasTensor(allocator, tensor);
        saved_count += 1;
    }
    const saved_output_copy = if (saved_output) |tensor| try aliasTensor(allocator, tensor) else null;
    errdefer if (saved_output_copy) |tensor| @constCast(tensor).deinit();
    const saved_aux_copy = if (saved_aux) |tensor| try aliasTensor(allocator, tensor) else null;
    errdefer if (saved_aux_copy) |tensor| @constCast(tensor).deinit();
    const ranges_copy = if (slice_ranges) |ranges| try allocator.dupe(SliceRange, ranges) else null;
    errdefer if (ranges_copy) |ranges| allocator.free(ranges);
    const axes_copy = if (permute_axes) |axes| try allocator.dupe(usize, axes) else null;
    errdefer if (axes_copy) |axes| allocator.free(axes);

    node.* = .{ .op_tag = op_tag, .parents = parents_copy, .saved = .{ .inputs = saved_copy, .output = saved_output_copy, .aux = saved_aux_copy }, .axis = axis, .keepdim = keepdim, .slice_ranges = ranges_copy, .permute_axes = axes_copy, .scalar_a = scalar_a, .scalar_b = scalar_b };
    return node;
}

pub fn deinitNode(allocator: std.mem.Allocator, node: *Node) void {
    if (node.saved.aux) |tensor| @constCast(tensor).deinit();
    if (node.saved.output) |tensor| @constCast(tensor).deinit();
    for (node.saved.inputs) |tensor| @constCast(tensor).deinit();
    allocator.free(node.parents);
    allocator.free(node.saved.inputs);
    if (node.slice_ranges) |ranges| allocator.free(ranges);
    if (node.permute_axes) |axes| allocator.free(axes);
    allocator.destroy(node);
}

pub fn releaseOwnedTensor(tensor: *Tensor) void {
    const state = State.fromTensor(tensor) orelse {
        tensor.deinit();
        return;
    };
    if (state.ref_count == 0) return;
    state.ref_count -= 1;
    if (state.ref_count != 0) return;
    if (state.gradient) |gradient| {
        gradient.deinit();
        state.gradient = null;
    }
    if (state.node) |node| {
        for (node.parents) |parent| {
            if (State.fromTensor(parent.value) != null) releaseOwnedTensor(parent.value);
        }
        deinitNode(state.allocator, node);
        state.node = null;
    }
    tensor.setAutogradStateRaw(null);
    state.allocator.destroy(state);
    tensor.deinit();
}

pub fn accumulateGradient(state: *State, gradient: *Tensor) !void {
    if (state.gradient) |existing| {
        const op = try Op.init(.add, &.{ existing, gradient }, .{ .none = {} });
        var result = try Engine.initWithCurrentTelemetry(state.allocator).executeRaw(op);
        errdefer result.deinit();
        existing.deinit();
        gradient.deinit();
        state.gradient = result.primary;
        result.primary = undefined;
        if (result.secondary) |secondary| secondary.deinit();
        result.secondary = null;
    } else {
        state.gradient = gradient;
    }
}

test {
    _ = Node;
    _ = Parent;
}
