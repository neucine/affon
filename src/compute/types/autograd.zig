const std = @import("std");
const Tensor = @import("tensor/tensor.zig").Tensor;
const DType = @import("tensor/dtype.zig").DType;
const OpTag = @import("operation/tag.zig").OpTag;
const SliceRange = @import("operation/options.zig").SliceRange;

pub const Saved = struct {
    inputs: []const *const Tensor = &.{},
    output: ?*const Tensor = null,
    aux: ?*const Tensor = null,
};

pub const Parent = struct {
    value: *Tensor,
    input_slot: usize,
};

pub const Node = struct {
    op_tag: OpTag,
    parents: []Parent,
    saved: Saved = .{},
    axis: ?usize = null,
    keepdim: ?bool = null,
    slice_ranges: ?[]const SliceRange = null,
    permute_axes: ?[]const usize = null,
    scalar_a: ?f64 = null,
    scalar_b: ?f64 = null,
};

pub const TrackingState = enum { plain, tracked, trainable };

pub const State = struct {
    allocator: std.mem.Allocator,
    requires_grad: bool = false,
    node: ?*Node = null,
    gradient: ?*Tensor = null,
    ref_count: usize = 1,

    pub fn create(allocator: std.mem.Allocator, tensor: *Tensor, requires_grad: bool) !*State {
        if (requires_grad and !isDifferentiableDtype(tensor.dtype)) return error.GradUnsupported;
        if (fromTensor(tensor) != null) return error.TensorAlreadyTracked;
        const state = try allocator.create(State);
        state.* = .{ .allocator = allocator, .requires_grad = requires_grad };
        tensor.setAutogradStateRaw(@ptrCast(state));
        return state;
    }

    pub fn fromTensor(tensor: *const Tensor) ?*State {
        const raw = tensor.autogradStateRaw() orelse return null;
        if (@intFromPtr(raw) % @alignOf(State) != 0) return null;
        return @ptrCast(@alignCast(raw));
    }

    pub fn trackingState(tensor: *const Tensor) TrackingState {
        const state = fromTensor(tensor) orelse return .plain;
        return if (state.requires_grad) .trainable else .tracked;
    }

    pub fn isTrainable(self: *const State) bool {
        return self.requires_grad;
    }

    pub fn isTrainableTensor(tensor: *const Tensor) bool {
        return if (fromTensor(tensor)) |state| state.isTrainable() else false;
    }

    pub fn setTrainable(self: *State, tensor: *const Tensor, enabled: bool) !void {
        if (enabled and !isDifferentiableDtype(tensor.dtype)) return error.GradUnsupported;
        self.requires_grad = enabled;
    }

    pub fn provenanceNode(self: *const State) ?*Node {
        return self.node;
    }

    pub fn retain(self: *State) void {
        self.ref_count += 1;
    }

    pub fn attachNode(self: *State, node: *Node) void {
        for (node.parents) |parent| {
            const parent_state = fromTensor(parent.value) orelse continue;
            parent_state.retain();
        }
        self.node = node;
    }
};

fn isDifferentiableDtype(dtype: DType) bool {
    return dtype == .f32 or dtype == .f64;
}

test {
    _ = State;
    _ = Node;
}

test "autograd state round trips through tensor storage" {
    const allocator = std.testing.allocator;
    const tensor = try Tensor.fromSliceF32(allocator, &.{2}, &.{ 1, 2 });
    defer tensor.deinit();
    const state = try State.create(allocator, tensor, true);
    defer allocator.destroy(state);
    try std.testing.expectEqual(state, State.fromTensor(tensor));
}
