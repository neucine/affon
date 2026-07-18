const std = @import("std");
const Value = @import("../types/tensor/value.zig").Value;
const DType = @import("../types/tensor/dtype.zig").DType;
const types = @import("types.zig");
const tape = @import("tape.zig");

pub const Parent = types.Parent;
pub const Node = types.Node;
pub const TrackingState = enum { plain, tracked, trainable };

pub const State = struct {
    allocator: std.mem.Allocator,
    requires_grad: bool = false,
    node: ?*Node = null,
    ref_count: usize = 1,

    pub fn create(allocator: std.mem.Allocator, value: *Value, requires_grad: bool) !*State {
        if (requires_grad and !isDifferentiableDtype(value.dtype)) return error.GradUnsupported;
        if (fromValue(value) != null) return error.ValueAlreadyTracked;
        const state = try allocator.create(State);
        state.* = .{ .allocator = allocator, .requires_grad = requires_grad };
        value.setAutogradStateRaw(@ptrCast(state));
        return state;
    }

    pub fn fromValue(value: *const Value) ?*State {
        const raw = value.autogradStateRaw() orelse return null;
        return @ptrCast(@alignCast(raw));
    }

    pub fn trackingState(value: *const Value) TrackingState {
        const state = fromValue(value) orelse return .plain;
        return if (state.requires_grad) .trainable else .tracked;
    }

    pub fn isTrainable(self: *const State) bool {
        return self.requires_grad;
    }

    pub fn isTrainableValue(value: *const Value) bool {
        return if (fromValue(value)) |state| state.isTrainable() else false;
    }

    pub fn setTrainable(self: *State, value: *const Value, enabled: bool) !void {
        if (enabled and !isDifferentiableDtype(value.dtype)) return error.GradUnsupported;
        self.requires_grad = enabled;
    }

    pub fn provenanceNode(self: *const State) ?*Node {
        return self.node;
    }

    pub fn retain(self: *State) void {
        self.ref_count += 1;
    }

    pub fn attachNode(self: *State, node: *Node) void {
        self.node = node;
    }

    pub fn releaseOwnedValue(value: *Value) void {
        const state = fromValue(value) orelse {
            value.deinit();
            return;
        };
        if (state.ref_count == 0) return;
        state.ref_count -= 1;
        if (state.ref_count != 0) return;
        if (state.node) |node| tape.deinitNode(state.allocator, node);
        value.setAutogradStateRaw(null);
        state.allocator.destroy(state);
        value.deinit();
    }
};

fn isDifferentiableDtype(dtype: DType) bool {
    return dtype == .f32 or dtype == .f64;
}
