const std = @import("std");
const ValueSpec = @import("../tensor/value_spec.zig").ValueSpec;
const tensor_value = @import("../tensor/value.zig");
const OpTag = @import("../operation/tag.zig").OpTag;
const OpOptions = @import("../operation/options.zig").OpOptions;
const ExecutionMetadata = @import("../operation/execution_metadata.zig").ExecutionMetadata;

pub const ValueId = u32;
pub const NodeId = u32;

pub const NodeKind = union(enum) {
    input: void,
    constant: void,
    op: OpTag,
};

pub const Node = struct {
    id: NodeId,
    kind: NodeKind,
    module_path: ?[]const u8 = null,
    options: OpOptions,
    execution_metadata: ExecutionMetadata,
    inputs: []ValueId,
    outputs: []ValueId,
};

pub const GraphValue = struct {
    id: ValueId,
    spec: ValueSpec,
    producer: NodeId,
};

pub const Graph = struct {
    allocator: std.mem.Allocator,
    nodes: std.ArrayList(Node),
    values: std.ArrayList(GraphValue),
    inputs: std.ArrayList(ValueId),
    outputs: std.ArrayList(ValueId),

    pub fn init(allocator: std.mem.Allocator) Graph {
        return .{
            .allocator = allocator,
            .nodes = .empty,
            .values = .empty,
            .inputs = .empty,
            .outputs = .empty,
        };
    }

    pub fn deinit(self: *Graph) void {
        for (self.nodes.items) |node| {
            if (node.module_path) |module_path| self.allocator.free(module_path);
            deinitOpOptions(self.allocator, node.options);
            self.allocator.free(node.inputs);
            self.allocator.free(node.outputs);
        }
        for (self.values.items) |*value| {
            var spec = value.spec;
            tensor_value.deinitAxes(self.allocator, spec.axes);
            spec.layout.deinit();
            spec.shape.deinit();
        }
        self.outputs.deinit(self.allocator);
        self.inputs.deinit(self.allocator);
        self.values.deinit(self.allocator);
        self.nodes.deinit(self.allocator);
        self.* = undefined;
    }

    pub fn addInput(self: *Graph, spec: ValueSpec) !ValueId {
        const id: ValueId = @intCast(self.values.items.len);
        const node_id: NodeId = @intCast(self.nodes.items.len);
        const outputs = try self.allocator.alloc(ValueId, 1);
        outputs[0] = id;
        const inputs = try self.allocator.alloc(ValueId, 0);
        errdefer self.allocator.free(inputs);

        try self.nodes.append(self.allocator, .{
            .id = node_id,
            .kind = .{ .input = {} },
            .module_path = null,
            .options = .{ .none = {} },
            .execution_metadata = .{},
            .inputs = inputs,
            .outputs = outputs,
        });
        errdefer _ = self.nodes.pop();

        try self.values.append(self.allocator, .{
            .id = id,
            .spec = try cloneSpec(self.allocator, spec),
            .producer = node_id,
        });
        errdefer _ = self.values.pop();

        try self.inputs.append(self.allocator, id);
        return id;
    }

    pub fn addOp(
        self: *Graph,
        tag: OpTag,
        input_ids: []const ValueId,
        options: OpOptions,
        output_spec: ValueSpec,
    ) !ValueId {
        const ids = try self.addOpMultiWithExecutionMetadata(tag, input_ids, options, .{}, &.{output_spec});
        defer self.allocator.free(ids);
        return ids[0];
    }

    pub fn addOpWithExecutionMetadata(
        self: *Graph,
        tag: OpTag,
        input_ids: []const ValueId,
        options: OpOptions,
        execution_metadata: ExecutionMetadata,
        output_spec: ValueSpec,
    ) !ValueId {
        const ids = try self.addOpMultiWithExecutionMetadata(tag, input_ids, options, execution_metadata, &.{output_spec});
        defer self.allocator.free(ids);
        return ids[0];
    }

    pub fn addOpMulti(
        self: *Graph,
        tag: OpTag,
        input_ids: []const ValueId,
        options: OpOptions,
        output_specs: []const ValueSpec,
    ) ![]ValueId {
        return self.addOpMultiWithExecutionMetadata(tag, input_ids, options, .{}, output_specs);
    }

    pub fn addOpMultiWithExecutionMetadata(
        self: *Graph,
        tag: OpTag,
        input_ids: []const ValueId,
        options: OpOptions,
        execution_metadata: ExecutionMetadata,
        output_specs: []const ValueSpec,
    ) ![]ValueId {
        if (output_specs.len == 0) return error.InvalidOutputCount;

        const output_id: ValueId = @intCast(self.values.items.len);
        const node_id: NodeId = @intCast(self.nodes.items.len);
        const inputs = try self.allocator.alloc(ValueId, input_ids.len);
        @memcpy(inputs, input_ids);
        errdefer self.allocator.free(inputs);
        const outputs = try self.allocator.alloc(ValueId, output_specs.len);
        for (outputs, 0..) |*out, i| out.* = output_id + @as(ValueId, @intCast(i));
        errdefer self.allocator.free(outputs);

        const owned_options = try cloneOpOptions(self.allocator, options);
        errdefer deinitOpOptions(self.allocator, owned_options);

        try self.nodes.append(self.allocator, .{
            .id = node_id,
            .kind = .{ .op = tag },
            .module_path = null,
            .options = owned_options,
            .execution_metadata = execution_metadata,
            .inputs = inputs,
            .outputs = outputs,
        });
        errdefer _ = self.nodes.pop();

        for (output_specs, 0..) |spec, i| {
            try self.values.append(self.allocator, .{
                .id = outputs[i],
                .spec = try cloneSpec(self.allocator, spec),
                .producer = node_id,
            });
        }

        const ids = try self.allocator.alloc(ValueId, outputs.len);
        @memcpy(ids, outputs);
        return ids;
    }

    pub fn setOutputs(self: *Graph, output_ids: []const ValueId) !void {
        self.outputs.clearRetainingCapacity();
        try self.outputs.appendSlice(self.allocator, output_ids);
    }
};

fn cloneSpec(allocator: std.mem.Allocator, spec: ValueSpec) !ValueSpec {
    return .{
        .shape = try @import("../tensor/shape.zig").Shape.initCopy(allocator, spec.shape.dims),
        .dtype = spec.dtype,
        .layout = try @import("../tensor/layout.zig").Layout.initCopy(allocator, spec.layout.strides, spec.layout.offset),
        .device = spec.device,
        .axes = try tensor_value.cloneAxes(allocator, spec.axes),
    };
}

fn cloneOpOptions(allocator: std.mem.Allocator, options: OpOptions) !OpOptions {
    return switch (options) {
        .reshape => |reshape| .{ .reshape = .{ .shape = try cloneUsizeSlice(allocator, reshape.shape) } },
        .slice => |slice| .{ .slice = .{ .ranges = try cloneSliceRanges(allocator, slice.ranges) } },
        .permute => |permute| .{ .permute = .{ .axes = try cloneUsizeSlice(allocator, permute.axes) } },
        .transpose => |transpose| .{
            .transpose = .{
                .permutation = if (transpose.permutation) |perm| try cloneUsizeSlice(allocator, perm) else null,
            },
        },
        else => options,
    };
}

fn deinitOpOptions(allocator: std.mem.Allocator, options: OpOptions) void {
    switch (options) {
        .reshape => |reshape| allocator.free(reshape.shape),
        .slice => |slice| allocator.free(slice.ranges),
        .permute => |permute| allocator.free(permute.axes),
        .transpose => |transpose| if (transpose.permutation) |perm| allocator.free(perm),
        else => {},
    }
}

fn cloneUsizeSlice(allocator: std.mem.Allocator, input: []const usize) ![]usize {
    const out = try allocator.alloc(usize, input.len);
    @memcpy(out, input);
    return out;
}

fn cloneSliceRanges(allocator: std.mem.Allocator, input: []const @import("../operation/options.zig").SliceRange) ![]@import("../operation/options.zig").SliceRange {
    const out = try allocator.alloc(@import("../operation/options.zig").SliceRange, input.len);
    @memcpy(out, input);
    return out;
}

test "graph captures input and one op node" {
    const allocator = std.testing.allocator;
    var graph = Graph.init(allocator);
    defer graph.deinit();

    var shape = try @import("../tensor/shape.zig").Shape.initCopy(allocator, &.{2});
    defer shape.deinit();
    var layout = try @import("../tensor/layout.zig").Layout.initContiguous(allocator, shape);
    defer layout.deinit();
    const s = ValueSpec{
        .shape = shape,
        .dtype = .f32,
        .layout = layout,
        .device = .cpu,
    };

    const a = try graph.addInput(s);
    const b = try graph.addOp(.neg, &.{a}, .{ .none = {} }, s);
    try graph.setOutputs(&.{b});

    try std.testing.expectEqual(@as(usize, 2), graph.nodes.items.len);
    try std.testing.expectEqual(@as(usize, 2), graph.values.items.len);
    try std.testing.expectEqual(@as(usize, 1), graph.outputs.items.len);
}
