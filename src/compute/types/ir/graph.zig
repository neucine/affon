const std = @import("std");
const TensorSpec = @import("../tensor/tensor_spec.zig").TensorSpec;
const tensor_value = @import("../tensor/tensor.zig");
const OpTag = @import("../operation/tag.zig").OpTag;
const OpOptions = @import("../operation/options.zig").OpOptions;
const ExecutionMetadata = @import("../operation/execution_metadata.zig").ExecutionMetadata;

pub const TensorId = u32;
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
    inputs: []TensorId,
    outputs: []TensorId,
};

pub const GraphTensor = struct {
    id: TensorId,
    spec: TensorSpec,
    producer: NodeId,
};

pub const ComputeGraph = struct {
    allocator: std.mem.Allocator,
    nodes: std.ArrayList(Node),
    values: std.ArrayList(GraphTensor),
    inputs: std.ArrayList(TensorId),
    outputs: std.ArrayList(TensorId),

    pub fn init(allocator: std.mem.Allocator) ComputeGraph {
        return .{
            .allocator = allocator,
            .nodes = .empty,
            .values = .empty,
            .inputs = .empty,
            .outputs = .empty,
        };
    }

    pub fn deinit(self: *ComputeGraph) void {
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

    pub fn addInput(self: *ComputeGraph, spec: TensorSpec) !TensorId {
        const id: TensorId = @intCast(self.values.items.len);
        const node_id: NodeId = @intCast(self.nodes.items.len);
        const outputs = try self.allocator.alloc(TensorId, 1);
        outputs[0] = id;
        const inputs = try self.allocator.alloc(TensorId, 0);
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
        self: *ComputeGraph,
        tag: OpTag,
        input_ids: []const TensorId,
        options: OpOptions,
        output_spec: TensorSpec,
    ) !TensorId {
        const ids = try self.addOpMultiWithExecutionMetadata(tag, input_ids, options, .{}, &.{output_spec});
        defer self.allocator.free(ids);
        return ids[0];
    }

    pub fn addOpWithExecutionMetadata(
        self: *ComputeGraph,
        tag: OpTag,
        input_ids: []const TensorId,
        options: OpOptions,
        execution_metadata: ExecutionMetadata,
        output_spec: TensorSpec,
    ) !TensorId {
        const ids = try self.addOpMultiWithExecutionMetadata(tag, input_ids, options, execution_metadata, &.{output_spec});
        defer self.allocator.free(ids);
        return ids[0];
    }

    pub fn addOpMulti(
        self: *ComputeGraph,
        tag: OpTag,
        input_ids: []const TensorId,
        options: OpOptions,
        output_specs: []const TensorSpec,
    ) ![]TensorId {
        return self.addOpMultiWithExecutionMetadata(tag, input_ids, options, .{}, output_specs);
    }

    pub fn addOpMultiWithExecutionMetadata(
        self: *ComputeGraph,
        tag: OpTag,
        input_ids: []const TensorId,
        options: OpOptions,
        execution_metadata: ExecutionMetadata,
        output_specs: []const TensorSpec,
    ) ![]TensorId {
        if (output_specs.len == 0) return error.InvalidOutputCount;

        const output_id: TensorId = @intCast(self.values.items.len);
        const node_id: NodeId = @intCast(self.nodes.items.len);
        const inputs = try self.allocator.alloc(TensorId, input_ids.len);
        @memcpy(inputs, input_ids);
        errdefer self.allocator.free(inputs);
        const outputs = try self.allocator.alloc(TensorId, output_specs.len);
        for (outputs, 0..) |*out, i| out.* = output_id + @as(TensorId, @intCast(i));
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

        const ids = try self.allocator.alloc(TensorId, outputs.len);
        @memcpy(ids, outputs);
        return ids;
    }

    pub fn setOutputs(self: *ComputeGraph, output_ids: []const TensorId) !void {
        self.outputs.clearRetainingCapacity();
        try self.outputs.appendSlice(self.allocator, output_ids);
    }
};

// Internal code may continue to use the shorter name while the public API
// presents this representation as a compute graph.
pub const Graph = ComputeGraph;

fn cloneSpec(allocator: std.mem.Allocator, spec: TensorSpec) !TensorSpec {
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
    const s = TensorSpec{
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
