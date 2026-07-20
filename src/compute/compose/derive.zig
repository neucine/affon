const std = @import("std");
const State = @import("../types/autograd.zig").State;
const tape = @import("../execution/autograd.zig");
const ir = @import("../types/ir/index.zig");
const Engine = @import("../engine.zig").Engine;
const graph_builder = @import("builder.zig");
const semantic = @import("../plan/sema/index.zig");
const op_mod = @import("../types/operation/op.zig");
const OpTag = @import("../types/operation/tag.zig").OpTag;
const OpOptions = @import("../types/operation/options.zig").OpOptions;
const Tensor = @import("../types/tensor/tensor.zig").Tensor;
const TensorSpec = @import("../types/tensor/tensor_spec.zig").TensorSpec;
const gradients = @import("gradients.zig");
const telemetry = @import("../telemetry.zig");

pub const UnsupportedDerivedGraph = error{
    UnsupportedDerivedGraph,
};

pub const DerivedGraph = struct {
    graph: ir.ComputeGraph,
    bound_inputs: []*Tensor,
    owned_bound_inputs: []bool,
    node_provenance_origins: []?*const tape.Node,
    tracked_tensors: []*Tensor,

    pub fn deinit(self: *DerivedGraph, allocator: std.mem.Allocator) void {
        for (self.bound_inputs, self.owned_bound_inputs) |value, owned| {
            if (owned) value.deinit();
        }
        allocator.free(self.node_provenance_origins);
        allocator.free(self.bound_inputs);
        allocator.free(self.owned_bound_inputs);
        allocator.free(self.tracked_tensors);
        self.graph.deinit();
        self.* = undefined;
    }
};

const BuildContext = struct {
    allocator: std.mem.Allocator,
    graph: ir.ComputeGraph,
    bound_inputs: std.ArrayList(*Tensor),
    owned_bound_inputs: std.ArrayList(bool),
    bound_tensor_ids: std.AutoHashMap(usize, ir.TensorId),
    grad_ids: std.AutoHashMap(*Tensor, ir.TensorId),
    node_provenance_origins: std.ArrayList(?*const tape.Node),
    tracked_tensors: std.ArrayList(*Tensor),
    current_origin: ?*const tape.Node,

    fn init(allocator: std.mem.Allocator) BuildContext {
        return .{
            .allocator = allocator,
            .graph = ir.ComputeGraph.init(allocator),
            .bound_inputs = .empty,
            .owned_bound_inputs = .empty,
            .bound_tensor_ids = std.AutoHashMap(usize, ir.TensorId).init(allocator),
            .grad_ids = std.AutoHashMap(*Tensor, ir.TensorId).init(allocator),
            .node_provenance_origins = .empty,
            .tracked_tensors = .empty,
            .current_origin = null,
        };
    }

    fn deinit(self: *BuildContext) void {
        for (self.bound_inputs.items, self.owned_bound_inputs.items) |value, owned| {
            if (owned) value.deinit();
        }
        self.tracked_tensors.deinit(self.allocator);
        self.node_provenance_origins.deinit(self.allocator);
        self.grad_ids.deinit();
        self.bound_tensor_ids.deinit();
        self.owned_bound_inputs.deinit(self.allocator);
        self.bound_inputs.deinit(self.allocator);
        self.graph.deinit();
        self.* = undefined;
    }

    fn finish(self: *BuildContext) !DerivedGraph {
        var graph = self.graph;
        self.graph = ir.ComputeGraph.init(self.allocator);
        errdefer graph.deinit();

        const bound_inputs = try self.bound_inputs.toOwnedSlice(self.allocator);
        errdefer self.allocator.free(bound_inputs);

        const owned_bound_inputs = try self.owned_bound_inputs.toOwnedSlice(self.allocator);
        errdefer self.allocator.free(owned_bound_inputs);

        const node_provenance_origins = try self.node_provenance_origins.toOwnedSlice(self.allocator);
        errdefer self.allocator.free(node_provenance_origins);

        const tracked_tensors = try self.tracked_tensors.toOwnedSlice(self.allocator);
        errdefer self.allocator.free(tracked_tensors);

        self.grad_ids.deinit();
        self.bound_tensor_ids.deinit();
        self.* = undefined;

        return .{
            .graph = graph,
            .bound_inputs = bound_inputs,
            .owned_bound_inputs = owned_bound_inputs,
            .node_provenance_origins = node_provenance_origins,
            .tracked_tensors = tracked_tensors,
        };
    }

    fn bindRuntimeTensor(self: *BuildContext, value: *Tensor, owned: bool) !ir.TensorId {
        const key = @intFromPtr(value);
        if (!owned) {
            if (self.bound_tensor_ids.get(key)) |existing| return existing;
        }
        const id = try graph_builder.addInputFromTensor(self.allocator, &self.graph, value);
        try self.node_provenance_origins.append(self.allocator, null);
        try self.bound_inputs.append(self.allocator, value);
        try self.owned_bound_inputs.append(self.allocator, owned);
        if (!owned) try self.bound_tensor_ids.put(key, id);
        return id;
    }

    fn bindOwnedTensor(self: *BuildContext, value: *Tensor) !ir.TensorId {
        return self.bindRuntimeTensor(value, true);
    }

    fn scalarInput(self: *BuildContext, like: *const Tensor, scalar: f64) !ir.TensorId {
        const out = try tape.createScalarLikeTensor(self.allocator, like, scalar);
        return self.bindOwnedTensor(out);
    }

    fn fullTensorLike(self: *BuildContext, like: *const Tensor, scalar: f64) !*Tensor {
        const out = try Tensor.createContiguous(self.allocator, like.shape.dims, like.dtype, like.device() orelse return error.InputNotMaterialized, false);
        const numel = like.shape.numel();
        switch (like.dtype) {
            .f32 => {
                const buf = try self.allocator.alloc(f32, numel);
                defer self.allocator.free(buf);
                @memset(buf, @as(f32, @floatCast(scalar)));
                try out.storage.?.writeFromHost(std.mem.sliceAsBytes(buf));
            },
            .f64 => {
                const buf = try self.allocator.alloc(f64, numel);
                defer self.allocator.free(buf);
                @memset(buf, scalar);
                try out.storage.?.writeFromHost(std.mem.sliceAsBytes(buf));
            },
            .i64 => return error.GradUnsupported,
        }
        return out;
    }

    fn fullTensor(self: *BuildContext, dims: []const usize, dtype: @import("../types/tensor/dtype.zig").DType, device: @import("../types/tensor/device.zig").Device, scalar: f64) !*Tensor {
        const out = try Tensor.createContiguous(self.allocator, dims, dtype, device, false);
        errdefer out.deinit();
        const numel = out.shape.numel();
        switch (dtype) {
            .f32 => {
                const buf = try self.allocator.alloc(f32, numel);
                defer self.allocator.free(buf);
                @memset(buf, @as(f32, @floatCast(scalar)));
                try out.storage.?.writeFromHost(std.mem.sliceAsBytes(buf));
            },
            .f64 => {
                const buf = try self.allocator.alloc(f64, numel);
                defer self.allocator.free(buf);
                @memset(buf, scalar);
                try out.storage.?.writeFromHost(std.mem.sliceAsBytes(buf));
            },
            .i64 => return error.GradUnsupported,
        }
        return out;
    }

    fn expandedAxisIndexTensor(self: *BuildContext, index: *const Tensor, out_shape: []const usize, axis: usize) !*Tensor {
        if (index.dtype != .i64) return error.GradUnsupported;
        if (axis >= out_shape.len) return error.InvalidAxis;
        if (!index.layout.isContiguous(index.shape) or index.layout.offset != 0) return error.GradUnsupported;

        const axis_len = index.shape.numel();
        if (out_shape[axis] != axis_len) return error.ShapeMismatch;
        const device = index.device() orelse return error.InputNotMaterialized;
        const out = try Tensor.createContiguous(self.allocator, out_shape, .i64, device, false);
        errdefer out.deinit();

        const src = try self.allocator.alloc(i64, axis_len);
        defer self.allocator.free(src);
        try index.storage.?.copyToHost(std.mem.sliceAsBytes(src));

        const numel = out.shape.numel();
        const dst = try self.allocator.alloc(i64, numel);
        defer self.allocator.free(dst);
        var outer: usize = 1;
        var inner: usize = 1;
        for (out_shape[0..axis]) |d| outer *= d;
        for (out_shape[axis + 1 ..]) |d| inner *= d;
        for (0..outer) |o| {
            for (0..axis_len) |a| {
                for (0..inner) |i| {
                    const lin = o * axis_len * inner + a * inner + i;
                    dst[lin] = src[a];
                }
            }
        }
        try out.storage.?.writeFromHost(std.mem.sliceAsBytes(dst));
        return out;
    }

    fn ownedI64Tensor(self: *BuildContext, dims: []const usize, values: []const i64, device: @import("../types/tensor/device.zig").Device) !*Tensor {
        const out = try Tensor.createContiguous(self.allocator, dims, .i64, device, false);
        errdefer out.deinit();
        if (out.shape.numel() != values.len) return error.SizeMismatch;
        try out.storage.?.writeFromHost(std.mem.sliceAsBytes(values));
        return out;
    }

    pub fn getAllocator(self: *BuildContext) std.mem.Allocator {
        return self.allocator;
    }

    pub fn bindTensor(self: *BuildContext, value: *Tensor) !ir.TensorId {
        return self.bindRuntimeTensor(value, false);
    }

    pub fn bindValue(self: *BuildContext, value: *Tensor) !ir.TensorId {
        return self.bindTensor(value);
    }

    pub fn valueSpec(self: *BuildContext, value_id: ir.TensorId) TensorSpec {
        return self.graph.values.items[value_id].spec;
    }

    pub fn sameShape(self: *BuildContext, grad_id: ir.TensorId, shape: []const usize) bool {
        return std.mem.eql(usize, self.graph.values.items[grad_id].spec.shape.dims, shape);
    }

    pub fn reduceToParentShape(self: *BuildContext, grad_id: ir.TensorId, parent: *const Tensor) !ir.TensorId {
        return reduceToParentShapeImpl(self, grad_id, parent);
    }

    pub fn identity(self: *BuildContext, grad_id: ir.TensorId) !ir.TensorId {
        _ = self;
        return grad_id;
    }

    pub fn neg(self: *BuildContext, id: ir.TensorId) !ir.TensorId {
        return self.addOp(.neg, &.{id}, .{ .none = {} });
    }
    pub fn sign(self: *BuildContext, id: ir.TensorId) !ir.TensorId {
        return self.addOp(.sign, &.{id}, .{ .none = {} });
    }
    pub fn relu(self: *BuildContext, id: ir.TensorId) !ir.TensorId {
        return self.addOp(.relu, &.{id}, .{ .none = {} });
    }
    pub fn geluGrad(self: *BuildContext, id: ir.TensorId) !ir.TensorId {
        return self.addOp(.gelu_grad, &.{id}, .{ .none = {} });
    }
    pub fn sigmoid(self: *BuildContext, id: ir.TensorId) !ir.TensorId {
        return self.addOp(.sigmoid, &.{id}, .{ .none = {} });
    }
    pub fn sqrt(self: *BuildContext, id: ir.TensorId) !ir.TensorId {
        return self.addOp(.sqrt, &.{id}, .{ .none = {} });
    }
    pub fn log(self: *BuildContext, id: ir.TensorId) !ir.TensorId {
        return self.addOp(.log, &.{id}, .{ .none = {} });
    }
    pub fn tanh(self: *BuildContext, id: ir.TensorId) !ir.TensorId {
        return self.addOp(.tanh, &.{id}, .{ .none = {} });
    }
    pub fn softmax(self: *BuildContext, id: ir.TensorId, axis: usize) !ir.TensorId {
        return self.addOp(.softmax, &.{id}, .{ .softmax = .{ .axis = axis } });
    }
    pub fn logSoftmax(self: *BuildContext, id: ir.TensorId, axis: usize) !ir.TensorId {
        return self.addOp(.log_softmax, &.{id}, .{ .log_softmax = .{ .axis = axis } });
    }
    pub fn crossEntropyIndexedBackward(self: *BuildContext, logits: ir.TensorId, targets: ir.TensorId, grad_out: ir.TensorId, axis: usize) !ir.TensorId {
        return self.addOp(.cross_entropy_indexed_backward, &.{ logits, targets, grad_out }, .{ .cross_entropy_indexed_backward = .{ .axis = axis } });
    }
    pub fn add(self: *BuildContext, a: ir.TensorId, b: ir.TensorId) !ir.TensorId {
        return self.addOp(.add, &.{ a, b }, .{ .none = {} });
    }
    pub fn mul(self: *BuildContext, a: ir.TensorId, b: ir.TensorId) !ir.TensorId {
        return self.addOp(.mul, &.{ a, b }, .{ .none = {} });
    }
    pub fn div(self: *BuildContext, a: ir.TensorId, b: ir.TensorId) !ir.TensorId {
        return self.addOp(.div, &.{ a, b }, .{ .none = {} });
    }
    pub fn sub(self: *BuildContext, a: ir.TensorId, b: ir.TensorId) !ir.TensorId {
        return self.addOp(.sub, &.{ a, b }, .{ .none = {} });
    }
    pub fn eq(self: *BuildContext, a: ir.TensorId, b: ir.TensorId) !ir.TensorId {
        return self.addOp(.eq, &.{ a, b }, .{ .none = {} });
    }
    pub fn lt(self: *BuildContext, a: ir.TensorId, b: ir.TensorId) !ir.TensorId {
        return self.addOp(.lt, &.{ a, b }, .{ .none = {} });
    }
    pub fn gt(self: *BuildContext, a: ir.TensorId, b: ir.TensorId) !ir.TensorId {
        return self.addOp(.gt, &.{ a, b }, .{ .none = {} });
    }
    pub fn matmul(self: *BuildContext, a: ir.TensorId, b: ir.TensorId) !ir.TensorId {
        return self.addOp(.matmul, &.{ a, b }, .{ .none = {} });
    }
    pub fn contiguous(self: *BuildContext, id: ir.TensorId) !ir.TensorId {
        return self.addOp(.contiguous, &.{id}, .{ .none = {} });
    }
    pub fn reshape(self: *BuildContext, id: ir.TensorId, shape: []const usize) !ir.TensorId {
        return self.addOp(.reshape, &.{id}, .{ .reshape = .{ .shape = shape } });
    }
    pub fn slice(self: *BuildContext, id: ir.TensorId, ranges: []const @import("../types/operation/options.zig").SliceRange) !ir.TensorId {
        return self.addOp(.slice, &.{id}, .{ .slice = .{ .ranges = ranges } });
    }
    pub fn transpose(self: *BuildContext, id: ir.TensorId) !ir.TensorId {
        return self.addOp(.transpose, &.{id}, .{ .none = {} });
    }
    pub fn permute(self: *BuildContext, id: ir.TensorId, axes: []const usize) !ir.TensorId {
        return self.addOp(.permute, &.{id}, .{ .permute = .{ .axes = axes } });
    }
    pub fn squeeze(self: *BuildContext, id: ir.TensorId, axis: usize) !ir.TensorId {
        return self.addOp(.squeeze, &.{id}, .{ .squeeze = .{ .axis = axis } });
    }
    pub fn unsqueeze(self: *BuildContext, id: ir.TensorId, axis: usize) !ir.TensorId {
        return self.addOp(.unsqueeze, &.{id}, .{ .unsqueeze = .{ .axis = axis } });
    }
    pub fn cast(self: *BuildContext, id: ir.TensorId, dtype: @import("../types/tensor/dtype.zig").DType) !ir.TensorId {
        return self.addOp(.cast, &.{id}, .{ .cast = .{ .to = dtype } });
    }
    pub fn maskedFillZero(self: *BuildContext, grad_id: ir.TensorId, mask_id: ir.TensorId) !ir.TensorId {
        return self.addOp(.masked_fill, &.{ grad_id, mask_id }, .{ .masked_fill = .{ .value = 0.0 } });
    }
    pub fn whereSelect(self: *BuildContext, cond_id: ir.TensorId, true_id: ir.TensorId, false_id: ir.TensorId) !ir.TensorId {
        return self.addOp(.where, &.{ cond_id, true_id, false_id }, .{ .none = {} });
    }
    pub fn meanAll(self: *BuildContext, id: ir.TensorId) !ir.TensorId {
        return self.addOp(.mean_all, &.{id}, .{ .reduce_all = .{} });
    }
    pub fn sumKeepdim(self: *BuildContext, id: ir.TensorId, axis: usize) !ir.TensorId {
        return self.addOp(.sum_axis, &.{id}, .{ .reduce_axis = .{ .axis = axis, .keepdim = true } });
    }
    pub fn meanKeepdim(self: *BuildContext, id: ir.TensorId, axis: usize) !ir.TensorId {
        return self.addOp(.mean_axis, &.{id}, .{ .reduce_axis = .{ .axis = axis, .keepdim = true } });
    }
    pub fn scatterAdd(self: *BuildContext, base: ir.TensorId, index: ir.TensorId, updates: ir.TensorId, axis: usize) !ir.TensorId {
        return self.addOp(.scatter_add, &.{ base, index, updates }, .{ .scatter_add = .{ .axis = axis } });
    }
    pub fn expandAxisIndex(self: *BuildContext, index: *const Tensor, out_shape: []const usize, axis: usize) !ir.TensorId {
        const value = try self.expandedAxisIndexTensor(index, out_shape, axis);
        return self.bindOwnedTensor(value);
    }
    pub fn scalarLike(self: *BuildContext, like: *const Tensor, scalar: f64) !ir.TensorId {
        return self.scalarInput(like, scalar);
    }
    pub fn fullLike(self: *BuildContext, like: *const Tensor, scalar: f64) !ir.TensorId {
        const value = try self.fullTensorLike(like, scalar);
        return self.bindOwnedTensor(value);
    }
    pub fn full(self: *BuildContext, dims: []const usize, dtype: @import("../types/tensor/dtype.zig").DType, device: @import("../types/tensor/device.zig").Device, scalar: f64) !ir.TensorId {
        const value = try self.fullTensor(dims, dtype, device, scalar);
        return self.bindOwnedTensor(value);
    }
    pub fn ownedI64(self: *BuildContext, dims: []const usize, values: []const i64, device: @import("../types/tensor/device.zig").Device) !ir.TensorId {
        const value = try self.ownedI64Tensor(dims, values, device);
        return self.bindOwnedTensor(value);
    }
    fn addOp(self: *BuildContext, tag: OpTag, input_ids: []const ir.TensorId, options: OpOptions) !ir.TensorId {
        const specs = try self.allocator.alloc(TensorSpec, input_ids.len);
        defer self.allocator.free(specs);
        for (input_ids, 0..) |input_id, i| specs[i] = self.graph.values.items[input_id].spec;

        var info = try semantic.inferFromSpecs(self.allocator, tag, specs, options);
        defer info.deinit();
        const primary = TensorSpec{
            .shape = info.shape,
            .dtype = info.dtype,
            .layout = info.layout,
            .device = info.device,
        };
        const value_id = try self.graph.addOp(tag, input_ids, options, primary);
        try self.node_provenance_origins.append(self.allocator, self.current_origin);
        return value_id;
    }

    fn ensureTrackedTensor(self: *BuildContext, value: *Tensor) !void {
        if (State.fromTensor(value)) |state| {
            if (state.provenanceNode() != null) return;
        }
        for (self.tracked_tensors.items) |existing| {
            if (existing == value) return;
        }
        try self.tracked_tensors.append(self.allocator, value);
    }

    fn accumulateGrad(self: *BuildContext, value: *Tensor, grad_id: ir.TensorId) !void {
        try self.ensureTrackedTensor(value);
        const normalized = try reduceToParentShapeImpl(self, grad_id, value);
        if (self.grad_ids.get(value)) |existing| {
            const normalized_existing = try reduceToParentShapeImpl(self, existing, value);
            const summed = try self.addOp(.add, &.{ normalized_existing, normalized }, .{ .none = {} });
            try self.grad_ids.put(value, summed);
        } else {
            try self.grad_ids.put(value, normalized);
        }
    }
};

fn sameShape(a: []const usize, b: []const usize) bool {
    return std.mem.eql(usize, a, b);
}

fn reduceToParentShapeImpl(ctx: *BuildContext, grad_id: ir.TensorId, parent: *const Tensor) !ir.TensorId {
    const grad_spec = ctx.graph.values.items[grad_id].spec;
    if (sameShape(grad_spec.shape.dims, parent.shape.dims)) return grad_id;
    return ctx.addOp(.reduce_to_shape, &.{grad_id}, .{ .reduce_to_shape = .{ .shape = parent.shape.dims } });
}

fn maybeMaterializeOutput(ctx: *BuildContext, grad_id: ir.TensorId) !ir.TensorId {
    const producer = ctx.graph.values.items[grad_id].producer;
    const node = ctx.graph.nodes.items[producer];
    return switch (node.kind) {
        .input, .constant => ctx.addOp(.contiguous, &.{grad_id}, .{ .none = {} }),
        else => grad_id,
    };
}

fn deriveParentGrad(ctx: *BuildContext, grad_out_id: ir.TensorId, node: *const tape.Node, parent_slot: usize) !ir.TensorId {
    return gradients.deriveParentGrad(ctx, grad_out_id, node, parent_slot) catch |err| switch (err) {
        error.GradUnsupported => UnsupportedDerivedGraph.UnsupportedDerivedGraph,
        else => err,
    };
}

fn topoSort(
    value: *Tensor,
    order: *std.ArrayList(*Tensor),
    visited: *std.AutoHashMap(*Tensor, void),
) !void {
    if (visited.contains(value)) return;
    try visited.put(value, {});
    const state = State.fromTensor(value) orelse return;
    if (state.provenanceNode()) |node| {
        for (node.parents) |p| {
            try topoSort(p.value, order, visited);
        }
    }
    try order.append(value.allocator, value);
}

pub fn buildFromLossTensor(loss: *Tensor) !DerivedGraph {
    var scope = telemetry.startSpan(.root, "compose/derive", .internal, &.{});
    var succeeded = false;
    defer if (!succeeded) scope.endError();

    const loss_state = State.fromTensor(loss) orelse return error.NoGradientGraph;
    if (!loss_state.isTrainable()) return UnsupportedDerivedGraph.UnsupportedDerivedGraph;
    const allocator = loss.allocator;
    var ctx = BuildContext.init(allocator);
    errdefer ctx.deinit();

    var order = std.ArrayList(*Tensor).empty;
    defer order.deinit(allocator);
    var visited = std.AutoHashMap(*Tensor, void).init(allocator);
    defer visited.deinit();
    try topoSort(loss, &order, &visited);

    const one_id = try ctx.fullLike(loss, 1.0);
    try ctx.grad_ids.put(loss, one_id);

    var order_index: usize = order.items.len;
    while (order_index > 0) : (order_index -= 1) {
        const value = order.items[order_index - 1];
        const state = State.fromTensor(value) orelse continue;
        const node = state.provenanceNode() orelse continue;
        ctx.current_origin = node;
        defer ctx.current_origin = null;
        const grad_out_id = ctx.grad_ids.get(value) orelse continue;
        for (node.parents) |parent| {
            _ = State.fromTensor(parent.value) orelse return error.InvalidAutogradState;
            const parent_grad = try deriveParentGrad(&ctx, grad_out_id, node, parent.input_slot);
            try ctx.accumulateGrad(parent.value, parent_grad);
        }
    }

    var outputs = try allocator.alloc(ir.TensorId, ctx.tracked_tensors.items.len);
    defer allocator.free(outputs);
    for (ctx.tracked_tensors.items, 0..) |value, i| {
        const grad_id = ctx.grad_ids.get(value) orelse return error.InvalidAutogradState;
        outputs[i] = try maybeMaterializeOutput(&ctx, grad_id);
    }
    try ctx.graph.setOutputs(outputs);
    const derived = try ctx.finish();
    succeeded = true;
    scope.end();
    return derived;
}

pub fn executeForTensor(loss_tensor: *Tensor, derived: *const DerivedGraph) !void {
    var scope = telemetry.startSpan(.root, "compose/grad", .internal, &.{});
    var succeeded = false;
    defer if (!succeeded) scope.endError();

    executeGraph(loss_tensor, derived) catch |err| switch (err) {
        error.ExecutionNotImplemented => try executeEager(loss_tensor, derived),
        else => return err,
    };
    succeeded = true;
    scope.end();
}

pub fn executeEager(loss_tensor: *Tensor, derived: *const DerivedGraph) !void {
    var values = try loss_tensor.allocator.alloc(?*Tensor, derived.graph.values.items.len);
    defer loss_tensor.allocator.free(values);
    var owned = try loss_tensor.allocator.alloc(bool, derived.graph.values.items.len);
    defer loss_tensor.allocator.free(owned);
    @memset(values, null);
    @memset(owned, false);

    defer {
        for (values, owned) |maybe_value, is_owned| {
            if (is_owned and maybe_value != null) maybe_value.?.deinit();
        }
    }

    for (derived.graph.inputs.items, derived.bound_inputs) |value_id, input| {
        values[value_id] = input;
    }

    for (derived.graph.nodes.items) |node| switch (node.kind) {
        .input, .constant => {},
        .op => |tag| {
            const inputs = try loss_tensor.allocator.alloc(*Tensor, node.inputs.len);
            defer loss_tensor.allocator.free(inputs);
            for (node.inputs, 0..) |input_id, i| {
                inputs[i] = values[input_id] orelse return error.UnboundGraphInput;
            }
            const op = try op_mod.Op.initWithExecutionMetadata(tag, inputs, node.options, node.execution_metadata);
            var result = try Engine.init(loss_tensor.allocator, .{}).executeRaw(op);
            errdefer result.deinit();
            values[node.outputs[0]] = result.primary;
            owned[node.outputs[0]] = true;
            if (node.outputs.len == 2) {
                const secondary = result.secondary orelse return error.InvalidGraphPlan;
                values[node.outputs[1]] = secondary;
                owned[node.outputs[1]] = true;
                result.secondary = null;
            }
            result.primary = undefined;
        },
    };

    if (derived.graph.outputs.items.len != derived.tracked_tensors.len) return error.InvalidGraphPlan;
    for (derived.graph.outputs.items, derived.tracked_tensors) |output_id, tracked_tensor| {
        const state = State.fromTensor(tracked_tensor) orelse return error.InvalidAutogradState;
        const output_value = values[output_id] orelse return error.UnboundGraphOutput;
        const grad = if (owned[output_id]) blk: {
            owned[output_id] = false;
            break :blk output_value;
        } else blk: {
            break :blk try tape.cloneTensor(loss_tensor.allocator, output_value);
        };
        errdefer if (!owned[output_id]) grad.deinit();
        try tape.accumulateGradient(state, grad);
    }
}

fn executeGraph(loss_tensor: *Tensor, derived: *const DerivedGraph) !void {
    const runtime = Engine.init(loss_tensor.allocator, .{});
    var result = try runtime.executeGraph(&derived.graph, derived.bound_inputs);
    defer result.deinit();

    if (result.outputs.len != derived.tracked_tensors.len) return error.InvalidGraphPlan;
    for (derived.tracked_tensors, result.outputs) |value, grad| {
        const state = State.fromTensor(value) orelse return error.InvalidAutogradState;
        const cloned = try tape.cloneTensor(loss_tensor.allocator, grad);
        errdefer cloned.deinit();
        try tape.accumulateGradient(state, cloned);
    }
}
