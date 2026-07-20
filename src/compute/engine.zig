const std = @import("std");
const compat = @import("../support/compat.zig");
const tensor = @import("types/tensor/index.zig");
const operation = @import("types/operation/index.zig");
const eager = @import("execution/eager/index.zig");
const semantic = @import("plan/sema/index.zig");
const eager_planning = @import("plan/eager.zig");
const graph_planning = @import("plan/graph.zig");
const graph_execution = @import("execution/graph/index.zig");
const transfer = @import("execution/transfer.zig");
const backend_dispatch = @import("backend/dispatch.zig");
const ir = @import("types/ir/index.zig");
const telemetry = @import("telemetry.zig");

pub const Tensor = tensor.Tensor;
pub const ComputeGraph = ir.ComputeGraph;
pub const GraphResult = graph_execution.Result;
pub const EagerResult = eager.ExecutionResult;
pub const DType = tensor.DType;
pub const Device = tensor.Device;

/// Operations exposed by the language-neutral compute engine.
/// Backend selection, semantic validation, planning, and fusion happen below
/// this boundary.
pub const Operation = enum {
    add,
    sub,
    mul,
    div,
    gt,
    abs,
    exp,
    log,
    neg,
    sqrt,
    sign,
    relu,
    sigmoid,
    silu,
    tanh,
    gelu,
    dot,
    matmul,
    sum,
    mean,
    min,
    max,
    variance,
    std,
    argmin,
    argmax,
    contiguous,
    clamp,
    softmax,
    cat,
    stack,
    reshape,
    slice,
    gather,
    index_select,
    topk,
    one_hot,
    permute,
    transpose,
    squeeze,
    unsqueeze,
    cast,
    where,
    masked_fill,
    cross_entropy_indexed,
};

pub const Outputs = struct {
    primary: *Tensor,
    secondary: ?*Tensor = null,

    pub fn deinit(self: *Outputs) void {
        if (self.secondary) |value| value.deinit();
        self.primary.deinit();
        self.* = undefined;
    }
};

pub const Engine = struct {
    allocator: std.mem.Allocator,

    pub const Config = struct {};

    pub fn init(allocator: std.mem.Allocator, _: Config) Engine {
        return .{ .allocator = allocator };
    }

    pub fn fromF32(self: Engine, shape: []const usize, values: []const f32) !*Tensor {
        return Tensor.fromSliceF32(self.allocator, shape, values);
    }

    pub fn fromF64(self: Engine, shape: []const usize, values: []const f64) !*Tensor {
        return Tensor.fromSliceF64(self.allocator, shape, values);
    }

    pub fn fromI64(self: Engine, shape: []const usize, values: []const i64) !*Tensor {
        return Tensor.fromSliceI64(self.allocator, shape, values);
    }

    pub fn empty(self: Engine, shape: []const usize, dtype: DType, device: Device) !*Tensor {
        return Tensor.createContiguous(self.allocator, shape, dtype, device, false);
    }

    pub fn zeros(self: Engine, shape: []const usize, dtype: DType, device: Device) !*Tensor {
        return Tensor.createContiguous(self.allocator, shape, dtype, device, true);
    }

    pub fn copyToHost(self: Engine, value: *const Tensor, out: []u8) !void {
        _ = self;
        try (try value.requireRuntimeBacking()).copyToHost(out);
    }

    pub fn scaleInPlace(self: Engine, value: *Tensor, scale: f64) !void {
        _ = self;
        const backing = try value.requireRuntimeBacking();
        try backend_dispatch.update(value.device() orelse .cpu, .scale, value.dtype, backing, null, scale);
    }

    pub fn copyInto(self: Engine, target: *Tensor, source: *const Tensor) !void {
        if (target.shape.numel() != source.shape.numel()) return error.ShapeMismatch;
        if (!source.layout.isContiguous(source.shape) or source.layout.offset != 0) {
            const contiguous_source = try self.contiguous(@constCast(source));
            defer contiguous_source.deinit();
            return self.copyInto(target, contiguous_source);
        }
        if (target.dtype == source.dtype) {
            _ = try transfer.copyValueStorageInto(self.allocator, source, target);
            return;
        }
        const converted = try self.cast(@constCast(source), target.dtype);
        defer converted.deinit();
        _ = try transfer.copyValueStorageInto(self.allocator, converted, target);
    }

    pub fn invoke(self: Engine, op: Operation, inputs: []const *Tensor) !*Tensor {
        return self.invokeWithOptions(op, inputs, defaultOptions(op));
    }

    pub fn invokeWithOptions(self: Engine, op: Operation, inputs: []const *Tensor, options: operation.OpOptions) !*Tensor {
        var result = try self.invokeAllWithOptions(op, inputs, options);
        if (result.secondary != null) {
            result.deinit();
            return error.MultiOutputRequiresOutputs;
        }
        const primary = result.primary;
        result.primary = undefined;
        return primary;
    }

    pub fn invokeAll(self: Engine, op: Operation, inputs: []const *Tensor) !Outputs {
        return self.invokeAllWithOptions(op, inputs, defaultOptions(op));
    }

    pub fn invokeAllWithOptions(self: Engine, op: Operation, inputs: []const *Tensor, options: operation.OpOptions) !Outputs {
        const raw = try operation.Op.init(toTag(op), inputs, options);
        var result = try self.executeRaw(raw);
        const outputs = Outputs{ .primary = result.primary, .secondary = result.secondary };
        result.primary = undefined;
        result.secondary = null;
        return outputs;
    }

    pub fn executeRaw(self: Engine, raw: operation.Op) !EagerResult {
        var execution_scope = telemetry.beginTrace(.execution, telemetry.traces.run);
        var execution_succeeded = false;
        defer if (!execution_succeeded) execution_scope.endError();

        var infer_scope = execution_scope.child("plan/infer", .internal, &.{});
        var info = semantic.infer(self.allocator, raw) catch |err| {
            infer_scope.endAt(compat.nanoTimestamp(), .err);
            return err;
        };
        infer_scope.end();
        defer info.deinit();

        var plan_scope = execution_scope.child("plan/create", .internal, &.{});
        var plan = eager_planning.create(self.allocator, raw, info) catch |err| {
            plan_scope.endAt(compat.nanoTimestamp(), .err);
            return err;
        };
        plan_scope.end();
        defer plan.deinit();
        const result = try eager.executeAllWithPlan(self.allocator, raw, &plan);
        execution_succeeded = true;
        execution_scope.end();
        return result;
    }

    pub fn executeGraph(self: Engine, graph: *const ComputeGraph, inputs: []const *Tensor) !GraphResult {
        var plan_scope = telemetry.startSpan(.root, "plan/create", .internal, &.{});
        var plan = graph_planning.create(self.allocator, graph) catch |err| {
            plan_scope.endAt(compat.nanoTimestamp(), .err);
            return err;
        };
        plan_scope.end();
        defer plan.deinit();
        return graph_execution.execute(self.allocator, graph, &plan, inputs);
    }

    pub fn add(self: Engine, lhs: *Tensor, rhs: *Tensor) !*Tensor {
        return self.invoke(.add, &.{ lhs, rhs });
    }

    pub fn sub(self: Engine, lhs: *Tensor, rhs: *Tensor) !*Tensor {
        return self.invoke(.sub, &.{ lhs, rhs });
    }

    pub fn mul(self: Engine, lhs: *Tensor, rhs: *Tensor) !*Tensor {
        return self.invoke(.mul, &.{ lhs, rhs });
    }

    pub fn div(self: Engine, lhs: *Tensor, rhs: *Tensor) !*Tensor {
        return self.invoke(.div, &.{ lhs, rhs });
    }

    pub fn gt(self: Engine, lhs: *Tensor, rhs: *Tensor) !*Tensor {
        return self.invoke(.gt, &.{ lhs, rhs });
    }

    pub fn abs(self: Engine, value: *Tensor) !*Tensor {
        return self.invoke(.abs, &.{value});
    }

    pub fn exp(self: Engine, value: *Tensor) !*Tensor {
        return self.invoke(.exp, &.{value});
    }

    pub fn log(self: Engine, value: *Tensor) !*Tensor {
        return self.invoke(.log, &.{value});
    }

    pub fn neg(self: Engine, value: *Tensor) !*Tensor {
        return self.invoke(.neg, &.{value});
    }

    pub fn sqrt(self: Engine, value: *Tensor) !*Tensor {
        return self.invoke(.sqrt, &.{value});
    }

    pub fn sign(self: Engine, value: *Tensor) !*Tensor {
        return self.invoke(.sign, &.{value});
    }

    pub fn relu(self: Engine, value: *Tensor) !*Tensor {
        return self.invoke(.relu, &.{value});
    }

    pub fn sigmoid(self: Engine, value: *Tensor) !*Tensor {
        return self.invoke(.sigmoid, &.{value});
    }

    pub fn silu(self: Engine, value: *Tensor) !*Tensor {
        return self.invoke(.silu, &.{value});
    }

    pub fn tanh(self: Engine, value: *Tensor) !*Tensor {
        return self.invoke(.tanh, &.{value});
    }

    pub fn gelu(self: Engine, value: *Tensor) !*Tensor {
        return self.invoke(.gelu, &.{value});
    }

    pub fn matmul(self: Engine, lhs: *Tensor, rhs: *Tensor) !*Tensor {
        return self.invoke(.matmul, &.{ lhs, rhs });
    }

    pub fn dot(self: Engine, lhs: *Tensor, rhs: *Tensor) !*Tensor {
        return self.invoke(.dot, &.{ lhs, rhs });
    }

    pub fn sum(self: Engine, value: *Tensor) !*Tensor {
        return self.invoke(.sum, &.{value});
    }

    pub fn reduce(self: Engine, op: Operation, value: *Tensor, axis: ?usize, keepdim: bool) !*Tensor {
        const tag: operation.OpTag = switch (op) {
            .sum => if (axis == null) .sum_all else .sum_axis,
            .mean => if (axis == null) .mean_all else .mean_axis,
            .min => if (axis == null) .min_all else .min_axis,
            .max => if (axis == null) .max_all else .max_axis,
            .variance => if (axis == null) .variance_all else .variance_axis,
            .std => if (axis == null) .std_all else .std_axis,
            .argmin => if (axis == null) .argmin_all else .argmin_axis,
            .argmax => if (axis == null) .argmax_all else .argmax_axis,
            else => return error.InvalidReduction,
        };
        const options: operation.OpOptions = if (axis) |value_axis|
            .{ .reduce_axis = .{ .axis = value_axis, .keepdim = keepdim } }
        else
            .{ .reduce_all = .{ .keepdim = keepdim } };
        const raw = try operation.Op.init(tag, &.{value}, options);
        var result = try self.executeRaw(raw);
        const primary = result.primary;
        result.primary = undefined;
        if (result.secondary) |secondary| secondary.deinit();
        return primary;
    }

    pub fn mean(self: Engine, value: *Tensor) !*Tensor {
        return self.invoke(.mean, &.{value});
    }

    pub fn min(self: Engine, value: *Tensor) !*Tensor {
        return self.invoke(.min, &.{value});
    }

    pub fn max(self: Engine, value: *Tensor) !*Tensor {
        return self.invoke(.max, &.{value});
    }

    pub fn variance(self: Engine, value: *Tensor) !*Tensor {
        return self.invoke(.variance, &.{value});
    }

    pub fn stddev(self: Engine, value: *Tensor) !*Tensor {
        return self.invoke(.std, &.{value});
    }

    pub fn argmin(self: Engine, value: *Tensor) !*Tensor {
        return self.invoke(.argmin, &.{value});
    }

    pub fn argmax(self: Engine, value: *Tensor) !*Tensor {
        return self.invoke(.argmax, &.{value});
    }

    pub fn clamp(self: Engine, value: *Tensor, minimum: f64, maximum: f64) !*Tensor {
        return self.invokeWithOptions(.clamp, &.{value}, .{ .clamp = .{ .min = minimum, .max = maximum } });
    }

    pub fn softmax(self: Engine, value: *Tensor, axis: usize) !*Tensor {
        return self.invokeWithOptions(.softmax, &.{value}, .{ .softmax = .{ .axis = axis } });
    }

    pub fn reshape(self: Engine, value: *Tensor, shape: []const usize) !*Tensor {
        return self.invokeWithOptions(.reshape, &.{value}, .{ .reshape = .{ .shape = shape } });
    }

    pub fn slice(self: Engine, value: *Tensor, ranges: []const operation.SliceRange) !*Tensor {
        return self.invokeWithOptions(.slice, &.{value}, .{ .slice = .{ .ranges = ranges } });
    }

    pub fn contiguous(self: Engine, value: *Tensor) !*Tensor {
        return self.invoke(.contiguous, &.{value});
    }

    pub fn permute(self: Engine, value: *Tensor, axes: []const usize) !*Tensor {
        return self.invokeWithOptions(.permute, &.{value}, .{ .permute = .{ .axes = axes } });
    }

    pub fn transpose(self: Engine, value: *Tensor, axis_a: usize, axis_b: usize) !*Tensor {
        const axes = self.allocator.alloc(usize, value.shape.rank()) catch return error.OutOfMemory;
        defer self.allocator.free(axes);
        for (axes, 0..) |*axis, index| axis.* = index;
        if (axis_a >= axes.len or axis_b >= axes.len) return error.InvalidAxis;
        std.mem.swap(usize, &axes[axis_a], &axes[axis_b]);
        return self.invokeWithOptions(.transpose, &.{value}, .{ .transpose = .{ .permutation = axes } });
    }

    pub fn squeeze(self: Engine, value: *Tensor, axis: ?usize) !*Tensor {
        return self.invokeWithOptions(.squeeze, &.{value}, .{ .squeeze = .{ .axis = axis } });
    }

    pub fn unsqueeze(self: Engine, value: *Tensor, axis: usize) !*Tensor {
        return self.invokeWithOptions(.unsqueeze, &.{value}, .{ .unsqueeze = .{ .axis = axis } });
    }

    pub fn cat(self: Engine, values: []const *Tensor, axis: usize) !*Tensor {
        return self.invokeWithOptions(.cat, values, .{ .concat = .{ .axis = axis } });
    }

    pub fn stack(self: Engine, values: []const *Tensor, axis: usize) !*Tensor {
        return self.invokeWithOptions(.stack, values, .{ .stack = .{ .axis = axis } });
    }

    pub fn gather(self: Engine, value: *Tensor, axis: usize, index: *Tensor) !*Tensor {
        return self.invokeWithOptions(.gather, &.{ value, index }, .{ .gather = .{ .axis = axis } });
    }

    pub fn indexSelect(self: Engine, value: *Tensor, axis: usize, index: *Tensor) !*Tensor {
        return self.invokeWithOptions(.index_select, &.{ value, index }, .{ .index_select = .{ .axis = axis } });
    }

    pub fn oneHot(self: Engine, value: *Tensor, num_classes: usize) !*Tensor {
        return self.invokeWithOptions(.one_hot, &.{value}, .{ .one_hot = .{ .num_classes = num_classes } });
    }

    pub fn topK(self: Engine, value: *Tensor, k: usize, axis: usize) !Outputs {
        return self.invokeAllWithOptions(.topk, &.{value}, .{ .topk = .{ .k = k, .axis = axis } });
    }

    pub fn cast(self: Engine, value: *Tensor, dtype: DType) !*Tensor {
        return self.invokeWithOptions(.cast, &.{value}, .{ .cast = .{ .to = dtype } });
    }

    pub fn whereSelect(self: Engine, condition: *Tensor, on_true: *Tensor, on_false: *Tensor) !*Tensor {
        return self.invoke(.where, &.{ condition, on_true, on_false });
    }

    pub fn maskedFill(self: Engine, value: *Tensor, mask: *Tensor, fill: f64) !*Tensor {
        return self.invokeWithOptions(.masked_fill, &.{ value, mask }, .{ .masked_fill = .{ .value = fill } });
    }

    pub fn crossEntropyIndexed(self: Engine, logits: *Tensor, targets: *Tensor, axis: usize) !*Tensor {
        return self.invokeWithOptions(.cross_entropy_indexed, &.{ logits, targets }, .{ .cross_entropy_indexed = .{ .axis = axis } });
    }
};

fn toTag(op: Operation) operation.OpTag {
    return switch (op) {
        .add => .add,
        .sub => .sub,
        .mul => .mul,
        .div => .div,
        .gt => .gt,
        .abs => .abs,
        .exp => .exp,
        .log => .log,
        .neg => .neg,
        .sqrt => .sqrt,
        .sign => .sign,
        .relu => .relu,
        .sigmoid => .sigmoid,
        .silu => .silu,
        .tanh => .tanh,
        .gelu => .gelu,
        .dot => .dot,
        .matmul => .matmul,
        .sum => .sum_all,
        .mean => .mean_all,
        .min => .min_all,
        .max => .max_all,
        .variance => .variance_all,
        .std => .std_all,
        .argmin => .argmin_all,
        .argmax => .argmax_all,
        .contiguous => .contiguous,
        .clamp => .clamp,
        .softmax => .softmax,
        .cat => .cat,
        .stack => .stack,
        .reshape => .reshape,
        .slice => .slice,
        .gather => .gather,
        .index_select => .index_select,
        .topk => .topk,
        .one_hot => .one_hot,
        .permute => .permute,
        .transpose => .transpose,
        .squeeze => .squeeze,
        .unsqueeze => .unsqueeze,
        .cast => .cast,
        .where => .where,
        .masked_fill => .masked_fill,
        .cross_entropy_indexed => .cross_entropy_indexed,
    };
}

fn defaultOptions(op: Operation) operation.OpOptions {
    return switch (op) {
        .abs, .exp, .log, .neg, .sqrt, .sign, .relu, .sigmoid, .silu, .tanh, .gelu, .add, .sub, .mul, .div, .gt, .dot, .matmul, .contiguous => .{ .none = {} },
        .sum, .mean, .min, .max, .variance, .std, .argmin, .argmax => .{ .reduce_all = .{} },
        .clamp, .softmax, .cat, .stack, .reshape, .slice, .gather, .index_select, .topk, .one_hot, .permute, .transpose, .squeeze, .unsqueeze, .cast, .where => .{ .none = {} },
        .masked_fill, .cross_entropy_indexed => .{ .none = {} },
    };
}

test "client executes typed values without exposing execution internals" {
    const engine = Engine.init(std.testing.allocator, .{});
    const lhs = try engine.fromF32(&.{2}, &.{ 1, 2 });
    defer lhs.deinit();
    const rhs = try engine.fromF32(&.{2}, &.{ 10, 20 });
    defer rhs.deinit();

    const added = try engine.add(lhs, rhs);
    defer added.deinit();
    const result = try engine.relu(added);
    defer result.deinit();

    var bytes: [2 * @sizeOf(f32)]u8 = undefined;
    try engine.copyToHost(result, &bytes);
    var values: [2]f32 = undefined;
    @memcpy(std.mem.asBytes(&values), &bytes);
    try std.testing.expectEqualSlices(f32, &.{ 11, 22 }, &values);
}
