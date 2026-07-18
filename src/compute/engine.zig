const std = @import("std");
const compat = @import("../support/compat.zig");
const tensor = @import("types/tensor/index.zig");
const operation = @import("types/operation/index.zig");
const eager = @import("execution/eager/index.zig");
const semantic = @import("plan/sema/index.zig");
const eager_planning = @import("plan/eager.zig");
const graph_planning = @import("plan/graph.zig");
const graph_execution = @import("execution/graph/index.zig");
const ir = @import("types/ir/index.zig");
const telemetry = @import("telemetry.zig");

pub const Tensor = tensor.Tensor;
pub const ComputeGraph = ir.ComputeGraph;
pub const GraphResult = graph_execution.Result;
pub const EagerResult = eager.ExecutionResult;
pub const DType = tensor.DType;
pub const Device = tensor.Device;
pub const Telemetry = telemetry.Interface;

/// Operations exposed by the language-neutral compute engine.
/// Backend selection, semantic validation, planning, and fusion happen below
/// this boundary.
pub const Operation = enum {
    add,
    sub,
    mul,
    div,
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

    pub const Config = struct {
        telemetry: Telemetry = .{},
    };

    pub fn init(allocator: std.mem.Allocator, config: Config) Engine {
        telemetry.init(config.telemetry);
        return .{ .allocator = allocator };
    }

    pub fn initWithCurrentTelemetry(allocator: std.mem.Allocator) Engine {
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

    pub fn invoke(self: Engine, op: Operation, inputs: []const *Tensor) !*Tensor {
        var result = try self.invokeAll(op, inputs);
        if (result.secondary != null) {
            result.deinit();
            return error.MultiOutputRequiresOutputs;
        }
        const primary = result.primary;
        result.primary = undefined;
        return primary;
    }

    pub fn invokeAll(self: Engine, op: Operation, inputs: []const *Tensor) !Outputs {
        const raw = try operation.Op.init(toTag(op), inputs, defaultOptions(op));
        var result = try self.executeRaw(raw);
        const outputs = Outputs{ .primary = result.primary, .secondary = result.secondary };
        result.primary = undefined;
        result.secondary = null;
        return outputs;
    }

    pub fn executeRaw(self: Engine, raw: operation.Op) !EagerResult {
        var execution_scope = telemetry.beginTrace(.execution, telemetry.traces.run);
        defer execution_scope.end();

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
        return eager.executeAllWithPlan(self.allocator, raw, &plan);
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

    pub fn relu(self: Engine, value: *Tensor) !*Tensor {
        return self.invoke(.relu, &.{value});
    }

    pub fn matmul(self: Engine, lhs: *Tensor, rhs: *Tensor) !*Tensor {
        return self.invoke(.matmul, &.{ lhs, rhs });
    }

    pub fn sum(self: Engine, value: *Tensor) !*Tensor {
        return self.invoke(.sum, &.{value});
    }
};

fn toTag(op: Operation) operation.OpTag {
    return switch (op) {
        .add => .add,
        .sub => .sub,
        .mul => .mul,
        .div => .div,
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
    };
}

fn defaultOptions(op: Operation) operation.OpOptions {
    return switch (op) {
        .abs, .exp, .log, .neg, .sqrt, .sign, .relu, .sigmoid, .silu, .tanh, .gelu, .add, .sub, .mul, .div, .dot, .matmul, .contiguous => .{ .none = {} },
        .sum, .mean, .min, .max, .variance, .std, .argmin, .argmax => .{ .reduce_all = .{} },
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
