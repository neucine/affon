const std = @import("std");
const tensor_value = @import("../../types/tensor/tensor.zig");
const Tensor = tensor_value.Tensor;
const Storage = @import("../../types/tensor/storage.zig").Storage;
const telemetry = @import("../../telemetry.zig");
const Op = @import("../../types/operation/op.zig").Op;
const Device = @import("../../types/tensor/device.zig").Device;
const diagnostic = @import("zig_libs").diagnostic;
const Shape = @import("../../types/tensor/shape.zig").Shape;
const Layout = @import("../../types/tensor/layout.zig").Layout;
const TensorSpec = @import("../../types/tensor/tensor_spec.zig").TensorSpec;
const prepared_execution = @import("../prepared.zig");
const reduction_execution = @import("../dispatch/reduction.zig");
const materialization_execution = @import("../materialization.zig");
const transfer_execution = @import("../transfer.zig");
const execution_metrics = @import("../metrics.zig");
const execution_layout = @import("../../plan/layout.zig");
const eager_plan = @import("../../types/ir/plan.zig");
const conversion_execution = @import("../dispatch/conversion.zig");
const elementwise_execution = @import("../dispatch/elementwise.zig");
const linalg_execution = @import("../dispatch/linalg.zig");
const normalization_execution = @import("../dispatch/normalization.zig");
const selection_execution = @import("../dispatch/selection.zig");
const shape_execution = @import("../dispatch/shape.zig");

fn metricAdd(definition: telemetry.MetricDefinition, delta: i64) void {
    telemetry.add(definition, delta);
}

fn metricSink() execution_metrics.Sink {
    return .{};
}

fn recordTransferSummary(summary: transfer_execution.TransferSummary) void {
    execution_metrics.recordTransferSummary(metricSink(), summary);
}

fn recordMaterializationSummary(summary: materialization_execution.MaterializationSummary) void {
    execution_metrics.recordMaterializationSummary(metricSink(), summary);
}

pub const ExecutionInputRequirement = eager_plan.ExecutionInputRequirement;
pub const EagerOpPlan = eager_plan.EagerPlan;

const StepTraceLabel = struct {
    group: telemetry.Group,
    name: []const u8,
};

fn attrString(key: []const u8, value: []const u8) telemetry.Attribute {
    return .{ .key = key, .value = .{ .string = value } };
}

fn attrInt(key: []const u8, value: usize) telemetry.Attribute {
    return .{ .key = key, .value = .{ .integer = @intCast(value) } };
}

pub const ExecutionResult = struct {
    primary: *Tensor,
    secondary: ?*Tensor = null,

    pub fn deinit(self: *ExecutionResult) void {
        if (self.secondary) |secondary| secondary.deinit();
        self.primary.deinit();
        self.* = undefined;
    }
};

pub fn executeAllWithPlan(allocator: std.mem.Allocator, op: Op, plan: *const EagerOpPlan) !ExecutionResult {
    try validateInputsForPlan(op, plan);
    return executePlan(allocator, op, plan);
}

fn executePlan(allocator: std.mem.Allocator, op: Op, plan: *const EagerOpPlan) !ExecutionResult {
    return switch (plan.kind) {
        .elementwise_binary, .elementwise_unary, .elementwise_generic, .reduction_all, .reduction, .index => executeAllocatedPlan(allocator, op, plan),
        .view => executeViewPlan(allocator, op, plan),
    };
}

fn executeAllocatedPlan(allocator: std.mem.Allocator, op: Op, plan: *const EagerOpPlan) !ExecutionResult {
    const primary = try prepared_execution.createOutputValue(allocator, eagerOutputSpec(plan.primary_output, plan.device), .eager);
    errdefer primary.deinit();

    var secondary: ?*Tensor = null;
    errdefer if (secondary) |s| s.deinit();
    if (plan.secondary_output) |secondary_spec| {
        secondary = try prepared_execution.createOutputValue(allocator, eagerOutputSpec(secondary_spec, plan.device), .eager);
    }

    var dispatch_scope = telemetry.beginTrace(.execution, telemetry.traces.run);
    var dispatch_succeeded = false;
    defer if (!dispatch_succeeded) dispatch_scope.endError();
    const step_trace = classifyStepTrace(op, plan);
    var step_name_buffer: [128]u8 = undefined;
    const step_name = std.fmt.bufPrint(&step_name_buffer, "{s}/{s}", .{ telemetry.groupName(step_trace.group), step_trace.name }) catch step_trace.name;
    var step_scope = telemetry.startSpan(.root, step_name, .internal, &.{
        attrString("op", @tagName(op.tag)),
        attrString("device", @tagName(plan.device)),
        attrString("dtype", @tagName(plan.primary_output.dtype)),
        attrString("input_requirement", @tagName(plan.input_requirement)),
        attrString("input_layout_decision", @tagName(plan.input_layout_decision)),
        attrInt("input_count", op.inputs.len),
        attrInt("output_bytes", plan.primary_output.bytes),
    });
    var step_succeeded = false;
    defer if (!step_succeeded) step_scope.endError();
    if (plan.input_layout_decision == .pack_to_dense) {
        step_scope.addEventNow("materialization_required", &.{
            attrString("reason", "pack_to_dense"),
            attrString("op", @tagName(op.tag)),
            attrString("device", @tagName(plan.device)),
        });
    }

    switch (plan.kind) {
        .elementwise_binary => try dispatchElementwiseBinary(allocator, op, plan, primary.storage.?),
        .elementwise_unary => try dispatchElementwiseUnary(allocator, op, plan, primary.storage.?),
        .elementwise_generic => try dispatchElementwiseGeneric(allocator, op, plan, primary),
        .reduction_all => try dispatchReductionAllKind(allocator, op, plan, primary.storage.?),
        .reduction => try dispatchReductionKind(allocator, op, plan, primary),
        .index => try dispatchIndexKind(op, plan, primary.storage.?, secondary),
        .view => return error.InvalidExecutionPlan,
    }

    const result = ExecutionResult{
        .primary = primary,
        .secondary = secondary,
    };
    step_succeeded = true;
    step_scope.end();
    dispatch_succeeded = true;
    dispatch_scope.end();
    return result;
}

fn classifyStepTrace(op: Op, plan: *const EagerOpPlan) StepTraceLabel {
    if (op.tag != .matmul or op.inputs.len != 2) {
        return .{ .group = .execution, .name = @tagName(op.tag) };
    }
    const matmul = plan.matmul orelse return .{ .group = .execution, .name = @tagName(op.tag) };
    return .{
        .group = .execution,
        .name = matmulFamilyName(matmul.family),
    };
}

fn matmulFamilyName(family: eager_plan.MatmulFamily) []const u8 {
    return switch (family) {
        .gemm_2d => "gemm_2d",
        .gemm_batched => "gemm_batched",
        .gemm_projection => "gemm_projection",
        .gemm_attention_scores => "gemm_attention_scores",
        .gemm_attention_values => "gemm_attention_values",
        .gemm_generic_unresolved => "gemm_generic_unresolved",
    };
}

fn executeViewPlan(allocator: std.mem.Allocator, op: Op, plan: *const EagerOpPlan) !ExecutionResult {
    if (op.inputs.len != 1) return error.InvalidInputCount;
    if (plan.secondary_output != null) return error.InvalidExecutionPlan;
    const input = op.inputs[0];
    const primary = try prepared_execution.createViewValue(allocator, input, eagerOutputSpec(plan.primary_output, plan.device));
    return .{
        .primary = primary,
        .secondary = null,
    };
}

fn eagerOutputSpec(output: EagerOpPlan.Output, device: Device) TensorSpec {
    return .{
        .shape = output.shape,
        .dtype = output.dtype,
        .layout = output.layout,
        .device = device,
        .axes = output.axes,
    };
}

fn dispatchElementwiseBinary(allocator: std.mem.Allocator, op: Op, plan: *const EagerOpPlan, output: *Storage) !void {
    switch (op.tag) {
        .add, .sub, .mul, .div => try dispatchBinary(allocator, op, plan, output),
        .eq, .lt, .gt => try dispatchCompare(allocator, op, plan, output),
        else => return error.InvalidExecutionPlan,
    }
}

fn dispatchElementwiseUnary(allocator: std.mem.Allocator, op: Op, plan: *const EagerOpPlan, output: *Storage) !void {
    switch (op.tag) {
        .abs, .exp, .gelu, .gelu_grad, .log, .neg, .sqrt, .sign, .relu, .sigmoid, .silu, .tanh => try dispatchUnary(allocator, op, plan, output),
        .clamp => try dispatchClamp(allocator, op, plan, output),
        else => return error.InvalidExecutionPlan,
    }
}

fn dispatchElementwiseGeneric(allocator: std.mem.Allocator, op: Op, plan: *const EagerOpPlan, output: *Tensor) !void {
    switch (op.tag) {
        .where => try dispatchWhere(allocator, op, plan, output.storage orelse return error.InvalidExecutionPlan),
        .masked_fill => try dispatchMaskedFill(allocator, op, plan, output.storage orelse return error.InvalidExecutionPlan),
        .cast => try dispatchCast(op, plan, output.storage orelse return error.InvalidExecutionPlan),
        .cross_entropy_indexed_backward => try dispatchCrossEntropyIndexedBackward(op, plan, output.storage orelse return error.InvalidExecutionPlan),
        .layer_norm => try dispatchLayerNorm(op, plan, output.storage orelse return error.InvalidExecutionPlan),
        .rms_norm => try dispatchRmsNorm(op, plan, output.storage orelse return error.InvalidExecutionPlan),
        .contiguous => try dispatchContiguous(op, output),
        .cat => try dispatchCat(allocator, op, plan, output.storage orelse return error.InvalidExecutionPlan),
        .stack => try dispatchStack(allocator, op, plan, output.storage orelse return error.InvalidExecutionPlan),
        .slice => try dispatchSlice(op, plan, output.storage orelse return error.InvalidExecutionPlan),
        else => return error.InvalidExecutionPlan,
    }
}

fn dispatchReductionAllKind(_: std.mem.Allocator, op: Op, plan: *const EagerOpPlan, output: *Storage) !void {
    switch (op.tag) {
        .sum_all, .mean_all, .min_all, .max_all, .variance_all, .std_all, .argmin_all, .argmax_all => try dispatchReductionAll(op, plan, output),
        .cross_entropy_indexed => try dispatchCrossEntropyIndexed(op, plan, output),
        .log_softmax_nll, .cross_entropy => try dispatchLogSoftmaxNll(op, plan, output),
        else => return error.InvalidExecutionPlan,
    }
}

fn dispatchReductionKind(allocator: std.mem.Allocator, op: Op, plan: *const EagerOpPlan, output: *Tensor) !void {
    switch (op.tag) {
        .sum_axis, .mean_axis, .min_axis, .max_axis, .variance_axis, .std_axis, .argmin_axis, .argmax_axis => try dispatchReductionAxis(op, plan, output.storage orelse return error.InvalidExecutionPlan),
        .reduce_to_shape => try dispatchReduceToShape(allocator, op, plan, output),
        .softmax => try dispatchSoftmax(op, plan, output.storage orelse return error.InvalidExecutionPlan),
        .log_softmax => try dispatchLogSoftmax(allocator, op, plan, output.storage orelse return error.InvalidExecutionPlan),
        .dot => try dispatchDot(op, plan, output.storage orelse return error.InvalidExecutionPlan),
        .matmul => try dispatchMatmul(op, plan, output.storage orelse return error.InvalidExecutionPlan),
        else => return error.InvalidExecutionPlan,
    }
}

fn dispatchIndexKind(op: Op, plan: *const EagerOpPlan, output: *Storage, secondary: ?*Tensor) !void {
    switch (op.tag) {
        .gather => try dispatchGather(op, plan, output),
        .embedding => try dispatchEmbedding(op, plan, output),
        .index_select => try dispatchIndexSelect(op, plan, output),
        .scatter_add => try dispatchScatterAdd(op, plan, output),
        .one_hot => try dispatchOneHot(op, plan, output),
        .topk => {
            const secondary_value = secondary orelse return error.InvalidExecutionPlan;
            try dispatchTopK(op, plan, output, secondary_value.storage orelse return error.InvalidExecutionPlan);
        },
        else => return error.InvalidExecutionPlan,
    }
}

fn dispatchReduceToShape(allocator: std.mem.Allocator, op: Op, plan: *const EagerOpPlan, output: *Tensor) !void {
    var prepared = try prepareInputValueForDecision(op.inputs[0], inputLayoutDecisionForPlan(plan));
    defer prepared.deinit();
    const reduce = plan.reduce_to_shape orelse return error.InvalidExecutionPlan;
    recordTransferSummary(try reduction_execution.reduceToShape(allocator, prepared.value, reduce, output, .eager));
}

fn dispatchScatterAdd(op: Op, plan: *const EagerOpPlan, output: *Storage) !void {
    var base = try prepareInputValueForDecision(op.inputs[0], inputLayoutDecisionForPlan(plan));
    defer base.deinit();
    var index = try prepareInputValueForDecision(op.inputs[1], inputLayoutDecisionForPlan(plan));
    defer index.deinit();
    var updates = try prepareInputValueForDecision(op.inputs[2], inputLayoutDecisionForPlan(plan));
    defer updates.deinit();
    const axis = switch (op.options) {
        .scatter_add => |s| s.axis,
        else => return error.InvalidOpOptions,
    };
    try selection_execution.dispatchScatterAdd(
        base.value.device() orelse return error.InputNotMaterialized,
        op.inputs[0].dtype,
        op.inputs[1].dtype,
        base.value,
        index.value,
        updates.value,
        output,
        axis,
    );
}

fn validateInputsForPlan(op: Op, plan: *const EagerOpPlan) !void {
    if (op.inputs.len != plan.inputs.len) {
        return diagnostic.withError(
            error.InputCountMismatch,
            "eager.plan: input count mismatch (op={d}, plan={d})",
            .{ op.inputs.len, plan.inputs.len },
        );
    }
    switch (inputLayoutDecisionForPlan(plan)) {
        .accept, .pack_to_dense => {},
        .unsupported => return diagnostic.withError(
            error.ExecutionNotImplemented,
            "eager.plan: input layout unsupported by kernel capability",
            .{},
        ),
    }

    for (op.inputs, 0..) |input, i| {
        const spec = plan.inputs[i];
        const device = input.device() orelse {
            return diagnostic.withError(
                error.InputNotMaterialized,
                "eager.plan: input[{d}] not materialized",
                .{i},
            );
        };
        if (device != plan.device or device != spec.device) {
            return diagnostic.withError(
                error.DeviceMismatch,
                "eager.plan: input[{d}] device mismatch (input={s}, plan={s}, spec={s})",
                .{ i, @tagName(device), @tagName(plan.device), @tagName(spec.device) },
            );
        }
        if (input.dtype != spec.dtype) {
            return diagnostic.withError(
                error.DTypeMismatch,
                "eager.plan: input[{d}] dtype mismatch (input={s}, spec={s})",
                .{ i, @tagName(input.dtype), @tagName(spec.dtype) },
            );
        }
        if (!@import("../../types/tensor/shape.zig").Shape.eql(input.shape, spec.shape)) {
            return diagnostic.withError(
                error.ShapeMismatch,
                "eager.plan: input[{d}] shape mismatch",
                .{i},
            );
        }
        if (plan.input_requirement == .require_contiguous_input and !input.layout.isContiguous(input.shape)) {
            return diagnostic.withError(
                error.InputNotContiguous,
                "eager.plan: input[{d}] must be contiguous",
                .{i},
            );
        }
        if (plan.input_requirement == .require_storage and input.storage == null) {
            return diagnostic.withError(
                error.InputNotMaterialized,
                "eager.plan: input[{d}] requires storage",
                .{i},
            );
        }
        if (plan.input_requirement == .require_contiguous_input) metricAdd(telemetry.metrics.execution.contiguous_input_required_count, 1);
        if (plan.input_requirement == .require_storage) metricAdd(telemetry.metrics.execution.storage_input_required_count, 1);
    }
}

fn dispatchBinary(_: std.mem.Allocator, op: Op, plan: *const EagerOpPlan, output: *Storage) !void {
    var lhs = try prepareInputValueForDecision(op.inputs[0], inputLayoutDecisionForPlan(plan));
    defer lhs.deinit();
    var rhs = try prepareInputValueForDecision(op.inputs[1], inputLayoutDecisionForPlan(plan));
    defer rhs.deinit();

    try elementwise_execution.dispatchBinary(
        lhs.value.device() orelse return error.InputNotMaterialized,
        op.tag,
        lhs.value.dtype,
        lhs.value,
        rhs.value,
        output,
        try prepared_execution.binaryElementwiseDescriptor(lhs.value, rhs.value, plan.broadcast),
    );
}

fn dispatchUnary(_: std.mem.Allocator, op: Op, plan: *const EagerOpPlan, output: *Storage) !void {
    var prepared = try prepareInputValueForDecision(op.inputs[0], inputLayoutDecisionForPlan(plan));
    defer prepared.deinit();
    try elementwise_execution.dispatchUnary(
        prepared.value.device() orelse return error.InputNotMaterialized,
        op.tag,
        prepared.value.dtype,
        prepared.value,
        output,
    );
}

fn dispatchCompare(_: std.mem.Allocator, op: Op, plan: *const EagerOpPlan, output: *Storage) !void {
    var lhs = try prepareInputValueForDecision(op.inputs[0], inputLayoutDecisionForPlan(plan));
    defer lhs.deinit();
    var rhs = try prepareInputValueForDecision(op.inputs[1], inputLayoutDecisionForPlan(plan));
    defer rhs.deinit();

    try elementwise_execution.dispatchCompare(
        lhs.value.device() orelse return error.InputNotMaterialized,
        op.tag,
        lhs.value.dtype,
        lhs.value,
        rhs.value,
        output,
        try prepared_execution.binaryElementwiseDescriptor(lhs.value, rhs.value, plan.broadcast),
    );
}

fn dispatchReductionAll(op: Op, plan: *const EagerOpPlan, output: *Storage) !void {
    var prepared = try prepareInputValueForDecision(op.inputs[0], inputLayoutDecisionForPlan(plan));
    defer prepared.deinit();
    try reduction_execution.dispatchAll(
        prepared.value.device() orelse return error.InputNotMaterialized,
        op.tag,
        prepared.value.dtype,
        prepared.value,
        output,
    );
}

fn dispatchReductionAxis(op: Op, plan: *const EagerOpPlan, output: *Storage) !void {
    var prepared = try prepareInputValueForDecision(op.inputs[0], inputLayoutDecisionForPlan(plan));
    defer prepared.deinit();
    const options = switch (op.options) {
        .reduce_axis => |r| r,
        else => return error.InvalidOpOptions,
    };
    try reduction_execution.dispatchAxis(
        prepared.value.device() orelse return error.InputNotMaterialized,
        op.tag,
        prepared.value.dtype,
        prepared.value,
        output,
        options.axis,
        options.keepdim,
    );
}

fn dispatchSoftmax(op: Op, plan: *const EagerOpPlan, output: *Storage) !void {
    var prepared = try prepareInputValueForDecision(op.inputs[0], inputLayoutDecisionForPlan(plan));
    defer prepared.deinit();
    const axis = switch (op.options) {
        .softmax => |s| s.axis,
        else => return error.InvalidOpOptions,
    };
    try reduction_execution.dispatchSoftmax(
        prepared.value.device() orelse return error.InputNotMaterialized,
        prepared.value.dtype,
        prepared.value,
        output,
        axis,
    );
}

fn dispatchLogSoftmax(allocator: std.mem.Allocator, op: Op, plan: *const EagerOpPlan, output: *Storage) !void {
    var prepared = try prepareInputValueForDecision(op.inputs[0], inputLayoutDecisionForPlan(plan));
    defer prepared.deinit();
    const axis = switch (op.options) {
        .log_softmax => |s| s.axis,
        else => return error.InvalidOpOptions,
    };
    try reduction_execution.dispatchLogSoftmax(
        allocator,
        prepared.value.device() orelse return error.InputNotMaterialized,
        prepared.value.dtype,
        prepared.value,
        output,
        axis,
    );
}

fn dispatchLogSoftmaxNll(op: Op, plan: *const EagerOpPlan, output: *Storage) !void {
    var logits = try prepareInputValueForDecision(op.inputs[0], inputLayoutDecisionForPlan(plan));
    defer logits.deinit();
    var targets = try prepareInputValueForDecision(op.inputs[1], inputLayoutDecisionForPlan(plan));
    defer targets.deinit();
    const axis = switch (op.options) {
        .none => @as(usize, 1),
        .log_softmax_nll => |v| v.axis,
        .cross_entropy => |v| v.axis,
        else => return error.InvalidOpOptions,
    };
    try reduction_execution.dispatchLogSoftmaxNll(
        logits.value.device() orelse return error.InputNotMaterialized,
        logits.value.dtype,
        logits.value,
        targets.value,
        output,
        axis,
    );
}

fn dispatchCrossEntropyIndexed(op: Op, plan: *const EagerOpPlan, output: *Storage) !void {
    var logits = try prepareInputValueForDecision(op.inputs[0], inputLayoutDecisionForPlan(plan));
    defer logits.deinit();
    var targets = try prepareInputValueForDecision(op.inputs[1], inputLayoutDecisionForPlan(plan));
    defer targets.deinit();
    const axis = switch (op.options) {
        .none => @as(usize, 1),
        .cross_entropy_indexed => |v| v.axis,
        else => return error.InvalidOpOptions,
    };
    if (axis != 1 or op.inputs[0].shape.rank() != 2) return error.InvalidAxis;
    try reduction_execution.dispatchCrossEntropyIndexed(
        logits.value.device() orelse return error.InputNotMaterialized,
        logits.value.dtype,
        logits.value,
        targets.value,
        output,
    );
}

fn dispatchCrossEntropyIndexedBackward(op: Op, plan: *const EagerOpPlan, output: *Storage) !void {
    var logits = try prepareInputValueForDecision(op.inputs[0], inputLayoutDecisionForPlan(plan));
    defer logits.deinit();
    var targets = try prepareInputValueForDecision(op.inputs[1], inputLayoutDecisionForPlan(plan));
    defer targets.deinit();
    var grad_out = try prepareInputValueForDecision(op.inputs[2], inputLayoutDecisionForPlan(plan));
    defer grad_out.deinit();
    const axis = switch (op.options) {
        .none => @as(usize, 1),
        .cross_entropy_indexed_backward => |v| v.axis,
        else => return error.InvalidOpOptions,
    };
    if (axis != 1 or op.inputs[0].shape.rank() != 2) return error.InvalidAxis;
    try reduction_execution.dispatchCrossEntropyIndexedBackward(
        logits.value.device() orelse return error.InputNotMaterialized,
        logits.value.dtype,
        logits.value,
        targets.value,
        grad_out.value,
        output,
    );
}

fn dispatchWhere(_: std.mem.Allocator, op: Op, plan: *const EagerOpPlan, output: *Storage) !void {
    var cond = try prepareInputValueForDecision(op.inputs[0], inputLayoutDecisionForPlan(plan));
    defer cond.deinit();
    var on_true = try prepareInputValueForDecision(op.inputs[1], inputLayoutDecisionForPlan(plan));
    defer on_true.deinit();
    var on_false = try prepareInputValueForDecision(op.inputs[2], inputLayoutDecisionForPlan(plan));
    defer on_false.deinit();

    try elementwise_execution.dispatchWhere(
        cond.value.device() orelse return error.InputNotMaterialized,
        cond.value.dtype,
        on_true.value.dtype,
        cond.value,
        on_true.value,
        on_false.value,
        output,
        try prepared_execution.whereDescriptor(cond.value, on_true.value, on_false.value, plan.broadcast),
    );
}

fn dispatchMaskedFill(_: std.mem.Allocator, op: Op, plan: *const EagerOpPlan, output: *Storage) !void {
    var input = try prepareInputValueForDecision(op.inputs[0], inputLayoutDecisionForPlan(plan));
    defer input.deinit();
    var mask = try prepareInputValueForDecision(op.inputs[1], inputLayoutDecisionForPlan(plan));
    defer mask.deinit();
    const fill = switch (op.options) {
        .masked_fill => |m| m,
        else => return error.InvalidOpOptions,
    };
    try elementwise_execution.dispatchMaskedFill(
        input.value.device() orelse return error.InputNotMaterialized,
        input.value.dtype,
        mask.value.dtype,
        input.value,
        mask.value,
        output,
        fill.value,
        try prepared_execution.maskedFillDescriptor(input.value, mask.value, plan.broadcast),
    );
}

fn dispatchCast(op: Op, plan: *const EagerOpPlan, output: *Storage) !void {
    var input = try prepareInputValueForDecision(op.inputs[0], inputLayoutDecisionForPlan(plan));
    defer input.deinit();
    const cast = switch (op.options) {
        .cast => |c| c,
        else => return error.InvalidOpOptions,
    };
    try conversion_execution.dispatchCast(
        input.value.device() orelse return error.InputNotMaterialized,
        op.inputs[0].dtype,
        cast.to,
        input.value,
        output,
    );
}

fn dispatchLayerNorm(op: Op, plan: *const EagerOpPlan, output: *Storage) !void {
    var input = try prepareInputValueForDecision(op.inputs[0], inputLayoutDecisionForPlan(plan));
    defer input.deinit();
    const ln = switch (op.options) {
        .layer_norm => |v| v,
        else => return error.InvalidOpOptions,
    };
    try normalization_execution.dispatchLayerNorm(
        input.value.device() orelse return error.InputNotMaterialized,
        op.inputs[0].dtype,
        input.value,
        output,
        ln.axis,
        ln.eps,
    );
}

fn dispatchRmsNorm(op: Op, plan: *const EagerOpPlan, output: *Storage) !void {
    var input = try prepareInputValueForDecision(op.inputs[0], inputLayoutDecisionForPlan(plan));
    defer input.deinit();
    const rn = switch (op.options) {
        .rms_norm => |v| v,
        else => return error.InvalidOpOptions,
    };
    try normalization_execution.dispatchRmsNorm(
        input.value.device() orelse return error.InputNotMaterialized,
        op.inputs[0].dtype,
        input.value,
        output,
        rn.axis,
        rn.eps,
    );
}
fn dispatchContiguous(op: Op, output: *Tensor) !void {
    recordMaterializationSummary(try materialization_execution.contiguousInto(op.inputs[0], output));
}

fn dispatchDot(op: Op, plan: *const EagerOpPlan, output: *Storage) !void {
    var lhs = try prepareInputValueForDecision(op.inputs[0], inputLayoutDecisionForPlan(plan));
    defer lhs.deinit();
    var rhs = try prepareInputValueForDecision(op.inputs[1], inputLayoutDecisionForPlan(plan));
    defer rhs.deinit();
    try linalg_execution.dispatchDot(
        lhs.value.device() orelse return error.InputNotMaterialized,
        lhs.value.dtype,
        lhs.value,
        rhs.value,
        output,
    );
}

fn dispatchMatmul(op: Op, plan: *const EagerOpPlan, output: *Storage) !void {
    var lhs = try prepareMatmulOperand(op.inputs[0], inputLayoutDecisionForPlan(plan));
    defer lhs.deinit();
    var rhs = try prepareMatmulOperand(op.inputs[1], inputLayoutDecisionForPlan(plan));
    defer rhs.deinit();
    const device = lhs.value.device() orelse return error.InputNotMaterialized;
    if (try linalg_execution.dispatchProjectionMatmulFastPath(
        device,
        lhs.value.dtype,
        lhs.value,
        rhs.value,
        plan.matmul != null and plan.matmul.?.projection,
        output,
    )) return;
    try linalg_execution.dispatchMatmulWithLayouts(
        device,
        lhs.value.dtype,
        lhs.value,
        rhs.value,
        output,
    );
}

fn inputLayoutDecisionForPlan(plan: *const EagerOpPlan) execution_layout.InputLayoutDecision {
    return plan.input_layout_decision;
}

fn dispatchGather(op: Op, plan: *const EagerOpPlan, output: *Storage) !void {
    var input = try prepareInputValueForDecision(op.inputs[0], inputLayoutDecisionForPlan(plan));
    defer input.deinit();
    var index = try prepareInputValueForDecision(op.inputs[1], inputLayoutDecisionForPlan(plan));
    defer index.deinit();
    const gather = switch (op.options) {
        .gather => |g| g,
        else => return error.InvalidOpOptions,
    };
    try selection_execution.dispatchGather(
        input.value.device() orelse return error.InputNotMaterialized,
        op.inputs[0].dtype,
        input.value,
        index.value,
        output,
        gather.axis,
    );
}

fn dispatchIndexSelect(op: Op, plan: *const EagerOpPlan, output: *Storage) !void {
    var input = try prepareInputValueForDecision(op.inputs[0], inputLayoutDecisionForPlan(plan));
    defer input.deinit();
    var index = try prepareInputValueForDecision(op.inputs[1], inputLayoutDecisionForPlan(plan));
    defer index.deinit();
    const index_select = switch (op.options) {
        .index_select => |s| s,
        else => return error.InvalidOpOptions,
    };
    try selection_execution.dispatchIndexSelect(
        input.value.device() orelse return error.InputNotMaterialized,
        op.inputs[0].dtype,
        input.value,
        index.value,
        output,
        index_select.axis,
    );
}

fn dispatchEmbedding(op: Op, plan: *const EagerOpPlan, output: *Storage) !void {
    var table = try prepareInputValueForDecision(op.inputs[0], inputLayoutDecisionForPlan(plan));
    defer table.deinit();
    var indices = try prepareInputValueForDecision(op.inputs[1], inputLayoutDecisionForPlan(plan));
    defer indices.deinit();
    try selection_execution.dispatchEmbedding(
        table.value.device() orelse return error.InputNotMaterialized,
        op.inputs[0].dtype,
        table.value,
        indices.value,
        output,
    );
}

fn dispatchOneHot(op: Op, plan: *const EagerOpPlan, output: *Storage) !void {
    var index = try prepareInputValueForDecision(op.inputs[0], inputLayoutDecisionForPlan(plan));
    defer index.deinit();
    const one_hot = switch (op.options) {
        .one_hot => |v| v,
        else => return error.InvalidOpOptions,
    };
    try selection_execution.dispatchOneHot(
        index.value.device() orelse return error.InputNotMaterialized,
        index.value,
        output,
        one_hot.num_classes,
    );
}

fn dispatchTopK(op: Op, plan: *const EagerOpPlan, output: *Storage, indices_output: *Storage) !void {
    var input = try prepareInputValueForDecision(op.inputs[0], inputLayoutDecisionForPlan(plan));
    defer input.deinit();
    const topk = switch (op.options) {
        .topk => |v| v,
        else => return error.InvalidOpOptions,
    };
    try selection_execution.dispatchTopK(
        op.inputs[0].allocator,
        input.value.device() orelse return error.InputNotMaterialized,
        op.inputs[0].dtype,
        input.value,
        output,
        indices_output,
        topk.axis,
        topk.k,
        topk.largest,
    );
}

fn dispatchCat(allocator: std.mem.Allocator, op: Op, plan: *const EagerOpPlan, output: *Storage) !void {
    const axis = switch (op.options) {
        .none => @as(usize, 0),
        .concat => |c| c.axis,
        else => return error.InvalidOpOptions,
    };
    const prepared_inputs = try allocator.alloc(PreparedInputValue, op.inputs.len);
    defer allocator.free(prepared_inputs);
    for (prepared_inputs, op.inputs) |*prepared, input| {
        prepared.* = try prepareInputValueForDecision(input, inputLayoutDecisionForPlan(plan));
    }
    defer for (prepared_inputs) |prepared| prepared.deinit();
    try shape_execution.dispatchCat(
        allocator,
        prepared_inputs[0].value.device() orelse return error.InputNotMaterialized,
        op.inputs[0].dtype,
        prepared_inputs,
        output,
        axis,
    );
}

fn dispatchStack(allocator: std.mem.Allocator, op: Op, plan: *const EagerOpPlan, output: *Storage) !void {
    const axis = switch (op.options) {
        .none => @as(usize, 0),
        .stack => |s| s.axis,
        else => return error.InvalidOpOptions,
    };
    const prepared_inputs = try allocator.alloc(PreparedInputValue, op.inputs.len);
    defer allocator.free(prepared_inputs);
    for (prepared_inputs, op.inputs) |*prepared, input| {
        prepared.* = try prepareInputValueForDecision(input, inputLayoutDecisionForPlan(plan));
    }
    defer for (prepared_inputs) |prepared| prepared.deinit();
    try shape_execution.dispatchStack(
        allocator,
        prepared_inputs[0].value.device() orelse return error.InputNotMaterialized,
        op.inputs[0].dtype,
        prepared_inputs,
        output,
        axis,
    );
}

fn dispatchSlice(op: Op, plan: *const EagerOpPlan, output: *Storage) !void {
    var input = try prepareInputValueForDecision(op.inputs[0], inputLayoutDecisionForPlan(plan));
    defer input.deinit();
    const ranges = switch (op.options) {
        .slice => |s| s.ranges,
        else => return error.InvalidOpOptions,
    };
    try shape_execution.dispatchSlice(
        input.value.device() orelse return error.InputNotMaterialized,
        op.inputs[0].dtype,
        input.value,
        output,
        ranges,
    );
}

fn dispatchClamp(_: std.mem.Allocator, op: Op, plan: *const EagerOpPlan, output: *Storage) !void {
    var prepared = try prepareInputValueForDecision(op.inputs[0], inputLayoutDecisionForPlan(plan));
    defer prepared.deinit();
    const clamp = switch (op.options) {
        .clamp => |c| c,
        else => return error.InvalidOpOptions,
    };
    try elementwise_execution.dispatchClamp(
        prepared.value.device() orelse return error.InputNotMaterialized,
        prepared.value.dtype,
        prepared.value,
        output,
        clamp.min,
        clamp.max,
    );
}

const PreparedInputValue = prepared_execution.PreparedInputValue;

fn prepareInputValueForDecision(value: *const Tensor, decision: execution_layout.InputLayoutDecision) !PreparedInputValue {
    return execution_metrics.prepareInputValue(metricSink(), value.allocator, value, decision, .eager);
}

const PreparedMatmulOperand = PreparedInputValue;

fn prepareMatmulOperand(value: *const Tensor, decision: execution_layout.InputLayoutDecision) !PreparedMatmulOperand {
    return prepareInputValueForDecision(value, decision);
}

fn makeViewForTest(
    allocator: std.mem.Allocator,
    base: *const Tensor,
    dims: []const usize,
    strides: []const isize,
    offset: usize,
) !*Tensor {
    const storage = base.storage orelse return error.InputNotMaterialized;
    storage.retain();
    errdefer storage.release();

    var shape = try Shape.initCopy(allocator, dims);
    errdefer shape.deinit();
    var layout = try Layout.initCopy(allocator, strides, offset);
    errdefer layout.deinit();

    const view = try allocator.create(Tensor);
    errdefer allocator.destroy(view);
    view.* = .{
        .allocator = allocator,
        .shape = shape,
        .dtype = base.dtype,
        .layout = layout,
        .storage = storage,
        .axes = try tensor_value.cloneAxes(allocator, base.axes),
    };
    return view;
}

test "eager add runs end to end" {
    const allocator = std.testing.allocator;
    const a = try Tensor.fromSliceF64(allocator, &.{3}, &.{ 1.0, 2.0, 3.0 });
    defer a.deinit();
    const b = try Tensor.fromSliceF64(allocator, &.{3}, &.{ 10.0, 20.0, 30.0 });
    defer b.deinit();

    const op = try Op.init(.add, &.{ a, b }, .{ .binary = .{} });
    const out = try @import("test_support.zig").execute(allocator, op);
    defer out.deinit();

    const bytes = try out.storage.?.readableBytes();
    const values = std.mem.bytesAsSlice(f64, bytes);
    try std.testing.expectEqual(@as(f64, 11.0), values[0]);
    try std.testing.expectEqual(@as(f64, 22.0), values[1]);
    try std.testing.expectEqual(@as(f64, 33.0), values[2]);
}

test "eager add plan requires storage" {
    const allocator = std.testing.allocator;
    const a = try Tensor.fromSliceF32(allocator, &.{2}, &.{ 1, 2 });
    defer a.deinit();
    const b = try Tensor.fromSliceF32(allocator, &.{2}, &.{ 3, 4 });
    defer b.deinit();

    const op = try Op.init(.add, &.{ a, b }, .{ .binary = .{} });
    var plan = try @import("test_support.zig").createPlan(allocator, op);
    defer plan.deinit();

    try std.testing.expectEqual(ExecutionInputRequirement.require_storage, plan.input_requirement);
    try std.testing.expectEqual(execution_layout.InputLayoutDecision.accept, plan.input_layout_decision);
}

test "eager sub runs end to end" {
    const allocator = std.testing.allocator;
    const a = try Tensor.fromSliceF64(allocator, &.{3}, &.{ 10.0, 20.0, 30.0 });
    defer a.deinit();
    const b = try Tensor.fromSliceF64(allocator, &.{3}, &.{ 1.0, 2.0, 3.0 });
    defer b.deinit();

    const op = try Op.init(.sub, &.{ a, b }, .{ .binary = .{} });
    const out = try @import("test_support.zig").execute(allocator, op);
    defer out.deinit();

    const bytes = try out.storage.?.readableBytes();
    const values = std.mem.bytesAsSlice(f64, bytes);
    try std.testing.expectEqual(@as(f64, 9.0), values[0]);
    try std.testing.expectEqual(@as(f64, 18.0), values[1]);
    try std.testing.expectEqual(@as(f64, 27.0), values[2]);
}

test "eager mul runs end to end" {
    const allocator = std.testing.allocator;
    const a = try Tensor.fromSliceF64(allocator, &.{3}, &.{ 1.0, 2.0, 3.0 });
    defer a.deinit();
    const b = try Tensor.fromSliceF64(allocator, &.{3}, &.{ 10.0, 20.0, 30.0 });
    defer b.deinit();

    const op = try Op.init(.mul, &.{ a, b }, .{ .binary = .{} });
    const out = try @import("test_support.zig").execute(allocator, op);
    defer out.deinit();

    const bytes = try out.storage.?.readableBytes();
    const values = std.mem.bytesAsSlice(f64, bytes);
    try std.testing.expectEqual(@as(f64, 10.0), values[0]);
    try std.testing.expectEqual(@as(f64, 40.0), values[1]);
    try std.testing.expectEqual(@as(f64, 90.0), values[2]);
}

test "eager div runs end to end" {
    const allocator = std.testing.allocator;
    const a = try Tensor.fromSliceF64(allocator, &.{3}, &.{ 10.0, 20.0, 30.0 });
    defer a.deinit();
    const b = try Tensor.fromSliceF64(allocator, &.{3}, &.{ 2.0, 4.0, 5.0 });
    defer b.deinit();

    const op = try Op.init(.div, &.{ a, b }, .{ .binary = .{} });
    const out = try @import("test_support.zig").execute(allocator, op);
    defer out.deinit();

    const bytes = try out.storage.?.readableBytes();
    const values = std.mem.bytesAsSlice(f64, bytes);
    try std.testing.expectEqual(@as(f64, 5.0), values[0]);
    try std.testing.expectEqual(@as(f64, 5.0), values[1]);
    try std.testing.expectEqual(@as(f64, 6.0), values[2]);
}

test "eager add right-aligns broadcast dims" {
    const allocator = std.testing.allocator;
    const a = try Tensor.fromSliceF64(allocator, &.{ 2, 3 }, &.{
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
    });
    defer a.deinit();
    const b = try Tensor.fromSliceF64(allocator, &.{3}, &.{ 10.0, 20.0, 30.0 });
    defer b.deinit();

    const op = try Op.init(.add, &.{ a, b }, .{ .binary = .{} });
    const out = try @import("test_support.zig").execute(allocator, op);
    defer out.deinit();

    try std.testing.expectEqualSlices(usize, &.{ 2, 3 }, out.shape.dims);
    const bytes = try out.storage.?.readableBytes();
    const values = std.mem.bytesAsSlice(f64, bytes);
    try std.testing.expectEqualSlices(f64, &.{
        11.0, 22.0, 33.0,
        14.0, 25.0, 36.0,
    }, values);
}

test "eager add packs negative-stride input to dense" {
    const allocator = std.testing.allocator;
    const base = try Tensor.fromSliceF32(allocator, &.{ 3, 2 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer base.deinit();
    const lhs = try makeViewForTest(allocator, base, &.{ 3, 2 }, &.{ -2, 1 }, 4);
    defer lhs.deinit();
    const rhs = try Tensor.fromSliceF32(allocator, &.{ 3, 2 }, &.{ 10, 20, 30, 40, 50, 60 });
    defer rhs.deinit();

    const op = try Op.init(.add, &.{ lhs, rhs }, .{ .binary = .{} });
    var plan = try @import("test_support.zig").createPlan(allocator, op);
    defer plan.deinit();
    try std.testing.expectEqual(execution_layout.InputLayoutDecision.pack_to_dense, plan.input_layout_decision);

    var out = try executeAllWithPlan(allocator, op, &plan);
    defer out.deinit();
    const values = std.mem.bytesAsSlice(f32, try out.primary.storage.?.readableBytes());
    try std.testing.expectEqualSlices(f32, &.{ 15, 26, 33, 44, 51, 62 }, values);
}

test "eager neg runs end to end" {
    const allocator = std.testing.allocator;
    const input = try Tensor.fromSliceF64(allocator, &.{3}, &.{ 1.5, -2.0, 3.25 });
    defer input.deinit();

    const op = try Op.init(.neg, &.{input}, .{ .unary = .{} });
    const out = try @import("test_support.zig").execute(allocator, op);
    defer out.deinit();

    const bytes = try out.storage.?.readableBytes();
    const values = std.mem.bytesAsSlice(f64, bytes);
    try std.testing.expectEqual(@as(f64, -1.5), values[0]);
    try std.testing.expectEqual(@as(f64, 2.0), values[1]);
    try std.testing.expectEqual(@as(f64, -3.25), values[2]);
}

test "eager relu runs end to end" {
    const allocator = std.testing.allocator;
    const input = try Tensor.fromSliceF64(allocator, &.{4}, &.{ -1.0, 0.5, 0.0, 3.25 });
    defer input.deinit();

    const op = try Op.init(.relu, &.{input}, .{ .unary = .{} });
    const out = try @import("test_support.zig").execute(allocator, op);
    defer out.deinit();

    const bytes = try out.storage.?.readableBytes();
    const values = std.mem.bytesAsSlice(f64, bytes);
    try std.testing.expectEqual(@as(f64, 0.0), values[0]);
    try std.testing.expectEqual(@as(f64, 0.5), values[1]);
    try std.testing.expectEqual(@as(f64, 0.0), values[2]);
    try std.testing.expectEqual(@as(f64, 3.25), values[3]);
}

test "eager exp runs end to end" {
    const allocator = std.testing.allocator;
    const input = try Tensor.fromSliceF64(allocator, &.{2}, &.{ 0.0, 1.0 });
    defer input.deinit();

    const op = try Op.init(.exp, &.{input}, .{ .unary = .{} });
    const out = try @import("test_support.zig").execute(allocator, op);
    defer out.deinit();

    const bytes = try out.storage.?.readableBytes();
    const values = std.mem.bytesAsSlice(f64, bytes);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), values[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.718281828459045), values[1], 1e-12);
}

test "eager gelu runs end to end" {
    const allocator = std.testing.allocator;
    const input = try Tensor.fromSliceF64(allocator, &.{1}, &.{1.0});
    defer input.deinit();

    const op = try Op.init(.gelu, &.{input}, .{ .unary = .{} });
    const out = try @import("test_support.zig").execute(allocator, op);
    defer out.deinit();

    const bytes = try out.storage.?.readableBytes();
    const values = std.mem.bytesAsSlice(f64, bytes);
    try std.testing.expectApproxEqAbs(@as(f64, 0.84119199), values[0], 1e-6);
}

test "eager clamp runs end to end" {
    const allocator = std.testing.allocator;
    const input = try Tensor.fromSliceF64(allocator, &.{4}, &.{ -2.0, -0.5, 1.5, 9.0 });
    defer input.deinit();

    const op = try Op.init(.clamp, &.{input}, .{ .clamp = .{ .min = 0.0, .max = 2.0 } });
    const out = try @import("test_support.zig").execute(allocator, op);
    defer out.deinit();

    const bytes = try out.storage.?.readableBytes();
    const values = std.mem.bytesAsSlice(f64, bytes);
    try std.testing.expectEqual(@as(f64, 0.0), values[0]);
    try std.testing.expectEqual(@as(f64, 0.0), values[1]);
    try std.testing.expectEqual(@as(f64, 1.5), values[2]);
    try std.testing.expectEqual(@as(f64, 2.0), values[3]);
}

test "eager sum_all runs end to end" {
    const allocator = std.testing.allocator;
    const input = try Tensor.fromSliceF64(allocator, &.{ 2, 3 }, &.{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 });
    defer input.deinit();

    const op = try Op.init(.sum_all, &.{input}, .{ .reduce_all = .{} });
    const out = try @import("test_support.zig").execute(allocator, op);
    defer out.deinit();

    const bytes = try out.storage.?.readableBytes();
    const values = std.mem.bytesAsSlice(f64, bytes);
    try std.testing.expectEqual(@as(usize, 1), out.shape.numel());
    try std.testing.expectEqual(@as(f64, 21.0), values[0]);
}

test "eager mean_all runs end to end" {
    const allocator = std.testing.allocator;
    const input = try Tensor.fromSliceF64(allocator, &.{4}, &.{ 2.0, 4.0, 6.0, 8.0 });
    defer input.deinit();

    const op = try Op.init(.mean_all, &.{input}, .{ .reduce_all = .{} });
    const out = try @import("test_support.zig").execute(allocator, op);
    defer out.deinit();

    const bytes = try out.storage.?.readableBytes();
    const values = std.mem.bytesAsSlice(f64, bytes);
    try std.testing.expectEqual(@as(f64, 5.0), values[0]);
}

test "eager log_softmax_nll runs end to end" {
    const allocator = std.testing.allocator;
    const logits = try Tensor.fromSliceF32(allocator, &.{ 2, 3 }, &.{ 1.0, 2.0, 3.0, 0.0, 1.0, 0.0 });
    defer logits.deinit();
    const targets = try Tensor.fromSliceF32(allocator, &.{ 2, 3 }, &.{ 0.0, 0.0, 1.0, 0.0, 1.0, 0.0 });
    defer targets.deinit();

    const op = try Op.init(.log_softmax_nll, &.{ logits, targets }, .{ .log_softmax_nll = .{ .axis = 1 } });
    const out = try @import("test_support.zig").execute(allocator, op);
    defer out.deinit();
    const values = std.mem.bytesAsSlice(f32, try out.storage.?.readableBytes());
    try std.testing.expectApproxEqAbs(@as(f32, 0.4795253), values[0], 1e-5);
}

test "eager cross_entropy runs end to end" {
    const allocator = std.testing.allocator;
    const logits = try Tensor.fromSliceF32(allocator, &.{ 2, 3 }, &.{ 1.0, 2.0, 3.0, 0.0, 1.0, 0.0 });
    defer logits.deinit();
    const targets = try Tensor.fromSliceF32(allocator, &.{ 2, 3 }, &.{ 0.0, 0.0, 1.0, 0.0, 1.0, 0.0 });
    defer targets.deinit();

    const op = try Op.init(.cross_entropy, &.{ logits, targets }, .{ .cross_entropy = .{ .axis = 1 } });
    const out = try @import("test_support.zig").execute(allocator, op);
    defer out.deinit();
    const values = std.mem.bytesAsSlice(f32, try out.storage.?.readableBytes());
    try std.testing.expectApproxEqAbs(@as(f32, 0.4795253), values[0], 1e-5);
}

test "eager cross_entropy_indexed runs end to end" {
    const allocator = std.testing.allocator;
    const logits = try Tensor.fromSliceF32(allocator, &.{ 2, 3 }, &.{ 1.0, 2.0, 3.0, 0.0, 1.0, 0.0 });
    defer logits.deinit();
    const targets = try Tensor.fromSliceI64(allocator, &.{2}, &.{ 2, 1 });
    defer targets.deinit();

    const op = try Op.init(.cross_entropy_indexed, &.{ logits, targets }, .{ .cross_entropy_indexed = .{ .axis = 1 } });
    const out = try @import("test_support.zig").execute(allocator, op);
    defer out.deinit();
    const values = std.mem.bytesAsSlice(f32, try out.storage.?.readableBytes());
    try std.testing.expectApproxEqAbs(@as(f32, 0.4795253), values[0], 1e-5);
}

test "eager cross_entropy_indexed_backward runs end to end" {
    const allocator = std.testing.allocator;
    const logits = try Tensor.fromSliceF32(allocator, &.{ 2, 3 }, &.{ 1.0, 2.0, 3.0, 0.0, 1.0, 0.0 });
    defer logits.deinit();
    const targets = try Tensor.fromSliceI64(allocator, &.{2}, &.{ 2, 1 });
    defer targets.deinit();
    const grad_out = try Tensor.fromSliceF32(allocator, &.{1}, &.{1.0});
    defer grad_out.deinit();

    const op = try Op.init(.cross_entropy_indexed_backward, &.{ logits, targets, grad_out }, .{ .cross_entropy_indexed_backward = .{ .axis = 1 } });
    const out = try @import("test_support.zig").execute(allocator, op);
    defer out.deinit();
    const values = std.mem.bytesAsSlice(f32, try out.storage.?.readableBytes());
    try std.testing.expectApproxEqAbs(@as(f32, 0.04501529), values[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.12236424), values[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, -0.16737953), values[2], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.10597078), values[3], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, -0.21194156), values[4], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.10597078), values[5], 1e-5);
}

test "eager log_softmax_nll packs negative-stride input to dense" {
    const allocator = std.testing.allocator;
    const logits_base = try Tensor.fromSliceF32(allocator, &.{ 2, 3 }, &.{ 1.0, 2.0, 3.0, 0.0, 1.0, 0.0 });
    defer logits_base.deinit();
    const logits = try makeViewForTest(allocator, logits_base, &.{ 2, 3 }, &.{ -3, 1 }, 3);
    defer logits.deinit();
    const targets_base = try Tensor.fromSliceF32(allocator, &.{ 2, 3 }, &.{ 0.0, 0.0, 1.0, 0.0, 1.0, 0.0 });
    defer targets_base.deinit();
    const targets = try makeViewForTest(allocator, targets_base, &.{ 2, 3 }, &.{ -3, 1 }, 3);
    defer targets.deinit();

    const op = try Op.init(.log_softmax_nll, &.{ logits, targets }, .{ .log_softmax_nll = .{ .axis = 1 } });
    var plan = try @import("test_support.zig").createPlan(allocator, op);
    defer plan.deinit();
    try std.testing.expectEqual(execution_layout.InputLayoutDecision.pack_to_dense, plan.input_layout_decision);

    var out = try executeAllWithPlan(allocator, op, &plan);
    defer out.deinit();
    const values = std.mem.bytesAsSlice(f32, try out.primary.storage.?.readableBytes());
    try std.testing.expectApproxEqAbs(@as(f32, 0.4795253), values[0], 1e-5);
}

test "eager argmax_all runs end to end" {
    const allocator = std.testing.allocator;
    const input = try Tensor.fromSliceF64(allocator, &.{5}, &.{ 1.0, 9.0, -2.0, 3.0, 2.0 });
    defer input.deinit();

    const op = try Op.init(.argmax_all, &.{input}, .{ .reduce_all = .{} });
    const out = try @import("test_support.zig").execute(allocator, op);
    defer out.deinit();

    const bytes = try out.storage.?.readableBytes();
    const values = std.mem.bytesAsSlice(i64, bytes);
    try std.testing.expectEqual(@as(i64, 1), values[0]);
}

test "eager variance_all runs end to end" {
    const allocator = std.testing.allocator;
    const input = try Tensor.fromSliceF64(allocator, &.{4}, &.{ 1.0, 2.0, 3.0, 4.0 });
    defer input.deinit();

    const op = try Op.init(.variance_all, &.{input}, .{ .reduce_all = .{} });
    const out = try @import("test_support.zig").execute(allocator, op);
    defer out.deinit();

    const bytes = try out.storage.?.readableBytes();
    const values = std.mem.bytesAsSlice(f64, bytes);
    try std.testing.expectApproxEqAbs(@as(f64, 1.25), values[0], 1e-12);
}

test "eager min_all and max_all run end to end" {
    const allocator = std.testing.allocator;
    const input = try Tensor.fromSliceF64(allocator, &.{5}, &.{ 3.0, -2.0, 9.0, 0.5, 1.0 });
    defer input.deinit();

    const min_op = try Op.init(.min_all, &.{input}, .{ .reduce_all = .{} });
    const min_out = try @import("test_support.zig").execute(allocator, min_op);
    defer min_out.deinit();
    const min_bytes = try min_out.storage.?.readableBytes();
    const min_values = std.mem.bytesAsSlice(f64, min_bytes);
    try std.testing.expectEqual(@as(f64, -2.0), min_values[0]);

    const max_op = try Op.init(.max_all, &.{input}, .{ .reduce_all = .{} });
    const max_out = try @import("test_support.zig").execute(allocator, max_op);
    defer max_out.deinit();
    const max_bytes = try max_out.storage.?.readableBytes();
    const max_values = std.mem.bytesAsSlice(f64, max_bytes);
    try std.testing.expectEqual(@as(f64, 9.0), max_values[0]);
}

test "eager argmin_all runs end to end" {
    const allocator = std.testing.allocator;
    const input = try Tensor.fromSliceF64(allocator, &.{5}, &.{ 3.0, -2.0, 9.0, -3.0, 1.0 });
    defer input.deinit();

    const op = try Op.init(.argmin_all, &.{input}, .{ .reduce_all = .{} });
    const out = try @import("test_support.zig").execute(allocator, op);
    defer out.deinit();

    const bytes = try out.storage.?.readableBytes();
    const values = std.mem.bytesAsSlice(i64, bytes);
    try std.testing.expectEqual(@as(i64, 3), values[0]);
}

test "eager std_all runs end to end" {
    const allocator = std.testing.allocator;
    const input = try Tensor.fromSliceF64(allocator, &.{4}, &.{ 1.0, 2.0, 3.0, 4.0 });
    defer input.deinit();

    const op = try Op.init(.std_all, &.{input}, .{ .reduce_all = .{} });
    const out = try @import("test_support.zig").execute(allocator, op);
    defer out.deinit();

    const bytes = try out.storage.?.readableBytes();
    const values = std.mem.bytesAsSlice(f64, bytes);
    try std.testing.expectApproxEqAbs(@as(f64, 1.118033988749895), values[0], 1e-12);
}

test "eager sum_axis runs end to end" {
    const allocator = std.testing.allocator;
    const input = try Tensor.fromSliceF64(allocator, &.{ 2, 3 }, &.{
        1, 2, 3,
        4, 5, 6,
    });
    defer input.deinit();

    const op = try Op.init(.sum_axis, &.{input}, .{ .reduce_axis = .{ .axis = 1, .keepdim = false } });
    const out = try @import("test_support.zig").execute(allocator, op);
    defer out.deinit();

    const bytes = try out.storage.?.readableBytes();
    const values = std.mem.bytesAsSlice(f64, bytes);
    try std.testing.expectEqual(@as(usize, 2), values.len);
    try std.testing.expectEqual(@as(f64, 6.0), values[0]);
    try std.testing.expectEqual(@as(f64, 15.0), values[1]);
}

test "eager sum_axis packs negative-stride input to dense" {
    const allocator = std.testing.allocator;
    const base = try Tensor.fromSliceF32(allocator, &.{ 3, 2 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer base.deinit();
    const input = try makeViewForTest(allocator, base, &.{ 3, 2 }, &.{ -2, 1 }, 4);
    defer input.deinit();

    const op = try Op.init(.sum_axis, &.{input}, .{ .reduce_axis = .{ .axis = 1, .keepdim = false } });
    var plan = try @import("test_support.zig").createPlan(allocator, op);
    defer plan.deinit();
    try std.testing.expectEqual(execution_layout.InputLayoutDecision.pack_to_dense, plan.input_layout_decision);

    var out = try executeAllWithPlan(allocator, op, &plan);
    defer out.deinit();
    const values = std.mem.bytesAsSlice(f32, try out.primary.storage.?.readableBytes());
    try std.testing.expectEqualSlices(f32, &.{ 11, 7, 3 }, values);
}

test "eager mean_axis keepdim runs end to end" {
    const allocator = std.testing.allocator;
    const input = try Tensor.fromSliceF64(allocator, &.{ 2, 3 }, &.{
        1, 2, 3,
        4, 5, 6,
    });
    defer input.deinit();

    const op = try Op.init(.mean_axis, &.{input}, .{ .reduce_axis = .{ .axis = 0, .keepdim = true } });
    const out = try @import("test_support.zig").execute(allocator, op);
    defer out.deinit();

    try std.testing.expectEqual(@as(usize, 2), out.shape.rank());
    try std.testing.expectEqual(@as(usize, 1), out.shape.dims[0]);
    try std.testing.expectEqual(@as(usize, 3), out.shape.dims[1]);
    const bytes = try out.storage.?.readableBytes();
    const values = std.mem.bytesAsSlice(f64, bytes);
    try std.testing.expectEqual(@as(usize, 3), values.len);
    try std.testing.expectEqual(@as(f64, 2.5), values[0]);
    try std.testing.expectEqual(@as(f64, 3.5), values[1]);
    try std.testing.expectEqual(@as(f64, 4.5), values[2]);
}

test "eager argmax_axis runs end to end" {
    const allocator = std.testing.allocator;
    const input = try Tensor.fromSliceF64(allocator, &.{ 2, 3 }, &.{
        1, 8, 3,
        9, 5, 6,
    });
    defer input.deinit();

    const op = try Op.init(.argmax_axis, &.{input}, .{ .reduce_axis = .{ .axis = 1, .keepdim = false } });
    const out = try @import("test_support.zig").execute(allocator, op);
    defer out.deinit();

    const bytes = try out.storage.?.readableBytes();
    const values = std.mem.bytesAsSlice(i64, bytes);
    try std.testing.expectEqual(@as(usize, 2), values.len);
    try std.testing.expectEqual(@as(i64, 1), values[0]);
    try std.testing.expectEqual(@as(i64, 0), values[1]);
}

test "eager where runs end to end" {
    const allocator = std.testing.allocator;
    const cond = try Tensor.fromSliceI64(allocator, &.{4}, &.{ 1, 0, 2, 0 });
    defer cond.deinit();
    const a = try Tensor.fromSliceF64(allocator, &.{4}, &.{ 10.0, 20.0, 30.0, 40.0 });
    defer a.deinit();
    const b = try Tensor.fromSliceF64(allocator, &.{4}, &.{ 1.0, 2.0, 3.0, 4.0 });
    defer b.deinit();

    const op = try Op.init(.where, &.{ cond, a, b }, .{ .none = {} });
    const out = try @import("test_support.zig").execute(allocator, op);
    defer out.deinit();

    const bytes = try out.storage.?.readableBytes();
    const values = std.mem.bytesAsSlice(f64, bytes);
    try std.testing.expectEqual(@as(f64, 10.0), values[0]);
    try std.testing.expectEqual(@as(f64, 2.0), values[1]);
    try std.testing.expectEqual(@as(f64, 30.0), values[2]);
    try std.testing.expectEqual(@as(f64, 4.0), values[3]);
}

test "eager masked_fill runs end to end" {
    const allocator = std.testing.allocator;
    const input = try Tensor.fromSliceF64(allocator, &.{4}, &.{ 1.0, 2.0, 3.0, 4.0 });
    defer input.deinit();
    const mask = try Tensor.fromSliceI64(allocator, &.{4}, &.{ 0, 1, 0, 1 });
    defer mask.deinit();

    const op = try Op.init(.masked_fill, &.{ input, mask }, .{ .masked_fill = .{ .value = -9.0 } });
    const out = try @import("test_support.zig").execute(allocator, op);
    defer out.deinit();

    const bytes = try out.storage.?.readableBytes();
    const values = std.mem.bytesAsSlice(f64, bytes);
    try std.testing.expectEqual(@as(f64, 1.0), values[0]);
    try std.testing.expectEqual(@as(f64, -9.0), values[1]);
    try std.testing.expectEqual(@as(f64, 3.0), values[2]);
    try std.testing.expectEqual(@as(f64, -9.0), values[3]);
}

test "eager where right-aligns broadcast dims" {
    const allocator = std.testing.allocator;
    const cond = try Tensor.fromSliceI64(allocator, &.{ 3, 1 }, &.{ 1, 0, 1 });
    defer cond.deinit();
    const on_true = try Tensor.fromSliceF32(allocator, &.{ 2, 3, 4 }, &.{
        1,  2,  3,  4,
        5,  6,  7,  8,
        9,  10, 11, 12,
        13, 14, 15, 16,
        17, 18, 19, 20,
        21, 22, 23, 24,
    });
    defer on_true.deinit();
    const on_false = try Tensor.fromSliceF32(allocator, &.{4}, &.{ 0, 0, 0, 0 });
    defer on_false.deinit();

    const op = try Op.init(.where, &.{ cond, on_true, on_false }, .{ .none = {} });
    const out = try @import("test_support.zig").execute(allocator, op);
    defer out.deinit();

    try std.testing.expectEqualSlices(usize, &.{ 2, 3, 4 }, out.shape.dims);
    const values = std.mem.bytesAsSlice(f32, try out.storage.?.readableBytes());
    try std.testing.expectEqual(@as(f32, 1), values[0]);
    try std.testing.expectEqual(@as(f32, 0), values[4]);
    try std.testing.expectEqual(@as(f32, 9), values[8]);
    try std.testing.expectEqual(@as(f32, 13), values[12]);
    try std.testing.expectEqual(@as(f32, 0), values[16]);
    try std.testing.expectEqual(@as(f32, 21), values[20]);
}

test "eager masked_fill right-aligns broadcast dims" {
    const allocator = std.testing.allocator;
    const input = try Tensor.fromSliceF32(allocator, &.{ 2, 3, 4 }, &.{
        1,  2,  3,  4,
        5,  6,  7,  8,
        9,  10, 11, 12,
        13, 14, 15, 16,
        17, 18, 19, 20,
        21, 22, 23, 24,
    });
    defer input.deinit();
    const mask = try Tensor.fromSliceI64(allocator, &.{ 3, 1 }, &.{ 1, 0, 1 });
    defer mask.deinit();

    const op = try Op.init(.masked_fill, &.{ input, mask }, .{ .masked_fill = .{ .value = -9.0 } });
    const out = try @import("test_support.zig").execute(allocator, op);
    defer out.deinit();

    try std.testing.expectEqualSlices(usize, &.{ 2, 3, 4 }, out.shape.dims);
    const values = std.mem.bytesAsSlice(f32, try out.storage.?.readableBytes());
    try std.testing.expectEqual(@as(f32, -9), values[0]);
    try std.testing.expectEqual(@as(f32, -9), values[3]);
    try std.testing.expectEqual(@as(f32, 5), values[4]);
    try std.testing.expectEqual(@as(f32, 8), values[7]);
    try std.testing.expectEqual(@as(f32, -9), values[8]);
    try std.testing.expectEqual(@as(f32, -9), values[23]);
}

test "eager softmax runs end to end" {
    const allocator = std.testing.allocator;
    const input = try Tensor.fromSliceF64(allocator, &.{ 2, 2 }, &.{
        1.0, 2.0,
        3.0, 4.0,
    });
    defer input.deinit();

    const op = try Op.init(.softmax, &.{input}, .{ .softmax = .{ .axis = 1 } });
    const out = try @import("test_support.zig").execute(allocator, op);
    defer out.deinit();

    const bytes = try out.storage.?.readableBytes();
    const values = std.mem.bytesAsSlice(f64, bytes);
    try std.testing.expectApproxEqAbs(@as(f64, 0.2689414213699951), values[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.7310585786300049), values[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.2689414213699951), values[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.7310585786300049), values[3], 1e-12);
}

test "eager layer_norm runs end to end on cpu" {
    const allocator = std.testing.allocator;
    const input = try Tensor.fromSliceF32(allocator, &.{ 2, 2 }, &.{ 1.0, 3.0, 2.0, 4.0 });
    defer input.deinit();

    const op = try Op.init(.layer_norm, &.{input}, .{ .layer_norm = .{ .axis = 1, .eps = 1e-5 } });
    const out = try @import("test_support.zig").execute(allocator, op);
    defer out.deinit();

    const bytes = try out.storage.?.readableBytes();
    const values = std.mem.bytesAsSlice(f32, bytes);
    try std.testing.expectApproxEqAbs(@as(f32, -0.999995), values[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.999995), values[1], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, -0.999995), values[2], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.999995), values[3], 1e-4);
}

test "eager layer_norm packs negative-stride input to dense" {
    const allocator = std.testing.allocator;
    const base = try Tensor.fromSliceF32(allocator, &.{ 2, 2 }, &.{ 1.0, 3.0, 2.0, 4.0 });
    defer base.deinit();
    const input = try makeViewForTest(allocator, base, &.{ 2, 2 }, &.{ -2, 1 }, 2);
    defer input.deinit();

    const op = try Op.init(.layer_norm, &.{input}, .{ .layer_norm = .{ .axis = 1, .eps = 1e-5 } });
    var plan = try @import("test_support.zig").createPlan(allocator, op);
    defer plan.deinit();
    try std.testing.expectEqual(execution_layout.InputLayoutDecision.pack_to_dense, plan.input_layout_decision);

    var out = try executeAllWithPlan(allocator, op, &plan);
    defer out.deinit();
    const values = std.mem.bytesAsSlice(f32, try out.primary.storage.?.readableBytes());
    try std.testing.expectApproxEqAbs(@as(f32, -0.999995), values[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.999995), values[1], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, -0.999995), values[2], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.999995), values[3], 1e-4);
}

test "eager contiguous runs end to end" {
    const allocator = std.testing.allocator;
    const input = try Tensor.fromSliceF32(allocator, &.{3}, &.{ 1.0, 2.0, 3.0 });
    defer input.deinit();

    const op = try Op.init(.contiguous, &.{input}, .{ .none = {} });
    const out = try @import("test_support.zig").execute(allocator, op);
    defer out.deinit();

    const bytes = try out.storage.?.readableBytes();
    const values = std.mem.bytesAsSlice(f32, bytes);
    try std.testing.expectEqual(@as(f32, 1.0), values[0]);
    try std.testing.expectEqual(@as(f32, 2.0), values[1]);
    try std.testing.expectEqual(@as(f32, 3.0), values[2]);
}

test "eager dot runs end to end" {
    const allocator = std.testing.allocator;
    const a = try Tensor.fromSliceF64(allocator, &.{3}, &.{ 1.0, 2.0, 3.0 });
    defer a.deinit();
    const b = try Tensor.fromSliceF64(allocator, &.{3}, &.{ 4.0, 5.0, 6.0 });
    defer b.deinit();
    const op = try Op.init(.dot, &.{ a, b }, .{ .none = {} });
    const out = try @import("test_support.zig").execute(allocator, op);
    defer out.deinit();
    const values = std.mem.bytesAsSlice(f64, try out.storage.?.readableBytes());
    try std.testing.expectEqual(@as(f64, 32.0), values[0]);
}

test "eager matmul runs end to end" {
    const allocator = std.testing.allocator;
    const a = try Tensor.fromSliceF64(allocator, &.{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer a.deinit();
    const b = try Tensor.fromSliceF64(allocator, &.{ 3, 2 }, &.{ 7, 8, 9, 10, 11, 12 });
    defer b.deinit();
    const op = try Op.init(.matmul, &.{ a, b }, .{ .none = {} });
    const out = try @import("test_support.zig").execute(allocator, op);
    defer out.deinit();
    const values = std.mem.bytesAsSlice(f64, try out.storage.?.readableBytes());
    try std.testing.expectEqual(@as(f64, 58), values[0]);
    try std.testing.expectEqual(@as(f64, 64), values[1]);
    try std.testing.expectEqual(@as(f64, 139), values[2]);
    try std.testing.expectEqual(@as(f64, 154), values[3]);
}

test "eager matmul accepts transposed view inputs" {
    const allocator = std.testing.allocator;
    const a = try Tensor.fromSliceF32(allocator, &.{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer a.deinit();
    const b = try Tensor.fromSliceF32(allocator, &.{ 2, 4 }, &.{ 1, 2, 3, 4, 5, 6, 7, 8 });
    defer b.deinit();

    const at_op = try Op.init(.transpose, &.{a}, .{ .transpose = .{ .permutation = &.{ 1, 0 } } });
    const at = try @import("test_support.zig").execute(allocator, at_op);
    defer at.deinit();

    const op = try Op.init(.matmul, &.{ at, b }, .{ .none = {} });
    const out = try @import("test_support.zig").execute(allocator, op);
    defer out.deinit();

    const values = std.mem.bytesAsSlice(f32, try out.storage.?.readableBytes());
    try std.testing.expectApproxEqAbs(@as(f32, 21.0), values[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 26.0), values[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 31.0), values[2], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 36.0), values[3], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 27.0), values[4], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 34.0), values[5], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 41.0), values[6], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 48.0), values[7], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 33.0), values[8], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 42.0), values[9], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 51.0), values[10], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 60.0), values[11], 1e-5);
}

test "eager matmul plan accepts transposed input without dense downgrade" {
    const allocator = std.testing.allocator;
    const a = try Tensor.fromSliceF32(allocator, &.{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer a.deinit();
    const b = try Tensor.fromSliceF32(allocator, &.{ 2, 4 }, &.{ 1, 2, 3, 4, 5, 6, 7, 8 });
    defer b.deinit();

    const at_op = try Op.init(.transpose, &.{a}, .{ .transpose = .{ .permutation = &.{ 1, 0 } } });
    const at = try @import("test_support.zig").execute(allocator, at_op);
    defer at.deinit();

    const op = try Op.init(.matmul, &.{ at, b }, .{ .none = {} });
    var plan = try @import("test_support.zig").createPlan(allocator, op);
    defer plan.deinit();
    try std.testing.expectEqual(execution_layout.InputLayoutDecision.accept, plan.input_layout_decision);
    try std.testing.expectEqual(ExecutionInputRequirement.preserve, plan.input_requirement);
}

test "eager matmul packs negative-stride input to dense" {
    const allocator = std.testing.allocator;
    const base = try Tensor.fromSliceF32(allocator, &.{ 3, 2 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer base.deinit();
    const lhs = try makeViewForTest(allocator, base, &.{ 3, 2 }, &.{ -2, 1 }, 4);
    defer lhs.deinit();
    const rhs = try Tensor.fromSliceF32(allocator, &.{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer rhs.deinit();

    const op = try Op.init(.matmul, &.{ lhs, rhs }, .{ .none = {} });
    var plan = try @import("test_support.zig").createPlan(allocator, op);
    defer plan.deinit();
    try std.testing.expectEqual(execution_layout.InputLayoutDecision.pack_to_dense, plan.input_layout_decision);
    try std.testing.expectEqual(ExecutionInputRequirement.preserve, plan.input_requirement);

    var out = try executeAllWithPlan(allocator, op, &plan);
    defer out.deinit();
    const values = std.mem.bytesAsSlice(f32, try out.primary.storage.?.readableBytes());
    try std.testing.expectEqualSlices(f32, &.{ 23, 34, 15, 22, 7, 10 }, values);
}

test "eager one_hot runs end to end" {
    const allocator = std.testing.allocator;
    const index = try Tensor.fromSliceI64(allocator, &.{3}, &.{ 0, 2, 1 });
    defer index.deinit();
    const op = try Op.init(.one_hot, &.{index}, .{ .one_hot = .{ .num_classes = 4 } });
    const out = try @import("test_support.zig").execute(allocator, op);
    defer out.deinit();
    const values = std.mem.bytesAsSlice(f32, try out.storage.?.readableBytes());
    try std.testing.expectEqual(@as(f32, 1), values[0]);
    try std.testing.expectEqual(@as(f32, 1), values[6]);
    try std.testing.expectEqual(@as(f32, 1), values[9]);
}

test "eager embedding runs end to end" {
    const allocator = std.testing.allocator;
    const table = try Tensor.fromSliceF32(allocator, &.{ 3, 2 }, &.{ 10, 11, 20, 21, 30, 31 });
    defer table.deinit();
    const index = try Tensor.fromSliceI64(allocator, &.{ 2, 2 }, &.{ 2, 0, 1, 2 });
    defer index.deinit();
    const op = try Op.init(.embedding, &.{ table, index }, .{ .none = {} });
    const out = try @import("test_support.zig").execute(allocator, op);
    defer out.deinit();

    const values = std.mem.bytesAsSlice(f32, try out.storage.?.readableBytes());
    try std.testing.expectEqualSlices(f32, &.{
        30, 31,
        10, 11,
        20, 21,
        30, 31,
    }, values);
}

test "eager gather packs negative-stride input to dense" {
    const allocator = std.testing.allocator;
    const base = try Tensor.fromSliceF32(allocator, &.{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer base.deinit();
    const input = try makeViewForTest(allocator, base, &.{ 2, 3 }, &.{ -3, 1 }, 3);
    defer input.deinit();
    const index = try Tensor.fromSliceI64(allocator, &.{ 2, 2 }, &.{ 2, 0, 1, 2 });
    defer index.deinit();

    const op = try Op.init(.gather, &.{ input, index }, .{ .gather = .{ .axis = 1 } });
    var plan = try @import("test_support.zig").createPlan(allocator, op);
    defer plan.deinit();
    try std.testing.expectEqual(execution_layout.InputLayoutDecision.pack_to_dense, plan.input_layout_decision);

    var out = try executeAllWithPlan(allocator, op, &plan);
    defer out.deinit();
    const values = std.mem.bytesAsSlice(f32, try out.primary.storage.?.readableBytes());
    try std.testing.expectEqualSlices(f32, &.{ 6, 4, 2, 3 }, values);
}

test "eager cat runs end to end" {
    const allocator = std.testing.allocator;
    const a = try Tensor.fromSliceF32(allocator, &.{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer a.deinit();
    const b = try Tensor.fromSliceF32(allocator, &.{ 2, 1 }, &.{ 5, 6 });
    defer b.deinit();
    const op = try Op.init(.cat, &.{ a, b }, .{ .concat = .{ .axis = 1 } });
    const out = try @import("test_support.zig").execute(allocator, op);
    defer out.deinit();
    const values = std.mem.bytesAsSlice(f32, try out.storage.?.readableBytes());
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 5, 3, 4, 6 }, values);
}

test "eager stack runs end to end" {
    const allocator = std.testing.allocator;
    const a = try Tensor.fromSliceF32(allocator, &.{2}, &.{ 1, 2 });
    defer a.deinit();
    const b = try Tensor.fromSliceF32(allocator, &.{2}, &.{ 3, 4 });
    defer b.deinit();
    const op = try Op.init(.stack, &.{ a, b }, .{ .stack = .{ .axis = 0 } });
    const out = try @import("test_support.zig").execute(allocator, op);
    defer out.deinit();
    const values = std.mem.bytesAsSlice(f32, try out.storage.?.readableBytes());
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3, 4 }, values);
}

test "eager slice copy path runs end to end" {
    const allocator = std.testing.allocator;
    const input = try Tensor.fromSliceF32(allocator, &.{6}, &.{ 1, 2, 3, 4, 5, 6 });
    defer input.deinit();
    const op = try Op.init(.slice, &.{input}, .{ .slice = .{ .ranges = &.{.{ .start = 1, .stop = 6, .step = 2 }} } });
    const out = try @import("test_support.zig").execute(allocator, op);
    defer out.deinit();
    const values = std.mem.bytesAsSlice(f32, try out.storage.?.readableBytes());
    try std.testing.expectEqualSlices(f32, &.{ 2, 4, 6 }, values);
}

test "eager dense op packs contiguous offset view before dispatch" {
    const allocator = std.testing.allocator;
    const input = try Tensor.fromSliceF32(allocator, &.{4}, &.{ 1, 2, 3, 4 });
    defer input.deinit();
    const slice = try Op.init(.slice, &.{input}, .{ .slice = .{ .ranges = &.{.{ .start = 1, .stop = 3, .step = 1 }} } });
    const view = try @import("test_support.zig").execute(allocator, slice);
    defer view.deinit();
    const rhs = try Tensor.fromSliceF32(allocator, &.{2}, &.{ 10, 20 });
    defer rhs.deinit();

    const add = try Op.init(.add, &.{ view, rhs }, .{ .binary = .{} });
    const out = try @import("test_support.zig").execute(allocator, add);
    defer out.deinit();

    const values = std.mem.bytesAsSlice(f32, try out.storage.?.readableBytes());
    try std.testing.expectEqualSlices(f32, &.{ 12, 23 }, values);
}

test "eager reshape aliases storage" {
    const allocator = std.testing.allocator;
    const input = try Tensor.fromSliceF32(allocator, &.{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer input.deinit();

    const op = try Op.init(.reshape, &.{input}, .{ .reshape = .{ .shape = &.{ 3, 2 } } });
    const out = try @import("test_support.zig").execute(allocator, op);
    defer out.deinit();

    try std.testing.expect(input.storage == out.storage);
    try std.testing.expectEqual(@as(usize, 2), out.shape.rank());
}

test "eager execution rejects non-contiguous reshape input" {
    const allocator = std.testing.allocator;
    const input = try Tensor.fromSliceF32(allocator, &.{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer input.deinit();

    const transpose = try Op.init(.transpose, &.{input}, .{ .transpose = .{ .permutation = &.{ 1, 0 } } });
    const transposed = try @import("test_support.zig").execute(allocator, transpose);
    defer transposed.deinit();

    const reshape = try Op.init(.reshape, &.{transposed}, .{ .reshape = .{ .shape = &.{ 6, 1 } } });
    try std.testing.expectError(error.InputNotContiguous, @import("test_support.zig").execute(allocator, reshape));
}

test "topk runs through executeAll with values and indices outputs" {
    const allocator = std.testing.allocator;
    const input = try Tensor.fromSliceF32(allocator, &.{ 2, 5 }, &.{
        1,  2, 3, 4, 5,
        10, 9, 8, 7, 6,
    });
    defer input.deinit();

    const op = try Op.init(.topk, &.{input}, .{ .topk = .{ .k = 2, .axis = 1 } });
    var result = try @import("test_support.zig").executeAll(allocator, op);
    defer result.deinit();
    try std.testing.expect(result.secondary != null);

    const value_bytes = try result.primary.storage.?.readableBytes();
    const values = std.mem.bytesAsSlice(f32, value_bytes);
    try std.testing.expectEqual(@as(f32, 5), values[0]);
    try std.testing.expectEqual(@as(f32, 4), values[1]);
    try std.testing.expectEqual(@as(f32, 10), values[2]);
    try std.testing.expectEqual(@as(f32, 9), values[3]);

    const idx_bytes = try result.secondary.?.storage.?.readableBytes();
    const idx = std.mem.bytesAsSlice(i64, idx_bytes);
    try std.testing.expectEqual(@as(i64, 4), idx[0]);
    try std.testing.expectEqual(@as(i64, 3), idx[1]);
    try std.testing.expectEqual(@as(i64, 0), idx[2]);
    try std.testing.expectEqual(@as(i64, 1), idx[3]);
}

test "eager reduce_to_shape sums broadcasted axes" {
    const allocator = std.testing.allocator;
    const input = try Tensor.fromSliceF32(allocator, &.{ 2, 3 }, &.{
        1, 2, 3,
        4, 5, 6,
    });
    defer input.deinit();
    const op = try Op.init(.reduce_to_shape, &.{input}, .{ .reduce_to_shape = .{ .shape = &.{ 1, 3 } } });
    const out = try @import("test_support.zig").execute(allocator, op);
    defer out.deinit();
    const values = std.mem.bytesAsSlice(f32, try out.storage.?.readableBytes());
    try std.testing.expectEqualSlices(f32, &.{ 5, 7, 9 }, values);
}

test "eager reduce_to_shape right-aligns target dims" {
    const allocator = std.testing.allocator;
    const input = try Tensor.fromSliceF32(allocator, &.{ 2, 2, 3 }, &.{
        1,  2,  3,
        4,  5,  6,
        7,  8,  9,
        10, 11, 12,
    });
    defer input.deinit();
    const op = try Op.init(.reduce_to_shape, &.{input}, .{ .reduce_to_shape = .{ .shape = &.{ 1, 3 } } });
    const out = try @import("test_support.zig").execute(allocator, op);
    defer out.deinit();
    const values = std.mem.bytesAsSlice(f32, try out.storage.?.readableBytes());
    try std.testing.expectEqualSlices(f32, &.{ 22, 26, 30 }, values);
}

test "eager reduce_to_shape preserves aligned inner batch dims" {
    const allocator = std.testing.allocator;
    const input = try Tensor.fromSliceF32(allocator, &.{ 2, 2, 3 }, &.{
        1,  2,  3,
        4,  5,  6,
        7,  8,  9,
        10, 11, 12,
    });
    defer input.deinit();
    const op = try Op.init(.reduce_to_shape, &.{input}, .{ .reduce_to_shape = .{ .shape = &.{ 2, 3 } } });
    const out = try @import("test_support.zig").execute(allocator, op);
    defer out.deinit();
    const values = std.mem.bytesAsSlice(f32, try out.storage.?.readableBytes());
    try std.testing.expectEqualSlices(f32, &.{
        8,  10, 12,
        14, 16, 18,
    }, values);
}

test "eager scatter_add axis accumulation" {
    const allocator = std.testing.allocator;
    const base = try Tensor.fromSliceF32(allocator, &.{ 2, 3 }, &.{
        1, 2, 3,
        4, 5, 6,
    });
    defer base.deinit();
    const index = try Tensor.fromSliceI64(allocator, &.{ 2, 2 }, &.{
        2, 0,
        1, 1,
    });
    defer index.deinit();
    const updates = try Tensor.fromSliceF32(allocator, &.{ 2, 2 }, &.{
        10, 20,
        30, 40,
    });
    defer updates.deinit();
    const op = try Op.init(.scatter_add, &.{ base, index, updates }, .{ .scatter_add = .{ .axis = 1 } });
    const out = try @import("test_support.zig").execute(allocator, op);
    defer out.deinit();
    const values = std.mem.bytesAsSlice(f32, try out.storage.?.readableBytes());
    try std.testing.expectEqualSlices(f32, &.{
        21, 2,  13,
        4,  75, 6,
    }, values);
}
