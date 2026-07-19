const std = @import("std");
const hao = @import("hao");
const engine_api = @import("../../compute/engine.zig");
const tensor_types = @import("../../compute/types/tensor/index.zig");
const dispatch = @import("../../compute/backend/dispatch.zig");
const config = @import("../../config.zig");
const metal_backend = @import("../../compute/backend/metal/common.zig");
const autograd_types = @import("../../compute/types/autograd.zig");
const autograd_execution = @import("../../compute/execution/autograd.zig");
const autograd_compose = @import("../../compute/compose/derive.zig");
const OpTag = @import("../../compute/types/operation/tag.zig").OpTag;
const Op = @import("../../compute/types/operation/op.zig").Op;
const execution_metadata = @import("../../compute/types/operation/execution_metadata.zig");
const SliceRange = @import("../../compute/types/operation/options.zig").SliceRange;
const grad_mode = @import("../../compute/grad_mode.zig");
const captured_schema = @import("captured/schema.zig");
const captured_lowering = @import("captured/lowering.zig");
const repr_common = @import("../../compute/shared/repr.zig");

const abi = hao.js.abi;
const allocator = std.heap.c_allocator;
const Tensor = engine_api.Tensor;
const AxisName = tensor_types.AxisName;
var random_state = std.Random.DefaultPrng.init(0xA66F_0001);


const tensor_type_id: u32 = 1;
const compiled_executable_type_id: u32 = 2;
var compiled_executable_class_id: u32 = 0;
var compiled_executable_runtime: ?*anyopaque = null;

const TensorObject = struct { value: *Tensor };
const CompiledExecutable = struct {
    json: []u8,
    mode: enum { none, graph, eager_forward } = .none,
    plan_outcome: enum { none, native_graph, eager_forward } = .none,
    plan_outcome_reason: ?[]u8 = null,
    last_fallback_kind: ?[]u8 = null,
    last_fallback_reason: ?[]u8 = null,
    specialization_capture_count: usize = 0,
    specialization_reuse_count: usize = 0,
    eager_fallback_count: usize = 0,

    fn deinit(self: *CompiledExecutable) void {
        allocator.free(self.json);
        if (self.plan_outcome_reason) |value| allocator.free(value);
        if (self.last_fallback_kind) |value| allocator.free(value);
        if (self.last_fallback_reason) |value| allocator.free(value);
        allocator.destroy(self);
    }
};

fn compiledExecutableFromValue(ctx: abi.JSContext, value: abi.JSValueConst) ?*CompiledExecutable {
    if (compiled_executable_class_id == 0) return null;
    const handle = abi.jsHostObjectHandleWithClass(ctx, value, compiled_executable_class_id, compiled_executable_type_id) orelse return null;
    return @ptrFromInt(@as(usize, @intCast(handle)));
}

fn compiledExecutableFinalizer(_: u32, handle: u64) callconv(.c) void {
    const executable: *CompiledExecutable = @ptrFromInt(@as(usize, @intCast(handle)));
    executable.deinit();
}

fn replaceOwnedString(slot: *?[]u8, value: ?[]const u8) !void {
    if (slot.*) |old| allocator.free(old);
    slot.* = if (value) |text| try allocator.dupe(u8, text) else null;
}

fn parseCapturedDType(name: []const u8) !engine_api.DType {
    if (std.mem.eql(u8, name, "f32")) return .f32;
    if (std.mem.eql(u8, name, "f64")) return .f64;
    if (std.mem.eql(u8, name, "i64")) return .i64;
    return error.InvalidDType;
}

fn createCapturedScalar(dtype: engine_api.DType, value: f64) !*Tensor {
    return switch (dtype) {
        .f32 => Tensor.fromSliceF32(allocator, &.{}, &.{@floatCast(value)}),
        .f64 => Tensor.fromSliceF64(allocator, &.{}, &.{value}),
        .i64 => Tensor.fromSliceI64(allocator, &.{}, &.{@intFromFloat(value)}),
    };
}

fn createCapturedRandom(shape: []const usize, dtype: engine_api.DType, normal: bool) !*Tensor {
    const value = try Tensor.createContiguous(allocator, shape, dtype, .cpu, false);
    errdefer value.deinit();
    var random = random_state.random();
    const bytes = try value.storage.?.writableBytes();
    switch (dtype) {
        .f32 => for (std.mem.bytesAsSlice(f32, bytes)) |*item| {
            item.* = if (normal) random.floatNorm(f32) else random.float(f32);
        },
        .f64 => for (std.mem.bytesAsSlice(f64, bytes)) |*item| {
            item.* = if (normal) random.floatNorm(f64) else random.float(f64);
        },
        .i64 => for (std.mem.bytesAsSlice(i64, bytes)) |*item| {
            item.* = @intCast(random.int(i32));
        },
    }
    return value;
}

fn parseCapturedProgram(executable: *const CompiledExecutable) !std.json.Parsed(captured_schema.CapturedProgramJson) {
    return std.json.parseFromSlice(captured_schema.CapturedProgramJson, allocator, executable.json, .{ .ignore_unknown_fields = false });
}

fn jsCompiledExecutableAnalyze(ctx: abi.JSContext, this_value: abi.JSValueConst, _: c_int, _: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    const executable = compiledExecutableFromValue(ctx, this_value) orelse return typeError(ctx, "invalid compiled executable");
    var parsed = parseCapturedProgram(executable) catch return errorValue(ctx, "failed to parse captured program");
    defer parsed.deinit();
    const analysis = captured_lowering.analyzeLowerability(parsed.value, parseCapturedDType);
    const result = abi.jsNewObject(ctx);
    if (abi.jsSetProperty(ctx, result, "lowerable", abi.jsBool(ctx, analysis.lowerable)) < 0) return errorValue(ctx, "failed to create analysis");
    if (analysis.failure) |failure| {
        const detail = abi.jsNewObject(ctx);
        if (abi.jsSetProperty(ctx, detail, "category", abi.jsString(ctx, failure.category)) < 0 or
            abi.jsSetProperty(ctx, detail, "reason", abi.jsString(ctx, failure.reason)) < 0)
            return errorValue(ctx, "failed to create analysis");
        if (failure.node_id) |node_id| {
            if (abi.jsSetProperty(ctx, detail, "nodeId", abi.jsInt32(ctx, @intCast(node_id))) < 0) return errorValue(ctx, "failed to create analysis");
        }
        if (failure.node_kind) |node_kind| {
            if (abi.jsSetProperty(ctx, detail, "nodeKind", abi.jsString(ctx, node_kind)) < 0) return errorValue(ctx, "failed to create analysis");
        }
        if (abi.jsSetProperty(ctx, result, "failure", detail) < 0) return errorValue(ctx, "failed to create analysis");
    }
    return result;
}

fn jsCompiledExecutableSummary(ctx: abi.JSContext, this_value: abi.JSValueConst, _: c_int, _: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    const executable = compiledExecutableFromValue(ctx, this_value) orelse return typeError(ctx, "invalid compiled executable");
    var parsed = parseCapturedProgram(executable) catch return errorValue(ctx, "failed to parse captured program");
    defer parsed.deinit();
    const result = abi.jsNewObject(ctx);
    const mode: [:0]const u8 = switch (executable.mode) { .none => "none", .graph => "graph", .eager_forward => "eager-forward" };
    const outcome: [:0]const u8 = switch (executable.plan_outcome) { .none => "none", .native_graph => "native-graph", .eager_forward => "eager-forward" };
    if (abi.jsSetProperty(ctx, result, "inputCount", abi.jsInt32(ctx, @intCast(parsed.value.inputArity))) < 0 or
        abi.jsSetProperty(ctx, result, "nodeCount", abi.jsInt32(ctx, @intCast(parsed.value.nodes.len))) < 0 or
        abi.jsSetProperty(ctx, result, "mode", abi.jsString(ctx, mode)) < 0 or
        abi.jsSetProperty(ctx, result, "planOutcome", abi.jsString(ctx, outcome)) < 0 or
        abi.jsSetProperty(ctx, result, "planOutcomeReason", if (executable.plan_outcome_reason) |value| abi.jsString(ctx, value) else abi.jsNull(ctx)) < 0 or
        abi.jsSetProperty(ctx, result, "specializationCaptureCount", abi.jsInt32(ctx, @intCast(executable.specialization_capture_count))) < 0 or
        abi.jsSetProperty(ctx, result, "specializationReuseCount", abi.jsInt32(ctx, @intCast(executable.specialization_reuse_count))) < 0 or
        abi.jsSetProperty(ctx, result, "eagerFallbackCount", abi.jsInt32(ctx, @intCast(executable.eager_fallback_count))) < 0 or
        abi.jsSetProperty(ctx, result, "lastFallbackKind", if (executable.last_fallback_kind) |value| abi.jsString(ctx, value) else abi.jsNull(ctx)) < 0 or
        abi.jsSetProperty(ctx, result, "lastFallbackReason", if (executable.last_fallback_reason) |value| abi.jsString(ctx, value) else abi.jsNull(ctx)) < 0)
        return errorValue(ctx, "failed to create executable summary");
    return result;
}

fn jsCompiledExecutableRun(ctx: abi.JSContext, this_value: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    if (argc != 1) return typeError(ctx, "compiled executable run expects an input array");
    const executable = compiledExecutableFromValue(ctx, this_value) orelse return typeError(ctx, "invalid compiled executable");
    const inputs = readTensorList(ctx, argv[0], "compiled inputs") orelse return typeError(ctx, "compiled inputs must be an array of Tensors");
    defer allocator.free(inputs);
    var parsed = parseCapturedProgram(executable) catch return errorValue(ctx, "failed to parse captured program");
    defer parsed.deinit();
    const analysis = captured_lowering.analyzeLowerability(parsed.value, parseCapturedDType);
    if (!analysis.lowerable) {
        if (analysis.failure) |failure| {
            if (failure.node_kind) |kind| {
                if (std.mem.eql(u8, kind, "where")) return errorValue(ctx, "compiled graph is not lowerable: where");
            }
        }
        return errorValue(ctx, "compiled graph is not lowerable");
    }
    var lowered = captured_lowering.lowerToGraph(allocator, parsed.value, inputs, parseCapturedDType, createCapturedScalar, createCapturedRandom) catch return errorValue(ctx, "failed to lower captured graph");
    defer lowered.deinit(allocator);
    var result = engine_api.Engine.init(allocator, .{}).executeGraph(&lowered.graph, lowered.input_values.items) catch return errorValue(ctx, "failed to execute captured graph");
    if (result.outputs.len != 1) {
        result.deinit();
        return errorValue(ctx, "captured graph produced an unsupported output count");
    }
    const output = result.outputs[0];
    for (result.values, 0..) |value, index| {
        if (value == output) {
            result.owned[index] = false;
            break;
        }
    }
    result.deinit();
    executable.mode = .graph;
    executable.plan_outcome = .native_graph;
    replaceOwnedString(&executable.plan_outcome_reason, null) catch return errorValue(ctx, "failed to record plan outcome");
    return createTensorObject(ctx, output);
}

fn optionalStringArgument(ctx: abi.JSContext, value: abi.JSValueConst) !?[]u8 {
    if (abi.jsIsUndefined(value) or abi.jsIsNull(value)) return null;
    const text = try abi.jsStringAlloc(ctx, value, allocator);
    return text;
}

fn jsCompiledExecutableRecordPlanOutcome(ctx: abi.JSContext, this_value: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    const executable = compiledExecutableFromValue(ctx, this_value) orelse return typeError(ctx, "invalid compiled executable");
    if (argc < 1) return typeError(ctx, "recordPlanOutcome expects an outcome");
    const outcome = abi.jsStringAlloc(ctx, argv[0], allocator) catch return errorValue(ctx, "invalid plan outcome");
    defer allocator.free(outcome);
    executable.plan_outcome = if (std.mem.eql(u8, outcome, "native-graph") or std.mem.eql(u8, outcome, "graph")) .native_graph else if (std.mem.eql(u8, outcome, "eager-forward")) .eager_forward else .none;
    executable.mode = if (executable.plan_outcome == .native_graph) .graph else if (executable.plan_outcome == .eager_forward) .eager_forward else .none;
    const reason = optionalStringArgument(ctx, if (argc > 2) argv[2] else abi.jsUndefined(ctx)) catch return errorValue(ctx, "invalid plan outcome reason");
    defer if (reason) |value| allocator.free(value);
    replaceOwnedString(&executable.plan_outcome_reason, reason) catch return errorValue(ctx, "failed to record plan outcome");
    return abi.jsUndefined(ctx);
}

fn jsCompiledExecutableRecordFallback(ctx: abi.JSContext, this_value: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    const executable = compiledExecutableFromValue(ctx, this_value) orelse return typeError(ctx, "invalid compiled executable");
    executable.mode = .eager_forward;
    executable.plan_outcome = .eager_forward;
    executable.eager_fallback_count += 1;
    const kind = optionalStringArgument(ctx, if (argc > 0) argv[0] else abi.jsUndefined(ctx)) catch return errorValue(ctx, "invalid fallback kind");
    defer if (kind) |value| allocator.free(value);
    const reason = optionalStringArgument(ctx, if (argc > 1) argv[1] else abi.jsUndefined(ctx)) catch return errorValue(ctx, "invalid fallback reason");
    defer if (reason) |value| allocator.free(value);
    replaceOwnedString(&executable.last_fallback_kind, kind) catch return errorValue(ctx, "failed to record fallback");
    replaceOwnedString(&executable.last_fallback_reason, reason) catch return errorValue(ctx, "failed to record fallback");
    return abi.jsUndefined(ctx);
}

fn jsCompiledExecutableRecordCapture(ctx: abi.JSContext, this_value: abi.JSValueConst, _: c_int, _: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    const executable = compiledExecutableFromValue(ctx, this_value) orelse return typeError(ctx, "invalid compiled executable");
    executable.specialization_capture_count += 1;
    return abi.jsUndefined(ctx);
}

fn jsCompiledExecutableRecordReuse(ctx: abi.JSContext, this_value: abi.JSValueConst, _: c_int, _: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    const executable = compiledExecutableFromValue(ctx, this_value) orelse return typeError(ctx, "invalid compiled executable");
    executable.specialization_reuse_count += 1;
    return abi.jsUndefined(ctx);
}

fn jsCompiledExecutableNoop(ctx: abi.JSContext, _: abi.JSValueConst, _: c_int, _: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    return abi.jsUndefined(ctx);
}

fn jsCreateCompiledExecutable(ctx: abi.JSContext, _: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    if (argc != 1) return typeError(ctx, "compiled executable requires captured program JSON");
    const json = abi.jsStringAlloc(ctx, argv[0], allocator) catch return typeError(ctx, "captured program must be a string");
    const executable = allocator.create(CompiledExecutable) catch {
        allocator.free(json);
        return errorValue(ctx, "out of memory");
    };
    executable.* = .{ .json = json };
    const runtime = abi.jsRuntime(ctx);
    if (compiled_executable_class_id == 0 or compiled_executable_runtime != runtime or !abi.jsHostObjectClassIsValid(ctx, compiled_executable_class_id)) {
        compiled_executable_class_id = abi.jsCreateHostObjectClass(ctx, "AffonCompiledExecutable");
        if (compiled_executable_class_id == 0) {
            compiledExecutableFinalizer(compiled_executable_type_id, @intCast(@intFromPtr(executable)));
            return errorValue(ctx, "failed to initialize compiled executable class");
        }
        compiled_executable_runtime = runtime;
    }
    const object = abi.createJSHostObjectWithClass(ctx, compiled_executable_class_id, compiled_executable_type_id, @intCast(@intFromPtr(executable)), compiledExecutableFinalizer);
    if (abi.jsIsException(object)) {
        compiledExecutableFinalizer(compiled_executable_type_id, @intCast(@intFromPtr(executable)));
        return object;
    }
    const methods = [_]struct { name: [:0]const u8, callback: abi.JSCallback, length: c_int }{
        .{ .name = "analyze", .callback = jsCompiledExecutableAnalyze, .length = 0 },
        .{ .name = "summary", .callback = jsCompiledExecutableSummary, .length = 0 },
        .{ .name = "run", .callback = jsCompiledExecutableRun, .length = 1 },
        .{ .name = "exportBundle", .callback = jsCompiledExecutableNoop, .length = 1 },
        .{ .name = "exportReport", .callback = jsCompiledExecutableNoop, .length = 1 },
        .{ .name = "recordPlanOutcome", .callback = jsCompiledExecutableRecordPlanOutcome, .length = 4 },
        .{ .name = "recordFallback", .callback = jsCompiledExecutableRecordFallback, .length = 3 },
        .{ .name = "recordSpecializationCapture", .callback = jsCompiledExecutableRecordCapture, .length = 1 },
        .{ .name = "recordSpecializationReuse", .callback = jsCompiledExecutableRecordReuse, .length = 1 },
    };
    for (methods) |method| if (abi.jsSetFunction(ctx, object, method.name, method.callback, method.length) < 0) {
        abi.jsFreeValue(ctx, object);
        return errorValue(ctx, "failed to initialize compiled executable");
    };
    return object;
}

fn trackResult(result: *Tensor, inputs: []const *Tensor, op_tag: OpTag, axis: ?usize, scalar_a: ?f64, scalar_b: ?f64) !void {
    return trackResultWithSavedInputs(result, inputs, inputs, op_tag, axis, null, false, scalar_a, scalar_b);
}

fn trackResultWithSavedInputs(
    result: *Tensor,
    inputs: []const *Tensor,
    saved_inputs: []const *const Tensor,
    op_tag: OpTag,
    axis: ?usize,
    permute_axes: ?[]const usize,
    keepdim: bool,
    scalar_a: ?f64,
    scalar_b: ?f64,
) !void {
    if (!grad_mode.isEnabled()) return;
    if (result.dtype == .i64) return;
    var requires_grad = false;
    var parents: std.ArrayList(autograd_types.Parent) = .empty;
    defer parents.deinit(allocator);
    for (inputs, 0..) |input, slot| {
        if (autograd_types.State.fromTensor(input)) |state| {
            if (state.isTrainable()) requires_grad = true;
            if (state.isTrainable()) try parents.append(allocator, .{ .value = input, .input_slot = slot });
        }
    }
    if (!requires_grad) return;
    const state = try autograd_types.State.create(allocator, result, true);
    errdefer {
        result.setAutogradStateRaw(null);
        allocator.destroy(state);
    }
    const node = try autograd_execution.createNode(allocator, op_tag, parents.items, saved_inputs, result, null, axis, keepdim, null, permute_axes, scalar_a, scalar_b);
    state.attachNode(node);
}

fn trackTopKResult(result: *Tensor, indices: *Tensor, input: *Tensor, axis: usize) !void {
    if (!grad_mode.isEnabled()) return;
    const state = autograd_types.State.fromTensor(input) orelse return;
    if (!state.isTrainable()) return;
    const output_state = try autograd_types.State.create(allocator, result, true);
    errdefer {
        result.setAutogradStateRaw(null);
        allocator.destroy(output_state);
    }
    const node = try autograd_execution.createNode(
        allocator,
        .topk,
        &.{.{ .value = input, .input_slot = 0 }},
        &.{input},
        null,
        indices,
        axis,
        false,
        null,
        null,
        null,
        null,
    );
    output_state.attachNode(node);
}

fn trackSliceResult(result: *Tensor, input: *Tensor, ranges: []const SliceRange) !void {
    if (!grad_mode.isEnabled()) return;
    var requires_grad = false;
    var parents: std.ArrayList(autograd_types.Parent) = .empty;
    defer parents.deinit(allocator);
    if (autograd_types.State.fromTensor(input)) |state| {
        requires_grad = state.isTrainable();
        if (requires_grad) try parents.append(allocator, .{ .value = input, .input_slot = 0 });
    }
    if (!requires_grad) return;
    const state = try autograd_types.State.create(allocator, result, true);
    errdefer {
        result.setAutogradStateRaw(null);
        allocator.destroy(state);
    }
    const node = try autograd_execution.createNode(allocator, .slice, parents.items, &.{input}, result, null, null, false, ranges, null, null, null);
    state.attachNode(node);
}

fn tensorFinalizer(_: u32, handle: u64) callconv(.c) void {
    const object: *TensorObject = @ptrFromInt(@as(usize, @intCast(handle)));
    if (autograd_types.State.fromTensor(object.value) != null) {
        autograd_execution.releaseOwnedTensor(object.value);
    } else {
        object.value.deinit();
    }
    allocator.destroy(object);
}

fn errorValue(ctx: abi.JSContext, message: [*:0]const u8) abi.JSValue {
    return abi.jsThrowError(ctx, message);
}

fn typeError(ctx: abi.JSContext, message: [*:0]const u8) abi.JSValue {
    return abi.jsThrowTypeError(ctx, message);
}

fn tensorFromValue(ctx: abi.JSContext, value: abi.JSValueConst) ?*Tensor {
    const handle = abi.jsHostObjectHandle(ctx, value, tensor_type_id) orelse return null;
    const object: *TensorObject = @ptrFromInt(@as(usize, @intCast(handle)));
    return object.value;
}

fn readAxesOption(ctx: abi.JSContext, options: abi.JSValueConst, rank: usize) !?[]const AxisName {
    if (abi.jsIsUndefined(options)) return null;
    const axes_value = abi.jsGetProperty(ctx, options, "axes");
    defer abi.jsFreeValue(ctx, axes_value);
    if (abi.jsIsUndefined(axes_value) or abi.jsIsNull(axes_value)) return null;
    if (!abi.jsIsArray(ctx, axes_value)) return error.InvalidAxes;
    const length_value = abi.jsGetProperty(ctx, axes_value, "length");
    defer abi.jsFreeValue(ctx, length_value);
    var length: i32 = 0;
    if (abi.jsToInt32(ctx, &length, length_value) < 0 or length < 0 or @as(usize, @intCast(length)) != rank) return error.AxisRankMismatch;
    const axes = try allocator.alloc(AxisName, rank);
    var initialized: usize = 0;
    errdefer {
        for (axes[0..initialized]) |axis| allocator.free(axis);
        allocator.free(axes);
    }
    for (0..rank) |index| {
        const item = abi.jsGetArrayElement(ctx, axes_value, @intCast(index));
        defer abi.jsFreeValue(ctx, item);
        if (!abi.jsIsString(item)) return error.InvalidAxes;
        axes[index] = try abi.jsStringAlloc(ctx, item, allocator);
        initialized += 1;
    }
    return axes;
}

fn freeAxes(axes: ?[]const AxisName) void {
    const names = axes orelse return;
    for (names) |name| allocator.free(name);
    allocator.free(names);
}

fn jsTensorGrad(ctx: abi.JSContext, this_value: abi.JSValueConst) callconv(.c) abi.JSValue {
    const tensor = tensorFromValue(ctx, this_value) orelse return abi.jsUndefined(ctx);
    const state = autograd_types.State.fromTensor(tensor) orelse return abi.jsUndefined(ctx);
    const gradient = state.gradient orelse return abi.jsUndefined(ctx);
    const copy = autograd_execution.cloneTensor(allocator, gradient) catch return errorValue(ctx, "failed to copy gradient");
    return createTensorObject(ctx, copy);
}

fn jsTensorGradDevice(ctx: abi.JSContext, this_value: abi.JSValueConst) callconv(.c) abi.JSValue {
    const tensor = tensorFromValue(ctx, this_value) orelse return abi.jsUndefined(ctx);
    const state = autograd_types.State.fromTensor(tensor) orelse return abi.jsUndefined(ctx);
    const gradient = state.gradient orelse return abi.jsUndefined(ctx);
    const device: [:0]const u8 = switch (gradient.device() orelse .cpu) {
        .cpu => "cpu",
        .metal => "metal",
    };
    return abi.jsString(ctx, device);
}

fn jsTensorBackward(ctx: abi.JSContext, this_value: abi.JSValueConst, _: c_int, _: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    const loss = tensorFromValue(ctx, this_value) orelse return typeError(ctx, "backward expects a Tensor");
    if (loss.shape.numel() != 1) return errorValue(ctx, "backward() failed: loss must be scalar");
    const previous_grad_mode = grad_mode.isEnabled();
    grad_mode.setEnabled(false);
    defer grad_mode.setEnabled(previous_grad_mode);
    var derived = autograd_compose.buildFromLossTensor(loss) catch return errorValue(ctx, "backward() failed: failed to build gradient graph");
    defer derived.deinit(allocator);
    autograd_compose.executeForTensor(loss, &derived) catch return errorValue(ctx, "backward() failed: failed to execute gradient graph");
    return abi.jsUndefined(ctx);
}

const TensorInputError = anyerror;

fn flattenTensorInput(
    ctx: abi.JSContext,
    value: abi.JSValueConst,
    depth: usize,
    shape: *std.ArrayList(usize),
    values: *std.ArrayList(f32),
) TensorInputError!void {
    if (abi.jsIsArray(ctx, value)) {
        const length_value = abi.jsGetProperty(ctx, value, "length");
        defer abi.jsFreeValue(ctx, length_value);
        const length = integerArgument(ctx, length_value, "tensor length") orelse return error.InvalidTensorData;
        if (shape.items.len < depth + 1) {
            try shape.append(allocator, length);
        } else if (shape.items[depth] != length) {
            return error.InconsistentTensorShape;
        }
        for (0..length) |index| {
            const item = abi.jsGetArrayElement(ctx, value, @intCast(index));
            defer abi.jsFreeValue(ctx, item);
            try flattenTensorInput(ctx, item, depth + 1, shape, values);
        }
        return;
    }
    if (depth != shape.items.len) return error.InconsistentTensorShape;
    var number: f64 = 0;
    if (abi.jsToFloat64(ctx, &number, value) < 0) return error.InvalidTensorData;
    try values.append(allocator, @floatCast(number));
}

fn createTensorObject(ctx: abi.JSContext, value: *Tensor) abi.JSValue {
    const object = allocator.create(TensorObject) catch {
        value.deinit();
        return errorValue(ctx, "out of memory");
    };
    object.* = .{ .value = value };
    const result = abi.createJSHostObject(ctx, tensor_type_id, @intCast(@intFromPtr(object)), tensorFinalizer);
    if (abi.jsIsException(result)) {
        value.deinit();
        allocator.destroy(object);
        return result;
    }
    const shape = abi.jsNewArray(ctx);
    if (abi.jsIsException(shape)) {
        abi.jsFreeValue(ctx, result);
        return errorValue(ctx, "failed to initialize Tensor object");
    }
    var shape_owned = true;
    defer if (shape_owned) abi.jsFreeValue(ctx, shape);
    for (value.shape.dims, 0..) |dim, index| {
        if (abi.jsSetArrayElement(ctx, shape, @intCast(index), abi.jsInt32(ctx, @intCast(dim))) < 0) {
            abi.jsFreeValue(ctx, result);
            return errorValue(ctx, "failed to initialize Tensor shape");
        }
    }
    if (abi.jsSetProperty(ctx, result, "shape", shape) < 0) {
        abi.jsFreeValue(ctx, result);
        return errorValue(ctx, "failed to initialize Tensor object");
    }
    shape_owned = false;
    if (value.axes) |axes| {
        const axes_value = abi.jsNewArray(ctx);
        if (abi.jsIsException(axes_value)) {
            abi.jsFreeValue(ctx, result);
            return errorValue(ctx, "failed to initialize Tensor axes");
        }
        var axes_owned = true;
        defer if (axes_owned) abi.jsFreeValue(ctx, axes_value);
        for (axes, 0..) |axis, index| {
            if (abi.jsSetArrayElement(ctx, axes_value, @intCast(index), abi.jsString(ctx, axis)) < 0) {
                abi.jsFreeValue(ctx, result);
                return errorValue(ctx, "failed to initialize Tensor axes");
            }
        }
        if (abi.jsSetProperty(ctx, result, "axes", axes_value) < 0) {
            abi.jsFreeValue(ctx, result);
            return errorValue(ctx, "failed to initialize Tensor axes");
        }
        axes_owned = false;
    }
    const dtype_name: [:0]const u8 = switch (value.dtype) {
        .f32 => "f32",
        .f64 => "f64",
        .i64 => "i64",
    };
    const device_name: [:0]const u8 = switch (value.device() orelse .cpu) {
        .cpu => "cpu",
        .metal => "metal",
    };
    if (abi.jsSetProperty(ctx, result, "ndim", abi.jsInt32(ctx, @intCast(value.shape.rank()))) < 0 or
        abi.jsSetProperty(ctx, result, "dtype", abi.jsString(ctx, dtype_name)) < 0 or
        abi.jsSetProperty(ctx, result, "device", abi.jsString(ctx, device_name)) < 0)
    {
        abi.jsFreeValue(ctx, result);
        return errorValue(ctx, "failed to initialize Tensor object");
    }
    if (abi.jsSetFunction(ctx, result, "item", jsTensorItem, 0) < 0 or
        abi.jsSetFunction(ctx, result, "to_array", jsTensorToArray, 0) < 0 or
        abi.jsSetFunction(ctx, result, "toString", jsTensorToString, 0) < 0 or
        abi.jsSetFunction(ctx, result, "repr", jsTensorRepr, 1) < 0 or
        abi.jsSetFunction(ctx, result, "to", jsTensorTo, 1) < 0 or
        abi.jsSetFunction(ctx, result, "slice", jsTensorSlice, 1) < 0 or
        abi.jsSetFunction(ctx, result, "backward", jsTensorBackward, 0) < 0 or
        abi.jsSetFunction(ctx, result, "sum", jsTensorSum, 0) < 0 or
        abi.jsSetFunction(ctx, result, "mean", jsTensorMean, 0) < 0 or
        abi.jsSetFunction(ctx, result, "min", jsTensorMin, 0) < 0 or
        abi.jsSetFunction(ctx, result, "max", jsTensorMax, 0) < 0 or
        abi.jsSetFunction(ctx, result, "variance", jsTensorVariance, 0) < 0 or
        abi.jsSetFunction(ctx, result, "std", jsTensorStd, 0) < 0 or
        abi.jsSetFunction(ctx, result, "argmin", jsTensorArgmin, 0) < 0 or
        abi.jsSetFunction(ctx, result, "argmax", jsTensorArgmax, 0) < 0 or
        abi.jsSetGetter(ctx, result, "grad", jsTensorGrad) < 0 or
        abi.jsSetGetter(ctx, result, "grad_device", jsTensorGradDevice) < 0)
    {
        abi.jsFreeValue(ctx, result);
        return errorValue(ctx, "failed to initialize Tensor object");
    }
    return result;
}

fn tensorScalarValue(ctx: abi.JSContext, value: *const Tensor, index: usize) !abi.JSValue {
    const bytes = try value.storage.?.readableBytes();
    return switch (value.dtype) {
        .f32 => abi.jsFloat64(ctx, @floatCast(std.mem.bytesAsSlice(f32, bytes)[index])),
        .f64 => abi.jsFloat64(ctx, std.mem.bytesAsSlice(f64, bytes)[index]),
        .i64 => abi.jsFloat64(ctx, @floatFromInt(std.mem.bytesAsSlice(i64, bytes)[index])),
    };
}

fn tensorArrayValue(ctx: abi.JSContext, value: *const Tensor, depth: usize, flat_index: *usize) !abi.JSValue {
    if (depth == value.shape.rank()) {
        const scalar = try tensorScalarValue(ctx, value, flat_index.*);
        flat_index.* += 1;
        return scalar;
    }
    const array = abi.jsNewArray(ctx);
    if (abi.jsIsException(array)) return error.JavaScriptException;
    errdefer abi.jsFreeValue(ctx, array);
    for (0..value.shape.dims[depth]) |index| {
        const item = try tensorArrayValue(ctx, value, depth + 1, flat_index);
        if (abi.jsSetArrayElement(ctx, array, @intCast(index), item) < 0) return error.JavaScriptException;
    }
    return array;
}

fn jsTensorItem(ctx: abi.JSContext, this_value: abi.JSValueConst, argc: c_int, _: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    if (argc != 0) return typeError(ctx, "Tensor.item expects no arguments");
    const input = tensorFromValue(ctx, this_value) orelse return typeError(ctx, "item expects a Tensor");
    if (input.shape.numel() != 1) return typeError(ctx, "item requires a single-element Tensor");
    var contiguous: ?*Tensor = null;
    defer if (contiguous) |value| value.deinit();
    const source = if (input.layout.isContiguous(input.shape) and input.layout.offset == 0)
        input
    else blk: {
        const value = engine_api.Engine.init(allocator, .{}).contiguous(input) catch return errorValue(ctx, "failed to read Tensor");
        contiguous = value;
        break :blk value;
    };
    var host: ?*Tensor = null;
    defer if (host) |value| value.deinit();
    const readable = if (source.device() == .metal) blk: {
        const value = Tensor.createContiguous(allocator, source.shape.dims, source.dtype, .cpu, false) catch return errorValue(ctx, "failed to read Tensor");
        engine_api.Engine.init(allocator, .{}).copyInto(value, source) catch {
            value.deinit();
            return errorValue(ctx, "failed to read Tensor");
        };
        host = value;
        break :blk value;
    } else source;
    return tensorScalarValue(ctx, readable, 0) catch return errorValue(ctx, "failed to read Tensor");
}

fn jsTensorToArray(ctx: abi.JSContext, this_value: abi.JSValueConst, argc: c_int, _: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    if (argc != 0) return typeError(ctx, "Tensor.to_array expects no arguments");
    const input = tensorFromValue(ctx, this_value) orelse return typeError(ctx, "to_array expects a Tensor");
    var contiguous: ?*Tensor = null;
    defer if (contiguous) |value| value.deinit();
    const source = if (input.layout.isContiguous(input.shape) and input.layout.offset == 0)
        input
    else blk: {
        const value = engine_api.Engine.init(allocator, .{}).contiguous(input) catch return errorValue(ctx, "failed to read Tensor");
        contiguous = value;
        break :blk value;
    };
    var host: ?*Tensor = null;
    defer if (host) |value| value.deinit();
    const readable = if (source.device() == .metal) blk: {
        const value = Tensor.createContiguous(allocator, source.shape.dims, source.dtype, .cpu, false) catch return errorValue(ctx, "failed to read Tensor");
        engine_api.Engine.init(allocator, .{}).copyInto(value, source) catch {
            value.deinit();
            return errorValue(ctx, "failed to read Tensor");
        };
        host = value;
        break :blk value;
    } else source;
    var flat_index: usize = 0;
    return tensorArrayValue(ctx, readable, 0, &flat_index) catch return errorValue(ctx, "failed to read Tensor");
}

fn jsTensorSlice(ctx: abi.JSContext, this_value: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    if (argc != 1) return typeError(ctx, "Tensor.slice expects a selector array");
    var args: [2]abi.JSValueConst = .{ this_value, argv[0] };
    return jsSlice(ctx, this_value, 2, &args);
}

fn jsTensorToString(ctx: abi.JSContext, this_value: abi.JSValueConst, argc: c_int, _: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    if (argc != 0) return typeError(ctx, "Tensor.toString expects no arguments");
    const input = tensorFromValue(ctx, this_value) orelse return typeError(ctx, "toString expects a Tensor");
    const text = repr_common.formatCompact(input, true, autograd_types.State.fromTensor(input) != null) catch return errorValue(ctx, "failed to format Tensor");
    defer repr_common.alloc().free(text);
    return abi.jsString(ctx, text);
}

fn jsTensorRepr(ctx: abi.JSContext, this_value: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    const input = tensorFromValue(ctx, this_value) orelse return typeError(ctx, "repr expects a Tensor");
    var options = repr_common.defaultOpts();
    if (argc > 0 and !abi.jsIsUndefined(argv[0]) and !abi.jsIsNull(argv[0])) {
        const mode = abi.jsGetProperty(ctx, argv[0], "mode");
        defer abi.jsFreeValue(ctx, mode);
        if (abi.jsStringEquals(ctx, mode, "html")) options.mode = .html;
        const sparse = abi.jsGetProperty(ctx, argv[0], "sparse");
        defer abi.jsFreeValue(ctx, sparse);
        _ = abi.jsToBool(ctx, &options.sparse, sparse);
    }
    const text = switch (options.mode) {
        .text => repr_common.formatText(input, true, options),
        .html => repr_common.formatHtml(input, true, options),
    } catch return errorValue(ctx, "failed to format Tensor");
    defer repr_common.alloc().free(text);
    if (options.mode == .text) return abi.jsString(ctx, text);
    const object = abi.jsNewObject(ctx);
    if (abi.jsSetProperty(ctx, object, "mime", abi.jsString(ctx, "text/html")) < 0 or
        abi.jsSetProperty(ctx, object, "data", abi.jsString(ctx, text)) < 0)
        return errorValue(ctx, "failed to create Tensor representation");
    return object;
}

fn jsTensorTo(ctx: abi.JSContext, this_value: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    if (argc != 1) return typeError(ctx, "Tensor.to expects a device");
    const input = tensorFromValue(ctx, this_value) orelse return typeError(ctx, "to expects a Tensor");
    const target_device: engine_api.Device = if (abi.jsStringEquals(ctx, argv[0], "cpu"))
        .cpu
    else if (abi.jsStringEquals(ctx, argv[0], "metal"))
        .metal
    else
        return typeError(ctx, "device must be 'cpu' or 'metal'");
    const target = Tensor.createContiguous(allocator, input.shape.dims, input.dtype, target_device, false) catch return errorValue(ctx, "failed to create device Tensor");
    errdefer target.deinit();
    engine_api.Engine.init(allocator, .{}).copyInto(target, input) catch return errorValue(ctx, "failed to transfer Tensor");
    target.setAxesCopy(input.axes) catch return errorValue(ctx, "failed to transfer Tensor axes");
    switch (autograd_types.State.trackingState(input)) {
        .plain => {},
        .tracked => _ = autograd_types.State.create(allocator, target, false) catch return errorValue(ctx, "failed to preserve Tensor state"),
        .trainable => _ = autograd_types.State.create(allocator, target, true) catch return errorValue(ctx, "failed to preserve Tensor state"),
    }
    return createTensorObject(ctx, target);
}

fn jsTensor(ctx: abi.JSContext, _: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    if (argc < 1 or argc > 2) return typeError(ctx, "tensor expects data and optional options");
    var shape: std.ArrayList(usize) = .empty;
    defer shape.deinit(allocator);
    var values: std.ArrayList(f32) = .empty;
    defer values.deinit(allocator);
    flattenTensorInput(ctx, argv[0], 0, &shape, &values) catch |err| return switch (err) {
        error.InconsistentTensorShape => typeError(ctx, "tensor nested arrays must have a consistent shape"),
        error.InvalidTensorData => typeError(ctx, "tensor data must contain numbers"),
        error.OutOfMemory => errorValue(ctx, "out of memory"),
        else => errorValue(ctx, "failed to read tensor data"),
    };
    var dtype: engine_api.DType = .f32;
    var device: engine_api.Device = switch (config.getDefaultDevice()) {
        .cpu => .cpu,
        .metal => .metal,
    };
    if (argc == 2 and !abi.jsIsUndefined(argv[1])) {
        const dtype_value = abi.jsGetProperty(ctx, argv[1], "dtype");
        defer abi.jsFreeValue(ctx, dtype_value);
        if (!abi.jsIsUndefined(dtype_value)) {
            if (abi.jsStringEquals(ctx, dtype_value, "f64")) {
                dtype = .f64;
            } else if (abi.jsStringEquals(ctx, dtype_value, "i64")) {
                dtype = .i64;
            } else if (!abi.jsStringEquals(ctx, dtype_value, "f32")) {
                return typeError(ctx, "unsupported tensor dtype");
            }
        }
        const device_value = abi.jsGetProperty(ctx, argv[1], "device");
        defer abi.jsFreeValue(ctx, device_value);
        if (!abi.jsIsUndefined(device_value)) {
            if (abi.jsStringEquals(ctx, device_value, "metal")) {
                device = .metal;
            } else if (!abi.jsStringEquals(ctx, device_value, "cpu")) {
                return typeError(ctx, "device must be 'cpu' or 'metal'");
            }
        }
    }
    const axes = readAxesOption(ctx, if (argc == 2) argv[1] else abi.jsUndefined(ctx), shape.items.len) catch return typeError(ctx, "axes must be a string array whose length matches tensor rank");
    defer freeAxes(axes);
    const cpu_tensor = switch (dtype) {
        .f32 => Tensor.fromSliceF32(allocator, shape.items, values.items),
        .f64 => blk: {
            const converted = allocator.alloc(f64, values.items.len) catch return errorValue(ctx, "out of memory");
            defer allocator.free(converted);
            for (converted, values.items) |*output, input| output.* = input;
            break :blk Tensor.fromSliceF64(allocator, shape.items, converted);
        },
        .i64 => blk: {
            const converted = allocator.alloc(i64, values.items.len) catch return errorValue(ctx, "out of memory");
            defer allocator.free(converted);
            for (converted, values.items) |*output, input| output.* = @intFromFloat(input);
            break :blk Tensor.fromSliceI64(allocator, shape.items, converted);
        },
    } catch return errorValue(ctx, "failed to create Tensor");
    cpu_tensor.setAxesCopy(axes) catch return errorValue(ctx, "failed to set Tensor axes");
    if (device == .cpu) return createTensorObject(ctx, cpu_tensor);
    defer cpu_tensor.deinit();
    const tensor = Tensor.createContiguous(allocator, shape.items, dtype, device, false) catch return errorValue(ctx, "failed to create device Tensor");
    errdefer tensor.deinit();
    engine_api.Engine.init(allocator, .{}).copyInto(tensor, cpu_tensor) catch return errorValue(ctx, "failed to transfer Tensor");
    tensor.setAxesCopy(axes) catch return errorValue(ctx, "failed to set Tensor axes");
    return createTensorObject(ctx, tensor);
}

fn createFilledTensor(ctx: abi.JSContext, shape_value: abi.JSValueConst, options_value: abi.JSValueConst, fill: f64) abi.JSValue {
    const shape = readShape(ctx, shape_value, "shape") orelse return typeError(ctx, "shape must be an array of non-negative integers");
    defer allocator.free(shape);
    var dtype: engine_api.DType = .f32;
    var device: engine_api.Device = switch (config.getDefaultDevice()) {
        .cpu => .cpu,
        .metal => .metal,
    };
    if (!abi.jsIsUndefined(options_value)) {
        const dtype_value = abi.jsGetProperty(ctx, options_value, "dtype");
        defer abi.jsFreeValue(ctx, dtype_value);
        if (!abi.jsIsUndefined(dtype_value)) {
            if (abi.jsStringEquals(ctx, dtype_value, "f64")) dtype = .f64
            else if (abi.jsStringEquals(ctx, dtype_value, "i64")) dtype = .i64
            else if (!abi.jsStringEquals(ctx, dtype_value, "f32")) return typeError(ctx, "unsupported tensor dtype");
        }
        const device_value = abi.jsGetProperty(ctx, options_value, "device");
        defer abi.jsFreeValue(ctx, device_value);
        if (!abi.jsIsUndefined(device_value)) {
            if (abi.jsStringEquals(ctx, device_value, "metal")) device = .metal
            else if (abi.jsStringEquals(ctx, device_value, "cpu")) device = .cpu
            else return typeError(ctx, "device must be 'cpu' or 'metal'");
        }
    }
    const axes = readAxesOption(ctx, options_value, shape.len) catch return typeError(ctx, "axes must be a string array whose length matches tensor rank");
    defer freeAxes(axes);
    const cpu_tensor = Tensor.createContiguous(allocator, shape, dtype, .cpu, false) catch return errorValue(ctx, "failed to create Tensor");
    errdefer cpu_tensor.deinit();
    cpu_tensor.setAxesCopy(axes) catch return errorValue(ctx, "failed to set Tensor axes");
    const bytes = cpu_tensor.storage.?.writableBytes() catch return errorValue(ctx, "failed to write Tensor");
    switch (dtype) {
        .f32 => @memset(std.mem.bytesAsSlice(f32, bytes), @floatCast(fill)),
        .f64 => @memset(std.mem.bytesAsSlice(f64, bytes), fill),
        .i64 => @memset(std.mem.bytesAsSlice(i64, bytes), @intFromFloat(fill)),
    }
    if (device == .cpu) return createTensorObject(ctx, cpu_tensor);
    defer cpu_tensor.deinit();
    const tensor = Tensor.createContiguous(allocator, shape, dtype, device, false) catch return errorValue(ctx, "failed to create device Tensor");
    errdefer tensor.deinit();
    engine_api.Engine.init(allocator, .{}).copyInto(tensor, cpu_tensor) catch return errorValue(ctx, "failed to transfer Tensor");
    tensor.setAxesCopy(axes) catch return errorValue(ctx, "failed to set Tensor axes");
    return createTensorObject(ctx, tensor);
}

fn jsEmpty(ctx: abi.JSContext, _: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    if (argc < 1 or argc > 2) return typeError(ctx, "empty expects a shape array and optional options");
    const shape = readShape(ctx, argv[0], "shape") orelse return typeError(ctx, "shape must be an array of non-negative integers");
    defer allocator.free(shape);
    return createFilledTensor(ctx, argv[0], if (argc == 2) argv[1] else abi.jsUndefined(ctx), 0);
}

fn jsZeros(ctx: abi.JSContext, _: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    if (argc < 1 or argc > 2) return typeError(ctx, "zeros expects a shape array and optional options");
    return createFilledTensor(ctx, argv[0], if (argc == 2) argv[1] else abi.jsUndefined(ctx), 0);
}

fn jsOnes(ctx: abi.JSContext, _: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    if (argc < 1 or argc > 2) return typeError(ctx, "ones expects a shape array and optional options");
    return createFilledTensor(ctx, argv[0], if (argc == 2) argv[1] else abi.jsUndefined(ctx), 1);
}

fn jsFull(ctx: abi.JSContext, _: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    if (argc < 2 or argc > 3) return typeError(ctx, "full expects a shape array, fill value, and optional options");
    var fill: f64 = 0;
    if (abi.jsToFloat64(ctx, &fill, argv[1]) < 0) return typeError(ctx, "fill value must be a number");
    return createFilledTensor(ctx, argv[0], if (argc == 3) argv[2] else abi.jsUndefined(ctx), fill);
}

fn jsParameter(ctx: abi.JSContext, _: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    if (argc < 1 or argc > 2) return typeError(ctx, "parameter expects a shape array and optional options");
    const shape = readShape(ctx, argv[0], "shape") orelse return typeError(ctx, "shape must be an array of non-negative integers");
    defer allocator.free(shape);
    var dtype: engine_api.DType = .f32;
    if (argc == 2 and !abi.jsIsUndefined(argv[1])) {
        const dtype_value = abi.jsGetProperty(ctx, argv[1], "dtype");
        defer abi.jsFreeValue(ctx, dtype_value);
        if (!abi.jsIsUndefined(dtype_value)) {
            if (abi.jsStringEquals(ctx, dtype_value, "f64")) {
                dtype = .f64;
            } else if (!abi.jsStringEquals(ctx, dtype_value, "f32")) {
                return typeError(ctx, "parameters require f32 or f64 dtype");
            }
        }
    }
    var device: engine_api.Device = switch (config.getDefaultDevice()) {
        .cpu => .cpu,
        .metal => .metal,
    };
    if (argc == 2 and !abi.jsIsUndefined(argv[1])) {
        const device_value = abi.jsGetProperty(ctx, argv[1], "device");
        defer abi.jsFreeValue(ctx, device_value);
        if (!abi.jsIsUndefined(device_value)) {
            if (abi.jsStringEquals(ctx, device_value, "metal")) {
                device = .metal;
            } else if (!abi.jsStringEquals(ctx, device_value, "cpu")) {
                return typeError(ctx, "device must be 'cpu' or 'metal'");
            }
        }
    }
    const axes = readAxesOption(ctx, if (argc == 2) argv[1] else abi.jsUndefined(ctx), shape.len) catch return typeError(ctx, "axes must be a string array whose length matches parameter rank");
    defer freeAxes(axes);
    const tensor = Tensor.createContiguous(allocator, shape, dtype, device, false) catch return errorValue(ctx, "failed to create Parameter");
    tensor.setAxesCopy(axes) catch return errorValue(ctx, "failed to set Parameter axes");
    _ = autograd_types.State.create(allocator, tensor, true) catch {
        tensor.deinit();
        return errorValue(ctx, "failed to create Parameter state");
    };
    const object = createTensorObject(ctx, tensor);
    if (abi.jsIsException(object)) return object;
    const metadata = abi.jsNewObject(ctx);
    if (abi.jsIsException(metadata) or abi.jsSetProperty(ctx, metadata, "role", abi.jsString(ctx, "parameter")) < 0 or
        abi.jsSetHiddenProperty(ctx, object, "$compute", metadata) < 0)
    {
        abi.jsFreeValue(ctx, object);
        return errorValue(ctx, "failed to initialize Parameter metadata");
    }
    return object;
}

fn jsSetDevice(ctx: abi.JSContext, _: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    if (argc != 1) return typeError(ctx, "setDevice expects 'cpu' or 'metal'");
    if (abi.jsStringEquals(ctx, argv[0], "cpu")) {
        config.setDefaultDevice(.cpu);
        return abi.jsUndefined(ctx);
    }
    if (abi.jsStringEquals(ctx, argv[0], "metal")) {
        if (!metal_backend.isAvailable()) return errorValue(ctx, "setDevice('metal'): metal is unavailable");
        config.setDefaultDevice(.metal);
        return abi.jsUndefined(ctx);
    }
    return typeError(ctx, "setDevice expects 'cpu' or 'metal'");
}

fn jsCopy(ctx: abi.JSContext, _: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    if (argc != 2) return typeError(ctx, "copy expects a target and source Tensor");
    const target = tensorFromValue(ctx, argv[0]) orelse return typeError(ctx, "copy expects a target Tensor");
    const source = tensorFromValue(ctx, argv[1]) orelse return typeError(ctx, "copy expects a source Tensor");
    engine_api.Engine.init(allocator, .{}).copyInto(target, source) catch |err| return switch (err) {
        error.ShapeMismatch => typeError(ctx, "copy requires tensors with the same number of elements"),
        else => errorValue(ctx, "copy failed"),
    };
    return abi.jsDupValue(ctx, argv[0]);
}

fn jsInternalMuladd(ctx: abi.JSContext, _: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    if (argc != 3) return typeError(ctx, "$muladd_ expects a target, scale, and addend Tensor");
    const target = tensorFromValue(ctx, argv[0]) orelse return typeError(ctx, "$muladd_ expects a target Tensor");
    const addend = tensorFromValue(ctx, argv[2]) orelse return typeError(ctx, "$muladd_ expects an addend Tensor");
    if (target.dtype != addend.dtype or target.device() != addend.device()) return typeError(ctx, "$muladd_ requires matching dtype and device");
    var scale: f64 = 0;
    if (abi.jsToFloat64(ctx, &scale, argv[1]) < 0) return typeError(ctx, "$muladd_ expects a numeric scale");
    const target_storage = target.storage orelse return errorValue(ctx, "$muladd_ target has no storage");
    const addend_storage = addend.storage orelse return errorValue(ctx, "$muladd_ addend has no storage");
    dispatch.update(target.device() orelse .cpu, .muladd, target.dtype, target_storage, addend_storage, scale) catch return errorValue(ctx, "$muladd_ failed");
    return abi.jsUndefined(ctx);
}

fn jsGrad(ctx: abi.JSContext, _: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    if (argc != 2) return typeError(ctx, "grad expects a loss Tensor and parameter array");
    const loss = tensorFromValue(ctx, argv[0]) orelse return typeError(ctx, "grad expects a loss Tensor");
    const params = readTensorList(ctx, argv[1], "parameters") orelse return typeError(ctx, "grad expects an array of Tensors");
    defer allocator.free(params);
    const previous_grad_mode = grad_mode.isEnabled();
    grad_mode.setEnabled(false);
    defer grad_mode.setEnabled(previous_grad_mode);
    var derived = autograd_compose.buildFromLossTensor(loss) catch return errorValue(ctx, "failed to build gradient graph");
    defer derived.deinit(allocator);
    autograd_compose.executeForTensor(loss, &derived) catch return errorValue(ctx, "failed to execute gradient graph");
    for (params, 0..) |parameter, index| {
        const item = abi.jsGetArrayElement(ctx, argv[1], @intCast(index));
        defer abi.jsFreeValue(ctx, item);
        const state = autograd_types.State.fromTensor(parameter) orelse continue;
        const gradient = state.gradient orelse continue;
        const copy = autograd_execution.cloneTensor(allocator, gradient) catch return errorValue(ctx, "failed to copy gradient");
        const object = createTensorObject(ctx, copy);
        if (abi.jsIsException(object)) return object;
        if (abi.jsSetProperty(ctx, item, "grad", object) < 0) return errorValue(ctx, "failed to attach gradient");
    }
    return abi.jsUndefined(ctx);
}

fn jsClipGradNorm(ctx: abi.JSContext, _: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    if (argc < 2 or argc > 3) return typeError(ctx, "clip_grad_norm expects parameters and max_norm");
    const parameters = readTensorList(ctx, argv[0], "parameters") orelse return typeError(ctx, "clip_grad_norm expects an array of Tensors");
    defer allocator.free(parameters);
    var max_norm: f64 = 0;
    if (abi.jsToFloat64(ctx, &max_norm, argv[1]) < 0 or !std.math.isFinite(max_norm) or max_norm <= 0) return typeError(ctx, "clip_grad_norm max_norm must be positive");
    var eps: f64 = 1e-6;
    if (argc == 3 and abi.jsToFloat64(ctx, &eps, argv[2]) < 0) return typeError(ctx, "clip_grad_norm eps must be a number");

    var sum_squared: f64 = 0;
    for (parameters) |parameter| {
        const state = autograd_types.State.fromTensor(parameter) orelse continue;
        const gradient = state.gradient orelse continue;
        const bytes = gradient.storage orelse continue;
        const readable = bytes.readableBytes() catch return errorValue(ctx, "clip_grad_norm failed to read gradient");
        switch (gradient.dtype) {
            .f32 => for (std.mem.bytesAsSlice(f32, readable)) |value| {
                sum_squared += @as(f64, value) * @as(f64, value);
            },
            .f64 => for (std.mem.bytesAsSlice(f64, readable)) |value| {
                sum_squared += value * value;
            },
            .i64 => return typeError(ctx, "clip_grad_norm requires differentiable gradients"),
        }
    }
    const total_norm = @sqrt(sum_squared);
    if (!std.math.isFinite(total_norm) or total_norm <= max_norm) return abi.jsFloat64(ctx, total_norm);
    const scale = max_norm / (total_norm + eps);
    for (parameters) |parameter| {
        const state = autograd_types.State.fromTensor(parameter) orelse continue;
        const gradient = state.gradient orelse continue;
        const storage = gradient.storage orelse continue;
        const writable = storage.writableBytes() catch return errorValue(ctx, "clip_grad_norm failed to update gradient");
        switch (gradient.dtype) {
            .f32 => for (std.mem.bytesAsSlice(f32, writable)) |*value| {
                value.* = @floatCast(@as(f64, value.*) * scale);
            },
            .f64 => for (std.mem.bytesAsSlice(f64, writable)) |*value| {
                value.* *= scale;
            },
            .i64 => unreachable,
        }
    }
    return abi.jsFloat64(ctx, total_norm);
}

fn jsClearGrad(ctx: abi.JSContext, _: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    if (argc != 1) return typeError(ctx, "clear_grad expects a parameter array");
    const params = readTensorList(ctx, argv[0], "parameters") orelse return typeError(ctx, "clear_grad expects an array of Tensors");
    defer allocator.free(params);
    for (params) |parameter| {
        if (autograd_types.State.fromTensor(parameter)) |state| {
            if (state.gradient) |gradient| {
                gradient.deinit();
                state.gradient = null;
            }
        }
    }
    return abi.jsUndefined(ctx);
}

fn jsNoGrad(ctx: abi.JSContext, _: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    if (argc < 1) return typeError(ctx, "no_grad expects a callback");
    const previous = grad_mode.isEnabled();
    grad_mode.setEnabled(false);
    defer grad_mode.setEnabled(previous);
    return abi.jsCall(ctx, argv[0], abi.jsUndefined(ctx), &.{});
}

fn jsRandom(ctx: abi.JSContext, shape_value: abi.JSValueConst, normal: bool) abi.JSValue {
    const shape = readShape(ctx, shape_value, "shape") orelse return typeError(ctx, "shape must be an array of non-negative integers");
    defer allocator.free(shape);
    const tensor = createRandomTensor(shape, .f32, normal) catch return errorValue(ctx, "failed to create Tensor");
    return createTensorObject(ctx, tensor);
}

fn parseDTypeOption(ctx: abi.JSContext, options: abi.JSValueConst, default: engine_api.DType) !engine_api.DType {
    if (abi.jsIsUndefined(options) or abi.jsIsNull(options)) return default;
    const dtype = abi.jsGetProperty(ctx, options, "dtype");
    defer abi.jsFreeValue(ctx, dtype);
    if (abi.jsIsUndefined(dtype)) return default;
    if (abi.jsStringEquals(ctx, dtype, "f32")) return .f32;
    if (abi.jsStringEquals(ctx, dtype, "f64")) return .f64;
    if (abi.jsStringEquals(ctx, dtype, "i64")) return .i64;
    return error.InvalidDType;
}

fn createRandomTensor(shape: []const usize, dtype: engine_api.DType, normal: bool) !*Tensor {
    if (dtype == .i64) return error.UnsupportedDType;
    const tensor = try Tensor.createContiguous(allocator, shape, dtype, .cpu, false);
    errdefer tensor.deinit();
    const bytes = try tensor.storage.?.writableBytes();
    var random = random_state.random();
    switch (dtype) {
        .f32 => for (std.mem.bytesAsSlice(f32, bytes)) |*value| {
            value.* = if (normal) @floatCast(boxMuller(&random)) else random.float(f32);
        },
        .f64 => for (std.mem.bytesAsSlice(f64, bytes)) |*value| {
            value.* = if (normal) boxMuller(&random) else random.float(f64);
        },
        .i64 => unreachable,
    }
    return tensor;
}

fn boxMuller(random: *std.Random) f64 {
    const r1 = @max(random.float(f64), 1e-12);
    const r2 = random.float(f64);
    return @sqrt(-2.0 * @log(r1)) * @cos(2.0 * std.math.pi * r2);
}

fn jsRand(ctx: abi.JSContext, _: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    if (argc < 1 or argc > 2) return typeError(ctx, "rand expects a shape array and optional options");
    const shape = readShape(ctx, argv[0], "shape") orelse return typeError(ctx, "shape must be an array of non-negative integers");
    defer allocator.free(shape);
    const dtype = parseDTypeOption(ctx, if (argc == 2) argv[1] else abi.jsUndefined(ctx), .f32) catch return typeError(ctx, "unsupported random tensor dtype");
    const tensor = createRandomTensor(shape, dtype, false) catch return errorValue(ctx, "failed to create Tensor");
    const axes = readAxesOption(ctx, if (argc == 2) argv[1] else abi.jsUndefined(ctx), shape.len) catch {
        tensor.deinit();
        return typeError(ctx, "axes must be a string array whose length matches tensor rank");
    };
    defer freeAxes(axes);
    tensor.setAxesCopy(axes) catch {
        tensor.deinit();
        return errorValue(ctx, "failed to set Tensor axes");
    };
    return createTensorObject(ctx, tensor);
}

fn jsRandn(ctx: abi.JSContext, _: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    if (argc < 1 or argc > 2) return typeError(ctx, "randn expects a shape array and optional options");
    const shape = readShape(ctx, argv[0], "shape") orelse return typeError(ctx, "shape must be an array of non-negative integers");
    defer allocator.free(shape);
    const dtype = parseDTypeOption(ctx, if (argc == 2) argv[1] else abi.jsUndefined(ctx), .f32) catch return typeError(ctx, "unsupported random tensor dtype");
    const tensor = createRandomTensor(shape, dtype, true) catch return errorValue(ctx, "failed to create Tensor");
    const axes = readAxesOption(ctx, if (argc == 2) argv[1] else abi.jsUndefined(ctx), shape.len) catch {
        tensor.deinit();
        return typeError(ctx, "axes must be a string array whose length matches tensor rank");
    };
    defer freeAxes(axes);
    tensor.setAxesCopy(axes) catch {
        tensor.deinit();
        return errorValue(ctx, "failed to set Tensor axes");
    };
    return createTensorObject(ctx, tensor);
}

fn jsSeed(ctx: abi.JSContext, _: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    if (argc != 1) return typeError(ctx, "seed expects one integer");
    var seed: i32 = 0;
    if (abi.jsToInt32(ctx, &seed, argv[0]) < 0) return typeError(ctx, "seed expects one integer");
    random_state = std.Random.DefaultPrng.init(@intCast(seed));
    return abi.jsUndefined(ctx);
}

fn jsArange(ctx: abi.JSContext, _: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    if (argc < 1 or argc > 3) return typeError(ctx, "arange expects start, optional end, and optional step");
    var first: f64 = 0;
    var last: f64 = 0;
    var step: f64 = 1;
    if (argc == 1) {
        if (abi.jsToFloat64(ctx, &last, argv[0]) < 0) return typeError(ctx, "arange arguments must be numbers");
    } else {
        if (abi.jsToFloat64(ctx, &first, argv[0]) < 0 or abi.jsToFloat64(ctx, &last, argv[1]) < 0) return typeError(ctx, "arange arguments must be numbers");
        if (argc == 3 and abi.jsToFloat64(ctx, &step, argv[2]) < 0) return typeError(ctx, "arange step must be a number");
    }
    if (step == 0) return typeError(ctx, "arange step cannot be zero");
    const count_float = @ceil((last - first) / step);
    if (count_float < 0 or count_float > @as(f64, @floatFromInt(std.math.maxInt(usize)))) return typeError(ctx, "invalid arange range");
    const values = allocator.alloc(f32, @intFromFloat(count_float)) catch return errorValue(ctx, "out of memory");
    defer allocator.free(values);
    for (values, 0..) |*value, index| value.* = @floatCast(first + @as(f64, @floatFromInt(index)) * step);
    const tensor = Tensor.fromSliceF32(allocator, &.{values.len}, values) catch return errorValue(ctx, "failed to create Tensor");
    return createTensorObject(ctx, tensor);
}

fn jsLinspace(ctx: abi.JSContext, _: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    if (argc < 2 or argc > 4) return typeError(ctx, "linspace expects start, end, optional steps, and options");
    var first: f64 = 0;
    var last: f64 = 0;
    if (abi.jsToFloat64(ctx, &first, argv[0]) < 0 or abi.jsToFloat64(ctx, &last, argv[1]) < 0) return typeError(ctx, "linspace bounds must be numbers");
    var steps: i32 = 100;
    if (argc == 3 and abi.jsToInt32(ctx, &steps, argv[2]) < 0) return typeError(ctx, "linspace steps must be an integer");
    if (argc == 4 and abi.jsToInt32(ctx, &steps, argv[2]) < 0) return typeError(ctx, "linspace steps must be an integer");
    if (steps < 0) return typeError(ctx, "linspace steps must be non-negative");
    const options = if (argc == 4) argv[3] else abi.jsUndefined(ctx);
    const dtype = parseDTypeOption(ctx, options, .f32) catch return typeError(ctx, "unsupported linspace dtype");
    const tensor = Tensor.createContiguous(allocator, &.{@intCast(steps)}, dtype, .cpu, false) catch return errorValue(ctx, "failed to create Tensor");
    errdefer tensor.deinit();
    const bytes = tensor.storage.?.writableBytes() catch return errorValue(ctx, "failed to create Tensor");
    const denom: f64 = if (steps > 1) @floatFromInt(steps - 1) else 1;
    switch (dtype) {
        .f32 => for (std.mem.bytesAsSlice(f32, bytes), 0..) |*value, index| {
            value.* = @floatCast(first + (last - first) * (@as(f64, @floatFromInt(index)) / denom));
        },
        .f64 => for (std.mem.bytesAsSlice(f64, bytes), 0..) |*value, index| {
            value.* = first + (last - first) * (@as(f64, @floatFromInt(index)) / denom);
        },
        .i64 => for (std.mem.bytesAsSlice(i64, bytes), 0..) |*value, index| {
            value.* = @intFromFloat(first + (last - first) * (@as(f64, @floatFromInt(index)) / denom));
        },
    }
    const axes = readAxesOption(ctx, options, 1) catch return typeError(ctx, "axes must be a string array whose length matches linspace rank");
    defer freeAxes(axes);
    tensor.setAxesCopy(axes) catch return errorValue(ctx, "failed to set Tensor axes");
    return createTensorObject(ctx, tensor);
}

fn jsCast(ctx: abi.JSContext, _: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    if (argc != 2) return typeError(ctx, "cast expects a Tensor and dtype");
    const input = tensorFromValue(ctx, argv[0]) orelse return typeError(ctx, "cast expects a Tensor");
    var dtype: engine_api.DType = .f32;
    if (abi.jsStringEquals(ctx, argv[1], "f64")) dtype = .f64 else if (abi.jsStringEquals(ctx, argv[1], "i64")) dtype = .i64 else if (!abi.jsStringEquals(ctx, argv[1], "f32")) return typeError(ctx, "unsupported tensor dtype");
    const result = engine_api.Engine.init(allocator, .{}).cast(input, dtype) catch return errorValue(ctx, "cast failed");
    trackResult(result, &.{input}, .cast, null, null, null) catch {
        result.deinit();
        return errorValue(ctx, "failed to record autograd state");
    };
    const object = createTensorObject(ctx, result);
    if (abi.jsIsException(object)) return object;
    return object;
}

fn jsSquare(ctx: abi.JSContext, _: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    return binary(ctx, argc, argv, .mul, "square failed");
}

fn jsGtScalar(ctx: abi.JSContext, _: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    if (argc != 2) return typeError(ctx, "gt_scalar expects a Tensor and threshold");
    const input = tensorFromValue(ctx, argv[0]) orelse return typeError(ctx, "gt_scalar expects a Tensor");
    var threshold: f64 = 0;
    if (abi.jsToFloat64(ctx, &threshold, argv[1]) < 0) return typeError(ctx, "threshold must be a number");
    const scalar = Tensor.createContiguous(allocator, &.{}, input.dtype, input.device() orelse .cpu, false) catch return errorValue(ctx, "gt_scalar failed");
    errdefer scalar.deinit();
    switch (input.dtype) {
        .f32 => {
            const value: f32 = @floatCast(threshold);
            scalar.storage.?.writeFromHost(std.mem.asBytes(&value)) catch return errorValue(ctx, "gt_scalar failed");
        },
        .f64 => scalar.storage.?.writeFromHost(std.mem.asBytes(&threshold)) catch return errorValue(ctx, "gt_scalar failed"),
        .i64 => {
            const value: i64 = @intFromFloat(threshold);
            scalar.storage.?.writeFromHost(std.mem.asBytes(&value)) catch return errorValue(ctx, "gt_scalar failed");
        },
    }
    const result = engine_api.Engine.init(allocator, .{}).gt(input, scalar) catch return errorValue(ctx, "gt_scalar failed");
    scalar.deinit();
    return createTensorObject(ctx, result);
}

fn jsWhere(ctx: abi.JSContext, _: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    if (argc != 3) return typeError(ctx, "where expects condition, true, and false tensors");
    const condition = tensorFromValue(ctx, argv[0]) orelse return typeError(ctx, "where expects Tensor values");
    const on_true = tensorFromValue(ctx, argv[1]) orelse return typeError(ctx, "where expects Tensor values");
    const on_false = tensorFromValue(ctx, argv[2]) orelse return typeError(ctx, "where expects Tensor values");
    const result = engine_api.Engine.init(allocator, .{}).whereSelect(condition, on_true, on_false) catch return errorValue(ctx, "where failed");
    trackResult(result, &.{ condition, on_true, on_false }, .where, null, null, null) catch {
        result.deinit();
        return errorValue(ctx, "failed to record autograd state");
    };
    const object = createTensorObject(ctx, result);
    if (abi.jsIsException(object)) return object;
    return object;
}

fn jsMaskedFill(ctx: abi.JSContext, _: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    if (argc != 3) return typeError(ctx, "masked_fill expects input, mask, and value");
    const input = tensorFromValue(ctx, argv[0]) orelse return typeError(ctx, "masked_fill expects Tensor values");
    const raw_mask = tensorFromValue(ctx, argv[1]) orelse return typeError(ctx, "masked_fill expects Tensor values");
    const mask = normalizeMaskTensor(raw_mask) catch return errorValue(ctx, "masked_fill failed: InvalidArgument");
    defer if (mask.owned) mask.tensor.deinit();
    var fill: f64 = 0;
    if (abi.jsToFloat64(ctx, &fill, argv[2]) < 0) return typeError(ctx, "masked_fill value must be a number");
    const result = engine_api.Engine.init(allocator, .{}).maskedFill(input, mask.tensor, fill) catch return errorValue(ctx, "masked_fill failed");
    trackResult(result, &.{ input, mask.tensor }, .masked_fill, null, fill, null) catch {
        result.deinit();
        return errorValue(ctx, "failed to record autograd state");
    };
    const object = createTensorObject(ctx, result);
    if (abi.jsIsException(object)) return object;
    return object;
}

fn jsCrossEntropyIndexed(ctx: abi.JSContext, _: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    if (argc < 2 or argc > 3) return typeError(ctx, "cross_entropy_indexed expects logits, targets, and optional axis");
    const logits = tensorFromValue(ctx, argv[0]) orelse return typeError(ctx, "cross_entropy_indexed expects Tensor values");
    const targets = tensorFromValue(ctx, argv[1]) orelse return typeError(ctx, "cross_entropy_indexed expects Tensor values");
    const axis = if (argc == 3) integerArgument(ctx, argv[2], "axis") orelse return typeError(ctx, "axis must be a non-negative integer") else 1;
    const result = engine_api.Engine.init(allocator, .{}).crossEntropyIndexed(logits, targets, axis) catch return errorValue(ctx, "cross_entropy_indexed failed");
    trackResultWithSavedInputs(result, &.{logits}, &.{ logits, targets }, .cross_entropy_indexed, axis, null, false, null, null) catch {
        result.deinit();
        return errorValue(ctx, "failed to record autograd state");
    };
    return createTensorObject(ctx, result);
}

fn jsAdd(ctx: abi.JSContext, _: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    return binary(ctx, argc, argv, .add, "add");
}

const BinaryOperation = enum { add, sub, mul, div, dot, matmul };

fn binary(ctx: abi.JSContext, argc: c_int, argv: [*c]abi.JSValueConst, operation: BinaryOperation, name: [*:0]const u8) abi.JSValue {
    if (argc != 2) return typeError(ctx, "binary operation expects two tensors");
    const lhs = tensorFromValue(ctx, argv[0]) orelse return typeError(ctx, "add expects Tensor values");
    const rhs = tensorFromValue(ctx, argv[1]) orelse return typeError(ctx, "add expects Tensor values");
    const engine = engine_api.Engine.init(allocator, .{});
    const result = switch (operation) {
        .add => engine.add(lhs, rhs),
        .sub => engine.sub(lhs, rhs),
        .mul => engine.mul(lhs, rhs),
        .div => engine.div(lhs, rhs),
        .dot => engine.dot(lhs, rhs),
        .matmul => engine.matmul(lhs, rhs),
    } catch return errorValue(ctx, name);
    const op_tag: OpTag = switch (operation) {
        .add => .add,
        .sub => .sub,
        .mul => .mul,
        .div => .div,
        .dot => .dot,
        .matmul => .matmul,
    };
    trackResult(result, &.{ lhs, rhs }, op_tag, null, null, null) catch {
        result.deinit();
        return errorValue(ctx, "failed to record autograd state");
    };
    return createTensorObject(ctx, result);
}

fn jsSub(ctx: abi.JSContext, _: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    return binary(ctx, argc, argv, .sub, "sub failed");
}

fn jsMul(ctx: abi.JSContext, _: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    return binary(ctx, argc, argv, .mul, "mul failed");
}

fn jsDiv(ctx: abi.JSContext, _: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    return binary(ctx, argc, argv, .div, "div failed");
}

fn jsMatmul(ctx: abi.JSContext, _: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    if (argc < 2 or argc > 3) return typeError(ctx, "matmul expects two Tensors and optional execution options");
    const lhs = tensorFromValue(ctx, argv[0]) orelse return typeError(ctx, "matmul expects a Tensor");
    const rhs = tensorFromValue(ctx, argv[1]) orelse return typeError(ctx, "matmul expects a Tensor");
    var metadata = execution_metadata.ExecutionMetadata{};
    if (argc == 3 and !abi.jsIsUndefined(argv[2])) {
        const hint = abi.jsGetProperty(ctx, argv[2], "hint");
        defer abi.jsFreeValue(ctx, hint);
        if (!abi.jsStringEquals(ctx, hint, "projection") and !abi.jsStringEquals(ctx, hint, "attention_scores") and !abi.jsStringEquals(ctx, hint, "attention_values")) {
            return typeError(ctx, "matmul execution hint must be one of projection, attention_scores, attention_values");
        }
        metadata.matmul_hint = if (abi.jsStringEquals(ctx, hint, "projection")) .projection else if (abi.jsStringEquals(ctx, hint, "attention_scores")) .attention_scores else .attention_values;
        const source = abi.jsGetProperty(ctx, argv[2], "source");
        defer abi.jsFreeValue(ctx, source);
        if (!abi.jsIsUndefined(source)) {
            if (!abi.jsStringEquals(ctx, source, "higher_level_module")) return typeError(ctx, "matmul execution source must be higher_level_module");
            metadata.hint_source = .higher_level_module;
        } else metadata.hint_source = .api_execution_arg;
    }
    const op = Op.initWithExecutionMetadata(.matmul, &.{ lhs, rhs }, .{ .none = {} }, metadata) catch return errorValue(ctx, "matmul failed");
    var result = engine_api.Engine.init(allocator, .{}).executeRaw(op) catch return errorValue(ctx, "matmul failed");
    errdefer result.deinit();
    trackResult(result.primary, &.{ lhs, rhs }, .matmul, null, null, null) catch return errorValue(ctx, "failed to record autograd state");
    const output = result.primary;
    result.primary = undefined;
    if (result.secondary) |secondary| secondary.deinit();
    result.secondary = null;
    return createTensorObject(ctx, output);
}

fn integerArgument(ctx: abi.JSContext, value: abi.JSValueConst, message: [*:0]const u8) ?usize {
    var integer: i32 = 0;
    if (abi.jsToInt32(ctx, &integer, value) < 0 or integer < 0) return null;
    _ = message;
    return @intCast(integer);
}

fn readShape(ctx: abi.JSContext, value: abi.JSValueConst, message: [*:0]const u8) ?[]usize {
    if (!abi.jsIsArray(ctx, value)) return null;
    const length_value = abi.jsGetProperty(ctx, value, "length");
    defer abi.jsFreeValue(ctx, length_value);
    const length = integerArgument(ctx, length_value, message) orelse return null;
    const shape = allocator.alloc(usize, length) catch return null;
    errdefer allocator.free(shape);
    for (shape, 0..) |*dimension, index| {
        const item = abi.jsGetArrayElement(ctx, value, @intCast(index));
        defer abi.jsFreeValue(ctx, item);
        dimension.* = integerArgument(ctx, item, message) orelse return null;
    }
    return shape;
}

fn readTensorList(ctx: abi.JSContext, value: abi.JSValueConst, message: [*:0]const u8) ?[]*Tensor {
    if (!abi.jsIsArray(ctx, value)) return null;
    const length_value = abi.jsGetProperty(ctx, value, "length");
    defer abi.jsFreeValue(ctx, length_value);
    const length = integerArgument(ctx, length_value, message) orelse return null;
    const tensors = allocator.alloc(*Tensor, length) catch return null;
    errdefer allocator.free(tensors);
    for (tensors, 0..) |*tensor, index| {
        const item = abi.jsGetArrayElement(ctx, value, @intCast(index));
        defer abi.jsFreeValue(ctx, item);
        tensor.* = tensorFromValue(ctx, item) orelse return null;
    }
    return tensors;
}

fn jsDot(ctx: abi.JSContext, _: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    return binary(ctx, argc, argv, .dot, "dot failed");
}

fn jsClamp(ctx: abi.JSContext, _: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    if (argc != 3) return typeError(ctx, "clamp expects a Tensor, min, and max");
    const input = tensorFromValue(ctx, argv[0]) orelse return typeError(ctx, "clamp expects a Tensor");
    var minimum: f64 = 0;
    var maximum: f64 = 0;
    if (abi.jsToFloat64(ctx, &minimum, argv[1]) < 0 or abi.jsToFloat64(ctx, &maximum, argv[2]) < 0) return typeError(ctx, "clamp bounds must be numbers");
    const result = engine_api.Engine.init(allocator, .{}).clamp(input, minimum, maximum) catch return errorValue(ctx, "clamp failed");
    trackResult(result, &.{input}, .clamp, null, minimum, maximum) catch {
        result.deinit();
        return errorValue(ctx, "failed to record autograd state");
    };
    return createTensorObject(ctx, result);
}

const Reduction = enum { sum, mean, min, max, variance, std, argmin, argmax };

fn reduction(ctx: abi.JSContext, argc: c_int, argv: [*c]abi.JSValueConst, operation: Reduction, name: [*:0]const u8) abi.JSValue {
    if (argc < 1 or argc > 3) return typeError(ctx, "reduction expects a Tensor, optional axis, and optional keepdim");
    const input = tensorFromValue(ctx, argv[0]) orelse return typeError(ctx, name);
    const axis = if (argc > 1 and !abi.jsIsUndefined(argv[1]) and !abi.jsIsNull(argv[1])) integerArgument(ctx, argv[1], "axis") orelse return typeError(ctx, "axis must be a non-negative integer") else null;
    if (axis) |value| if (value >= input.shape.rank()) return errorValue(ctx, "reduction axis is out of bounds");
    var keepdim = false;
    if (argc > 2 and !abi.jsIsUndefined(argv[2]) and !abi.jsIsNull(argv[2])) {
        if (abi.jsToBool(ctx, &keepdim, argv[2]) < 0) return typeError(ctx, "keepdim must be boolean");
    }
    const engine = engine_api.Engine.init(allocator, .{});
    const result = switch (operation) {
        .sum => engine.reduce(.sum, input, axis, keepdim),
        .mean => engine.reduce(.mean, input, axis, keepdim),
        .min => engine.reduce(.min, input, axis, keepdim),
        .max => engine.reduce(.max, input, axis, keepdim),
        .variance => engine.reduce(.variance, input, axis, keepdim),
        .std => engine.reduce(.std, input, axis, keepdim),
        .argmin => engine.reduce(.argmin, input, axis, keepdim),
        .argmax => engine.reduce(.argmax, input, axis, keepdim),
    } catch return errorValue(ctx, name);
    const op_tag: OpTag = switch (operation) {
        .sum => if (axis == null) .sum_all else .sum_axis,
        .mean => if (axis == null) .mean_all else .mean_axis,
        .min => if (axis == null) .min_all else .min_axis,
        .max => if (axis == null) .max_all else .max_axis,
        .variance => if (axis == null) .variance_all else .variance_axis,
        .std => if (axis == null) .std_all else .std_axis,
        .argmin => if (axis == null) .argmin_all else .argmin_axis,
        .argmax => if (axis == null) .argmax_all else .argmax_axis,
    };
    trackResultWithSavedInputs(result, &.{input}, &.{input}, op_tag, axis, null, keepdim, null, null) catch {
        result.deinit();
        return errorValue(ctx, "failed to record autograd state");
    };
    return createTensorObject(ctx, result);
}

fn instanceReduction(ctx: abi.JSContext, this_value: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst, operation: Reduction, name: [*:0]const u8) abi.JSValue {
    if (argc > 2) return typeError(ctx, "reduction expects an optional axis and keepdim");
    var args: [3]abi.JSValueConst = undefined;
    args[0] = this_value;
    for (0..@intCast(argc)) |index| args[index + 1] = argv[index];
    return reduction(ctx, argc + 1, &args, operation, name);
}

fn jsTensorSum(ctx: abi.JSContext, this_value: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue { return instanceReduction(ctx, this_value, argc, argv, .sum, "sum failed"); }
fn jsTensorMean(ctx: abi.JSContext, this_value: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue { return instanceReduction(ctx, this_value, argc, argv, .mean, "mean failed"); }
fn jsTensorMin(ctx: abi.JSContext, this_value: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue { return instanceReduction(ctx, this_value, argc, argv, .min, "min failed"); }
fn jsTensorMax(ctx: abi.JSContext, this_value: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue { return instanceReduction(ctx, this_value, argc, argv, .max, "max failed"); }
fn jsTensorVariance(ctx: abi.JSContext, this_value: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue { return instanceReduction(ctx, this_value, argc, argv, .variance, "variance failed"); }
fn jsTensorStd(ctx: abi.JSContext, this_value: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue { return instanceReduction(ctx, this_value, argc, argv, .std, "std failed"); }
fn jsTensorArgmin(ctx: abi.JSContext, this_value: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue { return instanceReduction(ctx, this_value, argc, argv, .argmin, "argmin failed"); }
fn jsTensorArgmax(ctx: abi.JSContext, this_value: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue { return instanceReduction(ctx, this_value, argc, argv, .argmax, "argmax failed"); }

fn jsMean(ctx: abi.JSContext, _: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    return reduction(ctx, argc, argv, .mean, "mean expects a Tensor");
}

fn jsMin(ctx: abi.JSContext, _: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    return reduction(ctx, argc, argv, .min, "min expects a Tensor");
}

fn jsMax(ctx: abi.JSContext, _: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    return reduction(ctx, argc, argv, .max, "max expects a Tensor");
}

fn jsVariance(ctx: abi.JSContext, _: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    return reduction(ctx, argc, argv, .variance, "variance expects a Tensor");
}

fn jsStd(ctx: abi.JSContext, _: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    return reduction(ctx, argc, argv, .std, "std expects a Tensor");
}

fn jsArgmin(ctx: abi.JSContext, _: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    return reduction(ctx, argc, argv, .argmin, "argmin expects a Tensor");
}

fn jsArgmax(ctx: abi.JSContext, _: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    return reduction(ctx, argc, argv, .argmax, "argmax expects a Tensor");
}

fn jsSoftmax(ctx: abi.JSContext, _: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    if (argc != 2) return typeError(ctx, "softmax expects a Tensor and axis");
    const input = tensorFromValue(ctx, argv[0]) orelse return typeError(ctx, "softmax expects a Tensor");
    const axis = integerArgument(ctx, argv[1], "axis") orelse return typeError(ctx, "axis must be a non-negative integer");
    const result = engine_api.Engine.init(allocator, .{}).softmax(input, axis) catch return errorValue(ctx, "softmax failed");
    trackResult(result, &.{input}, .softmax, axis, null, null) catch {
        result.deinit();
        return errorValue(ctx, "failed to record autograd state");
    };
    return createTensorObject(ctx, result);
}

fn jsReshape(ctx: abi.JSContext, _: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    if (argc != 2) return typeError(ctx, "reshape expects a Tensor and shape");
    const input = tensorFromValue(ctx, argv[0]) orelse return typeError(ctx, "reshape expects a Tensor");
    const shape = readShape(ctx, argv[1], "shape") orelse return typeError(ctx, "shape must be an array of non-negative integers");
    defer allocator.free(shape);
    const result = engine_api.Engine.init(allocator, .{}).reshape(input, shape) catch return errorValue(ctx, "reshape failed");
    trackResult(result, &.{input}, .reshape, null, null, null) catch {
        result.deinit();
        return errorValue(ctx, "failed to record autograd state");
    };
    return createTensorObject(ctx, result);
}

fn normalizeIndex(index: i64, dimension: usize) ?usize {
    const dim: i64 = @intCast(dimension);
    const normalized = if (index < 0) dim + index else index;
    if (normalized < 0 or normalized >= dim) return null;
    return @intCast(normalized);
}

fn normalizeSliceBound(index: i64, dimension: usize) ?usize {
    const dim: i64 = @intCast(dimension);
    const normalized = if (index < 0) dim + index else index;
    if (normalized < 0 or normalized > dim) return null;
    return @intCast(normalized);
}

fn parseSliceRange(ctx: abi.JSContext, value: abi.JSValueConst, dimension: usize) ?SliceRange {
    if (abi.jsIsNumber(value)) {
        var index: i32 = 0;
        if (abi.jsToInt32(ctx, &index, value) < 0) return null;
        const start = normalizeIndex(index, dimension) orelse return null;
        return .{ .start = start, .stop = start + 1 };
    }
    const text = abi.jsStringAlloc(ctx, value, allocator) catch return null;
    defer allocator.free(text);
    var parts = std.mem.splitScalar(u8, text, ':');
    const start_text = parts.next() orelse return null;
    const stop_text = parts.next();
    if (stop_text == null) {
        const index = std.fmt.parseInt(i64, start_text, 10) catch return null;
        const normalized = normalizeIndex(index, dimension) orelse return null;
        return .{ .start = normalized, .stop = normalized + 1, .step = 1 };
    }
    const step_text = parts.next();
    if (parts.next() != null) return null;
    const start: i64 = if (start_text.len == 0) 0 else std.fmt.parseInt(i64, start_text, 10) catch return null;
    const stop_part = stop_text.?;
    const stop: i64 = if (stop_part.len == 0) @intCast(dimension) else std.fmt.parseInt(i64, stop_part, 10) catch return null;
    const step: i64 = if (step_text) |text_part| if (text_part.len == 0) 1 else std.fmt.parseInt(i64, text_part, 10) catch return null else 1;
    if (step <= 0) return null;
    return .{
        .start = normalizeSliceBound(start, dimension) orelse return null,
        .stop = normalizeSliceBound(stop, dimension) orelse return null,
        .step = @intCast(step),
    };
}

fn jsSlice(ctx: abi.JSContext, _: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    if (argc != 2 or !abi.jsIsArray(ctx, argv[1])) return typeError(ctx, "slice expects a Tensor and selector array");
    const input = tensorFromValue(ctx, argv[0]) orelse return typeError(ctx, "slice expects a Tensor");
    const selector_length_value = abi.jsGetProperty(ctx, argv[1], "length");
    defer abi.jsFreeValue(ctx, selector_length_value);
    const selector_length = integerArgument(ctx, selector_length_value, "selector length") orelse return typeError(ctx, "slice selectors must be an array");
    if (selector_length > input.shape.rank()) return typeError(ctx, "slice has too many selectors");

    var ranges: std.ArrayList(SliceRange) = .empty;
    defer ranges.deinit(allocator);
    var reverse_axes: std.ArrayList(usize) = .empty;
    defer reverse_axes.deinit(allocator);
    for (0..input.shape.rank()) |axis| {
        const selector = if (axis < selector_length) abi.jsGetArrayElement(ctx, argv[1], @intCast(axis)) else abi.jsString(ctx, ":");
        defer if (axis < selector_length) abi.jsFreeValue(ctx, selector);
        if (abi.jsStringEquals(ctx, selector, "::-1")) {
            ranges.append(allocator, .{ .start = 0, .stop = input.shape.dims[axis], .step = 1 }) catch return errorValue(ctx, "out of memory");
            reverse_axes.append(allocator, axis) catch return errorValue(ctx, "out of memory");
            continue;
        }
        const range = parseSliceRange(ctx, selector, input.shape.dims[axis]) orelse {
            if (abi.jsIsNumber(selector)) return errorValue(ctx, "slice index out of bounds");
            return typeError(ctx, "slice selector is invalid");
        };
        ranges.append(allocator, range) catch return errorValue(ctx, "out of memory");
    }
    var result = engine_api.Engine.init(allocator, .{}).slice(input, ranges.items) catch return errorValue(ctx, "slice failed");
    errdefer result.deinit();
    for (reverse_axes.items) |axis| {
        const axis_size = result.shape.dims[axis];
        const indices = Tensor.createContiguous(allocator, &.{axis_size}, .i64, result.device() orelse .cpu, false) catch return errorValue(ctx, "slice reverse failed");
        defer indices.deinit();
        const host_indices = allocator.alloc(i64, axis_size) catch return errorValue(ctx, "slice failed");
        defer allocator.free(host_indices);
        for (host_indices, 0..) |*index, position| index.* = @intCast(axis_size - 1 - position);
        indices.storage.?.writeFromHost(std.mem.sliceAsBytes(host_indices)) catch return errorValue(ctx, "slice failed");
        const reversed = engine_api.Engine.init(allocator, .{}).indexSelect(result, axis, indices) catch return errorValue(ctx, "slice failed");
        result.deinit();
        result = reversed;
    }
    trackSliceResult(result, input, ranges.items) catch return errorValue(ctx, "failed to record autograd state");
    const object = createTensorObject(ctx, result);
    if (abi.jsIsException(object)) return object;
    return object;
}

fn jsContiguous(ctx: abi.JSContext, _: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    if (argc != 1) return typeError(ctx, "contiguous expects one Tensor");
    const input = tensorFromValue(ctx, argv[0]) orelse return typeError(ctx, "contiguous expects a Tensor");
    const result = engine_api.Engine.init(allocator, .{}).contiguous(input) catch return errorValue(ctx, "contiguous failed");
    trackResult(result, &.{input}, .contiguous, null, null, null) catch {
        result.deinit();
        return errorValue(ctx, "failed to record autograd state");
    };
    return createTensorObject(ctx, result);
}

fn jsPermute(ctx: abi.JSContext, _: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    if (argc != 2) return typeError(ctx, "permute expects a Tensor and axes");
    const input = tensorFromValue(ctx, argv[0]) orelse return typeError(ctx, "permute expects a Tensor");
    const axes = readShape(ctx, argv[1], "axes") orelse return typeError(ctx, "axes must be an array of non-negative integers");
    defer allocator.free(axes);
    const result = engine_api.Engine.init(allocator, .{}).permute(input, axes) catch return errorValue(ctx, "permute failed");
    trackResultWithSavedInputs(result, &.{input}, &.{input}, .permute, null, axes, false, null, null) catch {
        result.deinit();
        return errorValue(ctx, "failed to record autograd state");
    };
    return createTensorObject(ctx, result);
}

fn jsTranspose(ctx: abi.JSContext, _: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    if (argc != 3) return typeError(ctx, "transpose expects a Tensor and two axes");
    const input = tensorFromValue(ctx, argv[0]) orelse return typeError(ctx, "transpose expects a Tensor");
    const axis_a = integerArgument(ctx, argv[1], "axis") orelse return typeError(ctx, "axis must be a non-negative integer");
    const axis_b = integerArgument(ctx, argv[2], "axis") orelse return typeError(ctx, "axis must be a non-negative integer");
    const result = engine_api.Engine.init(allocator, .{}).transpose(input, axis_a, axis_b) catch return errorValue(ctx, "transpose failed");
    trackResult(result, &.{input}, .transpose, null, null, null) catch {
        result.deinit();
        return errorValue(ctx, "failed to record autograd state");
    };
    return createTensorObject(ctx, result);
}

fn jsSqueeze(ctx: abi.JSContext, _: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    if (argc < 1 or argc > 2) return typeError(ctx, "squeeze expects a Tensor and optional axis");
    const input = tensorFromValue(ctx, argv[0]) orelse return typeError(ctx, "squeeze expects a Tensor");
    const axis = if (argc == 2) integerArgument(ctx, argv[1], "axis") orelse return typeError(ctx, "axis must be a non-negative integer") else null;
    const result = engine_api.Engine.init(allocator, .{}).squeeze(input, axis) catch return errorValue(ctx, "squeeze failed");
    trackResult(result, &.{input}, .squeeze, axis, null, null) catch {
        result.deinit();
        return errorValue(ctx, "failed to record autograd state");
    };
    return createTensorObject(ctx, result);
}

fn jsUnsqueeze(ctx: abi.JSContext, _: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    if (argc != 2) return typeError(ctx, "unsqueeze expects a Tensor and axis");
    const input = tensorFromValue(ctx, argv[0]) orelse return typeError(ctx, "unsqueeze expects a Tensor");
    const axis = integerArgument(ctx, argv[1], "axis") orelse return typeError(ctx, "axis must be a non-negative integer");
    const result = engine_api.Engine.init(allocator, .{}).unsqueeze(input, axis) catch return errorValue(ctx, "unsqueeze failed");
    trackResult(result, &.{input}, .unsqueeze, axis, null, null) catch {
        result.deinit();
        return errorValue(ctx, "failed to record autograd state");
    };
    return createTensorObject(ctx, result);
}

fn jsCat(ctx: abi.JSContext, _: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    if (argc < 1 or argc > 2) return typeError(ctx, "cat expects a Tensor array and optional axis");
    const tensors = readTensorList(ctx, argv[0], "tensors") orelse return typeError(ctx, "cat expects an array of Tensors");
    defer allocator.free(tensors);
    const axis = if (argc == 2) integerArgument(ctx, argv[1], "axis") orelse return typeError(ctx, "axis must be a non-negative integer") else 0;
    const result = engine_api.Engine.init(allocator, .{}).cat(tensors, axis) catch |err| return switch (err) {
        error.ShapeMismatch => errorValue(ctx, "cat() failed: shape mismatch"),
        error.DeviceMismatch => errorValue(ctx, "cat() failed: DeviceMismatch"),
        else => errorValue(ctx, "cat() failed"),
    };
    trackResult(result, tensors, .cat, axis, null, null) catch {
        result.deinit();
        return errorValue(ctx, "failed to record autograd state");
    };
    return createTensorObject(ctx, result);
}

fn jsStack(ctx: abi.JSContext, _: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    if (argc < 1 or argc > 2) return typeError(ctx, "stack expects a Tensor array and optional axis");
    const tensors = readTensorList(ctx, argv[0], "tensors") orelse return typeError(ctx, "stack expects an array of Tensors");
    defer allocator.free(tensors);
    const axis = if (argc == 2) integerArgument(ctx, argv[1], "axis") orelse return typeError(ctx, "axis must be a non-negative integer") else 0;
    const result = engine_api.Engine.init(allocator, .{}).stack(tensors, axis) catch |err| return switch (err) {
        error.ShapeMismatch => errorValue(ctx, "stack() failed: shape mismatch"),
        error.DeviceMismatch => errorValue(ctx, "stack() failed: DeviceMismatch"),
        else => errorValue(ctx, "stack() failed"),
    };
    trackResult(result, tensors, .stack, axis, null, null) catch {
        result.deinit();
        return errorValue(ctx, "failed to record autograd state");
    };
    return createTensorObject(ctx, result);
}

const NormalizedIndex = struct {
    tensor: *Tensor,
    owned: bool,
};

fn normalizeIndexTensor(input: *Tensor) !NormalizedIndex {
    if (input.dtype == .i64) return .{ .tensor = input, .owned = false };
    if (input.dtype != .f32 and input.dtype != .f64) return error.InvalidArgument;
    var host_input: ?*Tensor = null;
    defer if (host_input) |value| value.deinit();
    const source_tensor = if (input.device() == .metal) blk: {
        const value = try Tensor.createContiguous(allocator, input.shape.dims, input.dtype, .cpu, false);
        errdefer value.deinit();
        engine_api.Engine.init(allocator, .{}).copyInto(value, input) catch {
            value.deinit();
            return error.DeviceTransfer;
        };
        host_input = value;
        break :blk value;
    } else input;
    const output_cpu = try Tensor.createContiguous(allocator, input.shape.dims, .i64, .cpu, false);
    errdefer output_cpu.deinit();
    const source = try source_tensor.storage.?.readableBytes();
    const destination = try output_cpu.storage.?.writableBytes();
    switch (input.dtype) {
        .f32 => for (std.mem.bytesAsSlice(f32, source), std.mem.bytesAsSlice(i64, destination)) |value, *index| {
            if (!std.math.isFinite(value) or @trunc(value) != value or value >= 9223372036854775808.0 or value < -9223372036854775808.0) return error.InvalidArgument;
            index.* = @intFromFloat(@as(f64, value));
        },
        .f64 => for (std.mem.bytesAsSlice(f64, source), std.mem.bytesAsSlice(i64, destination)) |value, *index| {
            if (!std.math.isFinite(value) or @trunc(value) != value or value >= 9223372036854775808.0 or value < -9223372036854775808.0) return error.InvalidArgument;
            index.* = @intFromFloat(value);
        },
        .i64 => unreachable,
    }
    if (input.device() != .metal) return .{ .tensor = output_cpu, .owned = true };
    const output = try Tensor.createContiguous(allocator, input.shape.dims, .i64, .metal, false);
    errdefer output.deinit();
    engine_api.Engine.init(allocator, .{}).copyInto(output, output_cpu) catch return error.DeviceTransfer;
    output_cpu.deinit();
    return .{ .tensor = output, .owned = true };
}

fn validateIndexBounds(index: *Tensor, limit: usize) !void {
    var host_index: ?*Tensor = null;
    defer if (host_index) |value| value.deinit();
    const source = if (index.device() == .metal) blk: {
        const value = try Tensor.createContiguous(allocator, index.shape.dims, .i64, .cpu, false);
        errdefer value.deinit();
        engine_api.Engine.init(allocator, .{}).copyInto(value, index) catch {
            value.deinit();
            return error.DeviceTransfer;
        };
        host_index = value;
        break :blk value;
    } else index;
    const bytes = try source.storage.?.readableBytes();
    for (std.mem.bytesAsSlice(i64, bytes)) |value| {
        if (value < 0 or value >= @as(i64, @intCast(limit))) return error.IndexOutOfBounds;
    }
}

fn normalizeMaskTensor(input: *Tensor) !NormalizedIndex {
    if (input.dtype == .i64) return .{ .tensor = input, .owned = false };
    if (input.dtype != .f32 and input.dtype != .f64) return error.InvalidArgument;
    return .{ .tensor = try engine_api.Engine.init(allocator, .{}).cast(input, .i64), .owned = true };
}

fn jsOneHot(ctx: abi.JSContext, _: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    if (argc != 2) return typeError(ctx, "one_hot expects indices and class count");
    const raw_input = tensorFromValue(ctx, argv[0]) orelse return typeError(ctx, "one_hot expects a Tensor");
    const classes = integerArgument(ctx, argv[1], "class count") orelse return typeError(ctx, "class count must be a non-negative integer");
    const input = normalizeIndexTensor(raw_input) catch return errorValue(ctx, "one_hot failed: InvalidArgument");
    defer if (input.owned) input.tensor.deinit();
    validateIndexBounds(input.tensor, classes) catch return errorValue(ctx, "one_hot failed: IndexOutOfBounds");
    const result = engine_api.Engine.init(allocator, .{}).oneHot(input.tensor, classes) catch return errorValue(ctx, "one_hot failed");
    return createTensorObject(ctx, result);
}

fn jsGather(ctx: abi.JSContext, _: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    if (argc != 3) return typeError(ctx, "gather expects input, axis, and index");
    const input = tensorFromValue(ctx, argv[0]) orelse return typeError(ctx, "gather expects a Tensor input");
    const axis = integerArgument(ctx, argv[1], "axis") orelse return typeError(ctx, "axis must be a non-negative integer");
    const raw_index = tensorFromValue(ctx, argv[2]) orelse return typeError(ctx, "gather expects a Tensor index");
    if (axis >= input.shape.rank()) return errorValue(ctx, "gather failed: InvalidAxis");
    const index = normalizeIndexTensor(raw_index) catch return errorValue(ctx, "gather failed: InvalidArgument");
    defer if (index.owned) index.tensor.deinit();
    validateIndexBounds(index.tensor, input.shape.dims[axis]) catch return errorValue(ctx, "gather failed: IndexOutOfBounds");
    const result = engine_api.Engine.init(allocator, .{}).gather(input, axis, index.tensor) catch return errorValue(ctx, "gather failed");
    trackResult(result, &.{ input, index.tensor }, .gather, axis, null, null) catch {
        result.deinit();
        return errorValue(ctx, "failed to record autograd state");
    };
    return createTensorObject(ctx, result);
}

fn jsIndexSelect(ctx: abi.JSContext, _: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    if (argc != 3) return typeError(ctx, "index_select expects input, axis, and index");
    const input = tensorFromValue(ctx, argv[0]) orelse return typeError(ctx, "index_select expects a Tensor input");
    const axis = integerArgument(ctx, argv[1], "axis") orelse return typeError(ctx, "axis must be a non-negative integer");
    const raw_index = tensorFromValue(ctx, argv[2]) orelse return typeError(ctx, "index_select expects a Tensor index");
    if (axis >= input.shape.rank()) return errorValue(ctx, "index_select failed: InvalidAxis");
    const index = normalizeIndexTensor(raw_index) catch return errorValue(ctx, "index_select failed: InvalidArgument");
    defer if (index.owned) index.tensor.deinit();
    validateIndexBounds(index.tensor, input.shape.dims[axis]) catch return errorValue(ctx, "index_select failed: IndexOutOfBounds");
    const result = engine_api.Engine.init(allocator, .{}).indexSelect(input, axis, index.tensor) catch return errorValue(ctx, "index_select failed");
    trackResult(result, &.{ input, index.tensor }, .index_select, axis, null, null) catch {
        result.deinit();
        return errorValue(ctx, "failed to record autograd state");
    };
    return createTensorObject(ctx, result);
}

fn jsTopk(ctx: abi.JSContext, _: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    if (argc < 2 or argc > 3) return typeError(ctx, "topk expects a Tensor, k, and optional axis");
    const input = tensorFromValue(ctx, argv[0]) orelse return typeError(ctx, "topk expects a Tensor");
    const k = integerArgument(ctx, argv[1], "k") orelse return typeError(ctx, "k must be a positive integer");
    const axis = if (argc == 3) integerArgument(ctx, argv[2], "axis") orelse return typeError(ctx, "axis must be a non-negative integer") else 0;
    if (axis >= input.shape.rank()) return errorValue(ctx, "topk failed: InvalidAxis");
    if (k == 0 or k > input.shape.dims[axis]) return errorValue(ctx, "topk failed: InvalidTopK");
    var outputs = engine_api.Engine.init(allocator, .{}).topK(input, k, axis) catch return errorValue(ctx, "topk failed");
    const indices_for_tracking = outputs.secondary orelse {
        outputs.deinit();
        return errorValue(ctx, "topk did not produce indices");
    };
    trackTopKResult(outputs.primary, indices_for_tracking, input, axis) catch {
        outputs.deinit();
        return errorValue(ctx, "failed to record autograd state");
    };
    const values = createTensorObject(ctx, outputs.primary);
    if (abi.jsIsException(values)) {
        outputs.primary = undefined;
        outputs.deinit();
        return values;
    }
    outputs.primary = undefined;
    const secondary = outputs.secondary orelse {
        abi.jsFreeValue(ctx, values);
        return errorValue(ctx, "topk did not produce indices");
    };
    const indices = createTensorObject(ctx, secondary);
    outputs.secondary = null;
    if (abi.jsIsException(indices)) {
        abi.jsFreeValue(ctx, values);
        return indices;
    }
    const result = abi.jsNewObject(ctx);
    if (abi.jsIsException(result)) {
        abi.jsFreeValue(ctx, values);
        abi.jsFreeValue(ctx, indices);
        return result;
    }
    if (abi.jsSetProperty(ctx, result, "values", values) < 0) {
        abi.jsFreeValue(ctx, result);
        abi.jsFreeValue(ctx, indices);
        return errorValue(ctx, "failed to initialize topk result");
    }
    if (abi.jsSetProperty(ctx, result, "indices", indices) < 0) {
        abi.jsFreeValue(ctx, result);
        abi.jsFreeValue(ctx, indices);
        return errorValue(ctx, "failed to initialize topk result");
    }
    return result;
}

const UnaryOperation = enum { abs, exp, log, neg, sqrt, sign, relu, sigmoid, silu, tanh, gelu };

fn unary(ctx: abi.JSContext, argc: c_int, argv: [*c]abi.JSValueConst, operation: UnaryOperation, name: [*:0]const u8) abi.JSValue {
    if (argc != 1) return typeError(ctx, "unary operation expects one tensor");
    const input = tensorFromValue(ctx, argv[0]) orelse return typeError(ctx, name);
    const engine = engine_api.Engine.init(allocator, .{});
    const result = switch (operation) {
        .abs => engine.abs(input),
        .exp => engine.exp(input),
        .log => engine.log(input),
        .neg => engine.neg(input),
        .sqrt => engine.sqrt(input),
        .sign => engine.sign(input),
        .relu => engine.relu(input),
        .sigmoid => engine.sigmoid(input),
        .silu => engine.silu(input),
        .tanh => engine.tanh(input),
        .gelu => engine.gelu(input),
    } catch return errorValue(ctx, name);
    const op_tag: OpTag = switch (operation) {
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
    };
    trackResult(result, &.{input}, op_tag, null, null, null) catch {
        result.deinit();
        return errorValue(ctx, "failed to record autograd state");
    };
    return createTensorObject(ctx, result);
}

fn jsAbs(ctx: abi.JSContext, _: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    return unary(ctx, argc, argv, .abs, "abs expects a Tensor");
}

fn jsExp(ctx: abi.JSContext, _: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    return unary(ctx, argc, argv, .exp, "exp expects a Tensor");
}

fn jsLog(ctx: abi.JSContext, _: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    return unary(ctx, argc, argv, .log, "log expects a Tensor");
}

fn jsNeg(ctx: abi.JSContext, _: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    return unary(ctx, argc, argv, .neg, "neg expects a Tensor");
}

fn jsSqrt(ctx: abi.JSContext, _: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    return unary(ctx, argc, argv, .sqrt, "sqrt expects a Tensor");
}

fn jsSign(ctx: abi.JSContext, _: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    return unary(ctx, argc, argv, .sign, "sign expects a Tensor");
}

fn jsRelu(ctx: abi.JSContext, _: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    return unary(ctx, argc, argv, .relu, "relu expects a Tensor");
}

fn jsSigmoid(ctx: abi.JSContext, _: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    return unary(ctx, argc, argv, .sigmoid, "sigmoid expects a Tensor");
}

fn jsSilu(ctx: abi.JSContext, _: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    return unary(ctx, argc, argv, .silu, "silu expects a Tensor");
}

fn jsTanh(ctx: abi.JSContext, _: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    return unary(ctx, argc, argv, .tanh, "tanh expects a Tensor");
}

fn jsGelu(ctx: abi.JSContext, _: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    return unary(ctx, argc, argv, .gelu, "gelu expects a Tensor");
}

fn jsSum(ctx: abi.JSContext, _: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    return reduction(ctx, argc, argv, .sum, "sum expects a Tensor");
}

const functions = [_]abi.JSFunction{
    .{ .name = "$create_compiled_executable_native", .callback = jsCreateCompiledExecutable, .length = 1 },
    .{ .name = "tensor", .callback = jsTensor, .length = 1 },
    .{ .name = "empty", .callback = jsEmpty, .length = 1 },
    .{ .name = "zeros", .callback = jsZeros, .length = 1 },
    .{ .name = "ones", .callback = jsOnes, .length = 1 },
    .{ .name = "full", .callback = jsFull, .length = 2 },
    .{ .name = "parameter", .callback = jsParameter, .length = 1 },
    .{ .name = "setDevice", .callback = jsSetDevice, .length = 1 },
    .{ .name = "copy", .callback = jsCopy, .length = 2 },
    .{ .name = "$muladd_", .callback = jsInternalMuladd, .length = 3 },
    .{ .name = "grad", .callback = jsGrad, .length = 2 },
    .{ .name = "clip_grad_norm", .callback = jsClipGradNorm, .length = 2 },
    .{ .name = "clear_grad", .callback = jsClearGrad, .length = 1 },
    .{ .name = "no_grad", .callback = jsNoGrad, .length = 1 },
    .{ .name = "rand", .callback = jsRand, .length = 1 },
    .{ .name = "randn", .callback = jsRandn, .length = 1 },
    .{ .name = "seed", .callback = jsSeed, .length = 1 },
    .{ .name = "arange", .callback = jsArange, .length = 1 },
    .{ .name = "linspace", .callback = jsLinspace, .length = 2 },
    .{ .name = "add", .callback = jsAdd, .length = 2 },
    .{ .name = "sub", .callback = jsSub, .length = 2 },
    .{ .name = "mul", .callback = jsMul, .length = 2 },
    .{ .name = "div", .callback = jsDiv, .length = 2 },
    .{ .name = "matmul", .callback = jsMatmul, .length = 2 },
    .{ .name = "dot", .callback = jsDot, .length = 2 },
    .{ .name = "square", .callback = jsSquare, .length = 1 },
    .{ .name = "gt_scalar", .callback = jsGtScalar, .length = 2 },
    .{ .name = "cast", .callback = jsCast, .length = 2 },
    .{ .name = "where", .callback = jsWhere, .length = 3 },
    .{ .name = "masked_fill", .callback = jsMaskedFill, .length = 3 },
    .{ .name = "cross_entropy_indexed", .callback = jsCrossEntropyIndexed, .length = 2 },
    .{ .name = "abs", .callback = jsAbs, .length = 1 },
    .{ .name = "exp", .callback = jsExp, .length = 1 },
    .{ .name = "log", .callback = jsLog, .length = 1 },
    .{ .name = "neg", .callback = jsNeg, .length = 1 },
    .{ .name = "sqrt", .callback = jsSqrt, .length = 1 },
    .{ .name = "sign", .callback = jsSign, .length = 1 },
    .{ .name = "relu", .callback = jsRelu, .length = 1 },
    .{ .name = "sigmoid", .callback = jsSigmoid, .length = 1 },
    .{ .name = "silu", .callback = jsSilu, .length = 1 },
    .{ .name = "tanh", .callback = jsTanh, .length = 1 },
    .{ .name = "gelu", .callback = jsGelu, .length = 1 },
    .{ .name = "clamp", .callback = jsClamp, .length = 3 },
    .{ .name = "softmax", .callback = jsSoftmax, .length = 2 },
    .{ .name = "sum", .callback = jsSum, .length = 1 },
    .{ .name = "mean", .callback = jsMean, .length = 1 },
    .{ .name = "min", .callback = jsMin, .length = 1 },
    .{ .name = "max", .callback = jsMax, .length = 1 },
    .{ .name = "variance", .callback = jsVariance, .length = 1 },
    .{ .name = "std", .callback = jsStd, .length = 1 },
    .{ .name = "argmin", .callback = jsArgmin, .length = 1 },
    .{ .name = "argmax", .callback = jsArgmax, .length = 1 },
    .{ .name = "reshape", .callback = jsReshape, .length = 2 },
    .{ .name = "slice", .callback = jsSlice, .length = 2 },
    .{ .name = "contiguous", .callback = jsContiguous, .length = 1 },
    .{ .name = "permute", .callback = jsPermute, .length = 2 },
    .{ .name = "transpose", .callback = jsTranspose, .length = 3 },
    .{ .name = "squeeze", .callback = jsSqueeze, .length = 1 },
    .{ .name = "unsqueeze", .callback = jsUnsqueeze, .length = 2 },
    .{ .name = "cat", .callback = jsCat, .length = 2 },
    .{ .name = "stack", .callback = jsStack, .length = 2 },
    .{ .name = "one_hot", .callback = jsOneHot, .length = 2 },
    .{ .name = "gather", .callback = jsGather, .length = 3 },
    .{ .name = "index_select", .callback = jsIndexSelect, .length = 3 },
    .{ .name = "topk", .callback = jsTopk, .length = 2 },
};
const function_ptrs = blk: {
    var pointers: [functions.len]*const abi.JSFunction = undefined;
    for (&functions, 0..) |*function, index| pointers[index] = function;
    break :blk pointers;
};

pub const specifier: [:0]const u8 = "affon:compute/native";

pub fn load(ctx: ?*anyopaque, module_name: [*c]const u8) ?*anyopaque {
    return @ptrCast(abi.createJSFunctionModule(allocator, @ptrCast(ctx), module_name, &function_ptrs));
}
