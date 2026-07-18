const std = @import("std");
const hao = @import("hao");
const engine_api = @import("../../compute/engine.zig");
const autograd_types = @import("../../compute/types/autograd.zig");
const autograd_execution = @import("../../compute/execution/autograd.zig");
const autograd_compose = @import("../../compute/compose/derive.zig");
const OpTag = @import("../../compute/types/operation/tag.zig").OpTag;
const SliceRange = @import("../../compute/types/operation/options.zig").SliceRange;
const grad_mode = @import("../../compute/grad_mode.zig");
const captured_schema = @import("captured/schema.zig");
const captured_lowering = @import("captured/lowering.zig");

const abi = hao.js.abi;
const allocator = std.heap.c_allocator;
const Tensor = engine_api.Tensor;
var random_state = std.Random.DefaultPrng.init(0xA66F_0001);

const tensor_type_id: u32 = 1;
const compiled_executable_type_id: u32 = 2;
var compiled_executable_class_id: u32 = 0;
var compiled_executable_runtime: ?*anyopaque = null;

const TensorObject = struct { value: *Tensor };
const CompiledExecutable = struct { json: []u8 };

fn compiledExecutableFromValue(ctx: abi.JSContext, value: abi.JSValueConst) ?*CompiledExecutable {
    if (compiled_executable_class_id == 0) return null;
    const handle = abi.jsHostObjectHandleWithClass(ctx, value, compiled_executable_class_id, compiled_executable_type_id) orelse return null;
    return @ptrFromInt(@as(usize, @intCast(handle)));
}

fn compiledExecutableFinalizer(_: u32, handle: u64) callconv(.c) void {
    const executable: *CompiledExecutable = @ptrFromInt(@as(usize, @intCast(handle)));
    allocator.free(executable.json);
    allocator.destroy(executable);
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
    return result;
}

fn jsCompiledExecutableSummary(ctx: abi.JSContext, this_value: abi.JSValueConst, _: c_int, _: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    const executable = compiledExecutableFromValue(ctx, this_value) orelse return typeError(ctx, "invalid compiled executable");
    var parsed = parseCapturedProgram(executable) catch return errorValue(ctx, "failed to parse captured program");
    defer parsed.deinit();
    const result = abi.jsNewObject(ctx);
    if (abi.jsSetProperty(ctx, result, "inputCount", abi.jsInt32(ctx, @intCast(parsed.value.inputArity))) < 0 or
        abi.jsSetProperty(ctx, result, "nodeCount", abi.jsInt32(ctx, @intCast(parsed.value.nodes.len))) < 0)
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
    var lowered = captured_lowering.lowerToGraph(allocator, parsed.value, inputs, parseCapturedDType, createCapturedScalar, createCapturedRandom) catch return errorValue(ctx, "failed to lower captured graph");
    defer lowered.deinit(allocator);
    var result = engine_api.Engine.init(allocator, .{}).executeGraph(&lowered.graph, lowered.input_values.items) catch return errorValue(ctx, "failed to execute captured graph");
    if (result.outputs.len != 1) {
        result.deinit();
        return errorValue(ctx, "captured graph produced an unsupported output count");
    }
    const output = result.outputs[0];
    result.outputs[0] = undefined;
    result.owned[lowered.graph.outputs.items[0]] = false;
    lowered.detached_value = output;
    result.deinit();
    return createTensorObject(ctx, output);
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
        .{ .name = "recordPlanOutcome", .callback = jsCompiledExecutableNoop, .length = 4 },
        .{ .name = "recordFallback", .callback = jsCompiledExecutableNoop, .length = 3 },
        .{ .name = "recordSpecializationCapture", .callback = jsCompiledExecutableNoop, .length = 1 },
        .{ .name = "recordSpecializationReuse", .callback = jsCompiledExecutableNoop, .length = 1 },
    };
    for (methods) |method| if (abi.jsSetFunction(ctx, object, method.name, method.callback, method.length) < 0) {
        abi.jsFreeValue(ctx, object);
        return errorValue(ctx, "failed to initialize compiled executable");
    };
    return object;
}

fn trackResult(result: *Tensor, inputs: []const *Tensor, op_tag: OpTag, axis: ?usize, scalar_a: ?f64, scalar_b: ?f64) !void {
    return trackResultWithSavedInputs(result, inputs, inputs, op_tag, axis, null, scalar_a, scalar_b);
}

fn trackResultWithSavedInputs(
    result: *Tensor,
    inputs: []const *Tensor,
    saved_inputs: []const *const Tensor,
    op_tag: OpTag,
    axis: ?usize,
    permute_axes: ?[]const usize,
    scalar_a: ?f64,
    scalar_b: ?f64,
) !void {
    if (!grad_mode.isEnabled()) return;
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
    const node = try autograd_execution.createNode(allocator, op_tag, parents.items, saved_inputs, result, null, axis, false, null, permute_axes, scalar_a, scalar_b);
    state.attachNode(node);
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
        abi.jsSetFunction(ctx, result, "repr", jsTensorToString, 0) < 0 or
        abi.jsSetFunction(ctx, result, "to", jsTensorTo, 1) < 0)
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
    return tensorScalarValue(ctx, input, 0) catch return errorValue(ctx, "failed to read Tensor");
}

fn jsTensorToArray(ctx: abi.JSContext, this_value: abi.JSValueConst, argc: c_int, _: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    if (argc != 0) return typeError(ctx, "Tensor.to_array expects no arguments");
    const input = tensorFromValue(ctx, this_value) orelse return typeError(ctx, "to_array expects a Tensor");
    var flat_index: usize = 0;
    return tensorArrayValue(ctx, input, 0, &flat_index) catch return errorValue(ctx, "failed to read Tensor");
}

fn jsTensorToString(ctx: abi.JSContext, this_value: abi.JSValueConst, argc: c_int, _: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    if (argc != 0) return typeError(ctx, "Tensor.toString expects no arguments");
    const input = tensorFromValue(ctx, this_value) orelse return typeError(ctx, "toString expects a Tensor");
    const dtype_name = switch (input.dtype) {
        .f32 => "f32",
        .f64 => "f64",
        .i64 => "i64",
    };
    const device_name = switch (input.device() orelse .cpu) {
        .cpu => "cpu",
        .metal => "metal",
    };
    const text = std.fmt.allocPrint(allocator, "Tensor(shape={any}, dtype={s}, device={s})", .{ input.shape.dims, dtype_name, device_name }) catch return errorValue(ctx, "out of memory");
    defer allocator.free(text);
    return abi.jsString(ctx, text);
}

fn jsTensorTo(ctx: abi.JSContext, this_value: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    if (argc != 1) return typeError(ctx, "Tensor.to expects a device");
    _ = tensorFromValue(ctx, this_value) orelse return typeError(ctx, "to expects a Tensor");
    if (!abi.jsStringEquals(ctx, argv[0], "cpu")) return typeError(ctx, "only cpu tensors are currently supported");
    return abi.jsDupValue(ctx, this_value);
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
    if (argc == 2) {
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
        if (!abi.jsIsUndefined(device_value) and !abi.jsStringEquals(ctx, device_value, "cpu")) return typeError(ctx, "only cpu tensors are currently supported");
    }
    const tensor = switch (dtype) {
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
    return createTensorObject(ctx, tensor);
}

fn createFilledTensor(ctx: abi.JSContext, shape_value: abi.JSValueConst, fill: f32, zeroed: bool) abi.JSValue {
    const shape = readShape(ctx, shape_value, "shape") orelse return typeError(ctx, "shape must be an array of non-negative integers");
    defer allocator.free(shape);
    const count = blk: {
        var count: usize = 1;
        for (shape) |dimension| count = std.math.mul(usize, count, dimension) catch return errorValue(ctx, "tensor is too large");
        break :blk count;
    };
    const values = allocator.alloc(f32, count) catch return errorValue(ctx, "out of memory");
    defer allocator.free(values);
    @memset(values, if (zeroed) 0 else fill);
    const tensor = Tensor.fromSliceF32(allocator, shape, values) catch return errorValue(ctx, "failed to create Tensor");
    return createTensorObject(ctx, tensor);
}

fn jsEmpty(ctx: abi.JSContext, _: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    if (argc != 1) return typeError(ctx, "empty expects a shape array");
    const shape = readShape(ctx, argv[0], "shape") orelse return typeError(ctx, "shape must be an array of non-negative integers");
    defer allocator.free(shape);
    const tensor = Tensor.createContiguous(allocator, shape, .f32, .cpu, false) catch return errorValue(ctx, "failed to create Tensor");
    return createTensorObject(ctx, tensor);
}

fn jsZeros(ctx: abi.JSContext, _: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    if (argc != 1) return typeError(ctx, "zeros expects a shape array");
    return createFilledTensor(ctx, argv[0], 0, true);
}

fn jsOnes(ctx: abi.JSContext, _: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    if (argc != 1) return typeError(ctx, "ones expects a shape array");
    return createFilledTensor(ctx, argv[0], 1, false);
}

fn jsFull(ctx: abi.JSContext, _: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    if (argc != 2) return typeError(ctx, "full expects a shape array and fill value");
    var fill: f64 = 0;
    if (abi.jsToFloat64(ctx, &fill, argv[1]) < 0) return typeError(ctx, "fill value must be a number");
    return createFilledTensor(ctx, argv[0], @floatCast(fill), false);
}

fn jsParameter(ctx: abi.JSContext, _: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    if (argc != 1) return typeError(ctx, "parameter expects a shape array");
    const shape = readShape(ctx, argv[0], "shape") orelse return typeError(ctx, "shape must be an array of non-negative integers");
    defer allocator.free(shape);
    const tensor = Tensor.createContiguous(allocator, shape, .f32, .cpu, false) catch return errorValue(ctx, "failed to create Parameter");
    _ = autograd_types.State.create(allocator, tensor, true) catch {
        tensor.deinit();
        return errorValue(ctx, "failed to create Parameter state");
    };
    return createTensorObject(ctx, tensor);
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

fn jsGrad(ctx: abi.JSContext, _: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    if (argc != 2) return typeError(ctx, "grad expects a loss Tensor and parameter array");
    const loss = tensorFromValue(ctx, argv[0]) orelse return typeError(ctx, "grad expects a loss Tensor");
    const params = readTensorList(ctx, argv[1], "parameters") orelse return typeError(ctx, "grad expects an array of Tensors");
    defer allocator.free(params);
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
    for (params, 0..) |parameter, index| {
        if (autograd_types.State.fromTensor(parameter)) |state| {
            if (state.gradient) |gradient| {
                gradient.deinit();
                state.gradient = null;
            }
        }
        const item = abi.jsGetArrayElement(ctx, argv[0], @intCast(index));
        defer abi.jsFreeValue(ctx, item);
        if (abi.jsSetProperty(ctx, item, "grad", abi.jsNull(ctx)) < 0) return errorValue(ctx, "failed to clear gradient");
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
    var count: usize = 1;
    for (shape) |dimension| count = std.math.mul(usize, count, dimension) catch return errorValue(ctx, "tensor is too large");
    const values = allocator.alloc(f32, count) catch return errorValue(ctx, "out of memory");
    defer allocator.free(values);
    const random = random_state.random();
    for (values) |*value| {
        if (normal) {
            value.* = random.float(f32) * 2.0 - 1.0;
        } else {
            value.* = random.float(f32);
        }
    }
    const tensor = Tensor.fromSliceF32(allocator, shape, values) catch return errorValue(ctx, "failed to create Tensor");
    return createTensorObject(ctx, tensor);
}

fn jsRand(ctx: abi.JSContext, _: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    if (argc != 1) return typeError(ctx, "rand expects a shape array");
    return jsRandom(ctx, argv[0], false);
}

fn jsRandn(ctx: abi.JSContext, _: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    if (argc != 1) return typeError(ctx, "randn expects a shape array");
    return jsRandom(ctx, argv[0], true);
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
    if (argc < 2 or argc > 3) return typeError(ctx, "linspace expects start, end, and optional steps");
    var first: f64 = 0;
    var last: f64 = 0;
    if (abi.jsToFloat64(ctx, &first, argv[0]) < 0 or abi.jsToFloat64(ctx, &last, argv[1]) < 0) return typeError(ctx, "linspace bounds must be numbers");
    var steps: i32 = 100;
    if (argc == 3 and abi.jsToInt32(ctx, &steps, argv[2]) < 0) return typeError(ctx, "linspace steps must be an integer");
    if (steps < 1) return typeError(ctx, "linspace steps must be positive");
    const values = allocator.alloc(f32, @intCast(steps)) catch return errorValue(ctx, "out of memory");
    defer allocator.free(values);
    for (values, 0..) |*value, index| {
        const ratio = if (steps == 1) 0 else @as(f64, @floatFromInt(index)) / @as(f64, @floatFromInt(steps - 1));
        value.* = @floatCast(first + (last - first) * ratio);
    }
    const tensor = Tensor.fromSliceF32(allocator, &.{values.len}, values) catch return errorValue(ctx, "failed to create Tensor");
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
    const mask = tensorFromValue(ctx, argv[1]) orelse return typeError(ctx, "masked_fill expects Tensor values");
    var fill: f64 = 0;
    if (abi.jsToFloat64(ctx, &fill, argv[2]) < 0) return typeError(ctx, "masked_fill value must be a number");
    const result = engine_api.Engine.init(allocator, .{}).maskedFill(input, mask, fill) catch return errorValue(ctx, "masked_fill failed");
    trackResult(result, &.{ input, mask }, .masked_fill, null, fill, null) catch {
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
    trackResultWithSavedInputs(result, &.{logits}, &.{ logits, targets }, .cross_entropy_indexed, axis, null, null, null) catch {
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
    return binary(ctx, argc, argv, .matmul, "matmul failed");
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
    if (argc < 1 or argc > 2) return typeError(ctx, "reduction expects a Tensor and optional axis");
    const input = tensorFromValue(ctx, argv[0]) orelse return typeError(ctx, name);
    const axis = if (argc == 2) integerArgument(ctx, argv[1], "axis") orelse return typeError(ctx, "axis must be a non-negative integer") else null;
    const engine = engine_api.Engine.init(allocator, .{});
    const result = switch (operation) {
        .sum => engine.reduce(.sum, input, axis, false),
        .mean => engine.reduce(.mean, input, axis, false),
        .min => engine.reduce(.min, input, axis, false),
        .max => engine.reduce(.max, input, axis, false),
        .variance => engine.reduce(.variance, input, axis, false),
        .std => engine.reduce(.std, input, axis, false),
        .argmin => engine.reduce(.argmin, input, axis, false),
        .argmax => engine.reduce(.argmax, input, axis, false),
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
    trackResult(result, &.{input}, op_tag, axis, null, null) catch {
        result.deinit();
        return errorValue(ctx, "failed to record autograd state");
    };
    return createTensorObject(ctx, result);
}

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
    const stop_text = parts.next() orelse return null;
    const step_text = parts.next();
    if (parts.next() != null) return null;
    const start: i64 = if (start_text.len == 0) 0 else std.fmt.parseInt(i64, start_text, 10) catch return null;
    const stop: i64 = if (stop_text.len == 0) @intCast(dimension) else std.fmt.parseInt(i64, stop_text, 10) catch return null;
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
    for (0..input.shape.rank()) |axis| {
        const selector = if (axis < selector_length) abi.jsGetArrayElement(ctx, argv[1], @intCast(axis)) else abi.jsString(ctx, ":");
        defer if (axis < selector_length) abi.jsFreeValue(ctx, selector);
        const range = parseSliceRange(ctx, selector, input.shape.dims[axis]) orelse return typeError(ctx, "slice selector is invalid");
        ranges.append(allocator, range) catch return errorValue(ctx, "out of memory");
    }
    const result = engine_api.Engine.init(allocator, .{}).slice(input, ranges.items) catch return errorValue(ctx, "slice failed");
    trackSliceResult(result, input, ranges.items) catch {
        result.deinit();
        return errorValue(ctx, "failed to record autograd state");
    };
    return createTensorObject(ctx, result);
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
    trackResultWithSavedInputs(result, &.{input}, &.{input}, .permute, null, axes, null, null) catch {
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
    const result = engine_api.Engine.init(allocator, .{}).cat(tensors, axis) catch return errorValue(ctx, "cat failed");
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
    const result = engine_api.Engine.init(allocator, .{}).stack(tensors, axis) catch return errorValue(ctx, "stack failed");
    trackResult(result, tensors, .stack, axis, null, null) catch {
        result.deinit();
        return errorValue(ctx, "failed to record autograd state");
    };
    return createTensorObject(ctx, result);
}

fn jsOneHot(ctx: abi.JSContext, _: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    if (argc != 2) return typeError(ctx, "one_hot expects indices and class count");
    const input = tensorFromValue(ctx, argv[0]) orelse return typeError(ctx, "one_hot expects a Tensor");
    const classes = integerArgument(ctx, argv[1], "class count") orelse return typeError(ctx, "class count must be a non-negative integer");
    const result = engine_api.Engine.init(allocator, .{}).oneHot(input, classes) catch return errorValue(ctx, "one_hot failed");
    return createTensorObject(ctx, result);
}

fn jsGather(ctx: abi.JSContext, _: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    if (argc != 3) return typeError(ctx, "gather expects input, axis, and index");
    const input = tensorFromValue(ctx, argv[0]) orelse return typeError(ctx, "gather expects a Tensor input");
    const axis = integerArgument(ctx, argv[1], "axis") orelse return typeError(ctx, "axis must be a non-negative integer");
    const index = tensorFromValue(ctx, argv[2]) orelse return typeError(ctx, "gather expects a Tensor index");
    const result = engine_api.Engine.init(allocator, .{}).gather(input, axis, index) catch return errorValue(ctx, "gather failed");
    trackResult(result, &.{ input, index }, .gather, axis, null, null) catch {
        result.deinit();
        return errorValue(ctx, "failed to record autograd state");
    };
    return createTensorObject(ctx, result);
}

fn jsIndexSelect(ctx: abi.JSContext, _: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    if (argc != 3) return typeError(ctx, "index_select expects input, axis, and index");
    const input = tensorFromValue(ctx, argv[0]) orelse return typeError(ctx, "index_select expects a Tensor input");
    const axis = integerArgument(ctx, argv[1], "axis") orelse return typeError(ctx, "axis must be a non-negative integer");
    const index = tensorFromValue(ctx, argv[2]) orelse return typeError(ctx, "index_select expects a Tensor index");
    const result = engine_api.Engine.init(allocator, .{}).indexSelect(input, axis, index) catch return errorValue(ctx, "index_select failed");
    trackResult(result, &.{ input, index }, .index_select, axis, null, null) catch {
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
    var outputs = engine_api.Engine.init(allocator, .{}).topK(input, k, axis) catch return errorValue(ctx, "topk failed");
    trackResult(outputs.primary, &.{input}, .topk, axis, null, null) catch {
        outputs.deinit();
        return errorValue(ctx, "failed to record autograd state");
    };
    const values = createTensorObject(ctx, outputs.primary);
    if (abi.jsIsException(values)) {
        outputs.primary = undefined;
        if (outputs.secondary) |secondary| secondary.deinit();
        return values;
    }
    outputs.primary = undefined;
    const indices = createTensorObject(ctx, outputs.secondary orelse return errorValue(ctx, "topk did not produce indices"));
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
    if (abi.jsSetProperty(ctx, result, "values", values) < 0 or abi.jsSetProperty(ctx, result, "indices", indices) < 0) {
        abi.jsFreeValue(ctx, result);
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
    .{ .name = "copy", .callback = jsCopy, .length = 2 },
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
