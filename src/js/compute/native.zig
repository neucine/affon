const std = @import("std");
const hao = @import("hao");
const compute = @import("compute");
const telemetry = @import("zig_libs").telemetry;
const config = @import("../../config.zig");
const compat = @import("../../support/compat.zig");

const abi = hao.js.abi;
const allocator = std.heap.c_allocator;
const tensor_type_id: u32 = 1;
const session_type_id: u32 = 2;
const executable_type_id: u32 = 3;

const ComputeTelemetrySink = struct {
    active_run_id: ?u64 = null,
    active_scope: ?telemetry.Scope = null,

    fn sink(self: *ComputeTelemetrySink) compute.EvidenceSink {
        return .{ .context = self, .emitFn = emit };
    }

    fn metric(group: []const u8, name: []const u8, kind: telemetry.MetricKind, unit: telemetry.MetricUnit) telemetry.MetricDefinition {
        return .{ .group = group, .name = name, .kind = kind, .unit = unit };
    }

    fn u64Value(value: u64) i64 {
        return @intCast(@min(value, @as(u64, std.math.maxInt(i64))));
    }

    fn usizeValue(value: usize) i64 {
        return std.math.cast(i64, value) orelse std.math.maxInt(i64);
    }

    fn event(self: *ComputeTelemetrySink, name: []const u8, attributes: []const telemetry.Attribute) void {
        if (self.active_scope) |scope| scope.addEventNow(name, attributes);
    }

    fn emit(context: *anyopaque, record: compute.EvidenceRecord) void {
        const self: *ComputeTelemetrySink = @ptrCast(@alignCast(context));
        switch (record.event) {
            .storage_allocated => |value| {
                const bytes = usizeValue(value.bytes);
                telemetry.add(metric("compute.storage", "allocated_bytes_total", .counter, .bytes), bytes);
                telemetry.add(metric("compute.storage", "live_bytes", .gauge, .bytes), bytes);
            },
            .storage_released => |value| {
                const bytes = usizeValue(value.bytes);
                telemetry.add(metric("compute.storage", "released_bytes_total", .counter, .bytes), bytes);
                telemetry.add(metric("compute.storage", "live_bytes", .gauge, .bytes), -bytes);
            },
            .execution_started => |value| {
                if (self.active_scope) |scope| scope.endError();
                self.active_run_id = value.run_id.int();
                self.active_scope = telemetry.startSpan(telemetry.trace.currentContext(), "compute.execution", .internal, &.{
                    .{ .key = "compute.session_id", .value = .{ .integer = usizeValue(record.session_id.int()) } },
                    .{ .key = "compute.run_id", .value = .{ .integer = u64Value(value.run_id.int()) } },
                    .{ .key = "compute.executable_id", .value = .{ .integer = u64Value(value.executable_id) } },
                });
                telemetry.add(metric("compute.execution", "runs_total", .counter, .count), 1);
            },
            .step_executed => |value| self.event("compute.step", &.{
                .{ .key = "compute.step_index", .value = .{ .integer = usizeValue(value.step_index) } },
            }),
            .invocation_submitted => |value| {
                telemetry.add(metric("compute.execution", "invocations_total", .counter, .count), 1);
                self.event("compute.invocation", &.{
                    .{ .key = "compute.step_index", .value = .{ .integer = usizeValue(value.step_index) } },
                    .{ .key = "compute.invocation_id", .value = .{ .integer = u64Value(value.invocation_id) } },
                });
            },
            .backend_timing => |value| {
                const elapsed = u64Value(value.elapsed_ns);
                telemetry.add(metric("compute.backend", "elapsed_nanoseconds_total", .counter, .nanoseconds), elapsed);
                self.event("compute.backend_timing", &.{
                    .{ .key = "compute.invocation_id", .value = .{ .integer = u64Value(value.invocation_id) } },
                    .{ .key = "compute.kernel_index", .value = .{ .integer = value.kernel_index } },
                    .{ .key = "compute.kernel_count", .value = .{ .integer = value.kernel_count } },
                    .{ .key = "compute.elapsed_ns", .value = .{ .integer = elapsed } },
                });
            },
            .synchronization => |value| {
                telemetry.add(metric("compute.synchronization", "count", .counter, .count), 1);
                self.event("compute.synchronization", &.{
                    .{ .key = "compute.reason", .value = .{ .string = @tagName(value.reason) } },
                });
            },
            .hardware_metric_availability => |value| self.event("compute.hardware_metric_availability", &.{
                .{ .key = "compute.metric", .value = .{ .string = @tagName(value.metric) } },
                .{ .key = "compute.availability", .value = .{ .string = @tagName(value.availability) } },
            }),
            .hardware_metric_sample => |value| self.event("compute.hardware_metric", &.{
                .{ .key = "compute.metric", .value = .{ .string = @tagName(value.metric) } },
                .{ .key = "compute.value", .value = .{ .integer = u64Value(value.value) } },
                .{ .key = "compute.scale", .value = .{ .integer = u64Value(value.scale) } },
            }),
            .execution_completed => |value| {
                if (self.active_run_id == value.run_id.int()) {
                    if (self.active_scope) |scope| scope.end();
                    self.active_scope = null;
                    self.active_run_id = null;
                }
            },
            .session_shutdown => {
                if (self.active_scope) |scope| scope.endError();
                self.active_scope = null;
                self.active_run_id = null;
            },
        }
    }
};

const SessionObject = struct {
    value: ?*compute.Session,
    backend: compute.Backend,
    telemetry_sink: ComputeTelemetrySink = .{},
    children: usize = 0,
    disposed: bool = false,
    host_alive: bool = true,
};

const TensorObject = struct {
    value: ?*compute.Tensor,
    backend: compute.Backend,
    owner: ?*SessionObject = null,
};

const ExecutableObject = struct {
    value: ?*compute.Executable,
    owner: *SessionObject,
    explanation_json: []u8,
    needs_seed: bool,
    owner_released: bool = false,
};

var cpu_session: ?*compute.Session = null;
var metal_session: ?*compute.Session = null;
var cuda_session: ?*compute.Session = null;

fn sessionFor(backend: compute.Backend) !*compute.Session {
    const slot = switch (backend) {
        .cpu => &cpu_session,
        .metal => &metal_session,
        .cuda => &cuda_session,
    };
    if (slot.* == null) slot.* = try compute.createSession(allocator, backend);
    return slot.*.?;
}

fn defaultBackend() compute.Backend {
    return switch (config.getDefaultDevice()) { .cpu => .cpu, .metal => .metal, .cuda => .cuda };
}

fn tensorObject(ctx: abi.JSContext, value: abi.JSValueConst) ?*TensorObject {
    const handle = abi.jsHostObjectHandle(ctx, value, tensor_type_id) orelse return null;
    return @ptrFromInt(@as(usize, @intCast(handle)));
}

fn sessionObject(ctx: abi.JSContext, value: abi.JSValueConst) ?*SessionObject {
    const handle = abi.jsHostObjectHandle(ctx, value, session_type_id) orelse return null;
    return @ptrFromInt(@as(usize, @intCast(handle)));
}

fn executableObject(ctx: abi.JSContext, value: abi.JSValueConst) ?*ExecutableObject {
    const handle = abi.jsHostObjectHandle(ctx, value, executable_type_id) orelse return null;
    return @ptrFromInt(@as(usize, @intCast(handle)));
}

fn maybeReleaseSession(object: *SessionObject) void {
    if (object.disposed and object.children == 0) if (object.value) |value| {
        value.deinit() catch unreachable;
        object.value = null;
    };
    if (!object.host_alive and object.children == 0) allocator.destroy(object);
}
fn retainSession(object: *SessionObject) void { object.children += 1; }
fn releaseSession(object: *SessionObject) void {
    std.debug.assert(object.children > 0);
    object.children -= 1;
    maybeReleaseSession(object);
}

fn tensorFinalizer(_: u32, handle: u64) callconv(.c) void {
    const object: *TensorObject = @ptrFromInt(@as(usize, @intCast(handle)));
    if (object.value) |value| value.deinit();
    if (object.owner) |owner| releaseSession(owner);
    allocator.destroy(object);
}

fn sessionFinalizer(_: u32, handle: u64) callconv(.c) void {
    const object: *SessionObject = @ptrFromInt(@as(usize, @intCast(handle)));
    object.disposed = true;
    object.host_alive = false;
    maybeReleaseSession(object);
}

fn executableFinalizer(_: u32, handle: u64) callconv(.c) void {
    const object: *ExecutableObject = @ptrFromInt(@as(usize, @intCast(handle)));
    if (object.value) |value| value.deinit();
    if (!object.owner_released) releaseSession(object.owner);
    allocator.free(object.explanation_json);
    allocator.destroy(object);
}

fn errorValue(ctx: abi.JSContext, message: [*:0]const u8) abi.JSValue { return abi.jsThrowError(ctx, message); }
fn typeError(ctx: abi.JSContext, message: [*:0]const u8) abi.JSValue { return abi.jsThrowTypeError(ctx, message); }

fn createTensorObjectOwned(ctx: abi.JSContext, value: *compute.Tensor, backend: compute.Backend, owner: ?*SessionObject) abi.JSValue {
    const object = allocator.create(TensorObject) catch { value.deinit(); return errorValue(ctx, "out of memory"); };
    if (owner) |session| retainSession(session);
    object.* = .{ .value = value, .backend = backend, .owner = owner };
    const result = abi.createJSHostObject(ctx, tensor_type_id, @intCast(@intFromPtr(object)), tensorFinalizer);
    if (abi.jsIsException(result)) {
        value.deinit();
        if (owner) |session| releaseSession(session);
        allocator.destroy(object);
        return result;
    }
    const shape = abi.jsNewArray(ctx);
    for (value.spec().dimensions(), 0..) |dimension, index| {
        if (abi.jsSetArrayElement(ctx, shape, @intCast(index), abi.jsInt32(ctx, @intCast(dimension))) < 0) return errorValue(ctx, "failed to create shape");
    }
    const dtype: [:0]const u8 = switch (value.spec().dtype()) { .f32 => "f32", .f64 => "f64", .i64 => "i64" };
    const device: [:0]const u8 = switch (backend) { .cpu => "cpu", .metal => "metal", .cuda => "cuda" };
    if (abi.jsSetProperty(ctx, result, "shape", shape) < 0 or
        abi.jsSetProperty(ctx, result, "ndim", abi.jsInt32(ctx, @intCast(value.spec().dimensions().len))) < 0 or
        abi.jsSetProperty(ctx, result, "dtype", abi.jsString(ctx, dtype)) < 0 or
        abi.jsSetProperty(ctx, result, "device", abi.jsString(ctx, device)) < 0 or
        abi.jsSetFunction(ctx, result, "item", jsItem, 0) < 0 or
        abi.jsSetFunction(ctx, result, "to_array", jsToArray, 0) < 0 or
        abi.jsSetFunction(ctx, result, "toString", jsToString, 0) < 0 or
        abi.jsSetFunction(ctx, result, "repr", jsToString, 0) < 0 or
        abi.jsSetFunction(ctx, result, "dispose", jsDisposeTensor, 0) < 0)
        return errorValue(ctx, "failed to create Tensor object");
    return result;
}

fn createTensorObject(ctx: abi.JSContext, value: *compute.Tensor, backend: compute.Backend) abi.JSValue {
    return createTensorObjectOwned(ctx, value, backend, null);
}

fn elementCount(shape: []const usize) usize { var n: usize = 1; for (shape) |d| n *= d; return n; }
fn scalar(ctx: abi.JSContext, tensor: *compute.Tensor, linear: usize) abi.JSValue {
    const shape = tensor.spec().dimensions();
    var remainder = linear;
    var element: isize = @intCast(tensor.offsetBytes() / tensor.spec().dtype().size());
    var axis = shape.len;
    while (axis > 0) { axis -= 1; const coordinate = remainder % shape[axis]; remainder /= shape[axis]; element += @as(isize, @intCast(coordinate)) * tensor.strides()[axis]; }
    const offset: usize = @intCast(element);
    const bytes = tensor.bytes();
    return switch (tensor.spec().dtype()) {
        .f32 => abi.jsFloat64(ctx, std.mem.bytesAsSlice(f32, @as([]align(4) const u8, @alignCast(bytes)))[offset]),
        .f64 => abi.jsFloat64(ctx, std.mem.bytesAsSlice(f64, @as([]align(8) const u8, @alignCast(bytes)))[offset]),
        .i64 => abi.jsFloat64(ctx, @floatFromInt(std.mem.bytesAsSlice(i64, @as([]align(8) const u8, @alignCast(bytes)))[offset])),
    };
}
fn arrayValue(ctx: abi.JSContext, tensor: *compute.Tensor, depth: usize, linear: *usize) abi.JSValue {
    if (depth == tensor.spec().dimensions().len) { const result = scalar(ctx, tensor, linear.*); linear.* += 1; return result; }
    const result = abi.jsNewArray(ctx);
    for (0..tensor.spec().dimensions()[depth]) |index| if (abi.jsSetArrayElement(ctx, result, @intCast(index), arrayValue(ctx, tensor, depth + 1, linear)) < 0) return errorValue(ctx, "failed to create Tensor array");
    return result;
}
fn liveTensor(ctx: abi.JSContext, value: abi.JSValueConst) ?*compute.Tensor {
    const object = tensorObject(ctx, value) orelse return null;
    return object.value orelse null;
}
fn jsToArray(ctx: abi.JSContext, this_value: abi.JSValueConst, _: c_int, _: [*c]abi.JSValueConst) callconv(.c) abi.JSValue { const value = liveTensor(ctx, this_value) orelse return typeError(ctx, "Tensor has been disposed"); var index: usize = 0; return arrayValue(ctx, value, 0, &index); }
fn jsItem(ctx: abi.JSContext, this_value: abi.JSValueConst, _: c_int, _: [*c]abi.JSValueConst) callconv(.c) abi.JSValue { const value = liveTensor(ctx, this_value) orelse return typeError(ctx, "Tensor has been disposed"); if (elementCount(value.spec().dimensions()) != 1) return typeError(ctx, "item requires one element"); return scalar(ctx, value, 0); }
fn jsToString(ctx: abi.JSContext, this_value: abi.JSValueConst, _: c_int, _: [*c]abi.JSValueConst) callconv(.c) abi.JSValue { const value = liveTensor(ctx, this_value) orelse return typeError(ctx, "Tensor has been disposed"); var buffer: [128]u8 = undefined; const text = std.fmt.bufPrint(&buffer, "Tensor(shape={any}, dtype={s})", .{ value.spec().dimensions(), @tagName(value.spec().dtype()) }) catch return errorValue(ctx, "repr failed"); return abi.jsString(ctx, text); }
fn jsDisposeTensor(ctx: abi.JSContext, this_value: abi.JSValueConst, _: c_int, _: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    const object = tensorObject(ctx, this_value) orelse return typeError(ctx, "invalid Tensor");
    if (object.value) |value| { value.deinit(); object.value = null; }
    if (object.owner) |owner| {
        releaseSession(owner);
        object.owner = null;
    }
    return abi.jsUndefined(ctx);
}

const Flat = struct { shape: std.ArrayList(usize) = .empty, values: std.ArrayList(f64) = .empty, fn deinit(self: *Flat) void { self.shape.deinit(allocator); self.values.deinit(allocator); } };
fn flatten(ctx: abi.JSContext, value: abi.JSValueConst, depth: usize, result: *Flat) !void {
    if (abi.jsIsArray(ctx, value)) {
        const length_value = abi.jsGetProperty(ctx, value, "length"); defer abi.jsFreeValue(ctx, length_value);
        var length: i32 = 0; if (abi.jsToInt32(ctx, &length, length_value) < 0 or length < 0) return error.InvalidInput;
        if (result.shape.items.len == depth) try result.shape.append(allocator, @intCast(length)) else if (result.shape.items[depth] != length) return error.JaggedArray;
        for (0..@as(usize, @intCast(length))) |index| { const item = abi.jsGetArrayElement(ctx, value, @intCast(index)); defer abi.jsFreeValue(ctx, item); try flatten(ctx, item, depth + 1, result); }
    } else { if (depth != result.shape.items.len) return error.JaggedArray; var number: f64 = 0; if (abi.jsToFloat64(ctx, &number, value) < 0) return error.InvalidInput; try result.values.append(allocator, number); }
}
const JsonSpec = struct {
    dtype: []const u8,
    shape: []const usize,
    axes: ?[]const []const u8 = null,
};

const JsonNode = struct {
    id: usize,
    kind: []const u8,
    path: ?std.json.Value = null,
    role: ?[]const u8 = null,
    name: ?[]const u8 = null,
    provenance: ?[]const u8 = null,
    spec: JsonSpec,
    op: ?[]const u8 = null,
    operands: ?[]const usize = null,
    options: ?std.json.Value = null,
    value: ?std.json.Value = null,
};

const JsonProgram = struct {
    kind: []const u8,
    nodes: []const JsonNode,
    outputs: []const usize,
    transitions: ?[]const std.json.Value = null,
};

fn parseBackend(text: []const u8) ?compute.Backend {
    if (std.mem.eql(u8, text, "cpu")) return .cpu;
    if (std.mem.eql(u8, text, "metal")) return .metal;
    if (std.mem.eql(u8, text, "cuda") or std.mem.startsWith(u8, text, "cuda:")) return .cuda;
    return null;
}

fn parseDType(text: []const u8) ?compute.DType {
    if (std.mem.eql(u8, text, "f32")) return .f32;
    if (std.mem.eql(u8, text, "f64")) return .f64;
    if (std.mem.eql(u8, text, "i64")) return .i64;
    return null;
}

fn jsonNumber(value: std.json.Value) ?f64 {
    return switch (value) { .float => |number| number, .integer => |number| @floatFromInt(number), else => null };
}

fn appendJsonNumbers(value: std.json.Value, values: *std.ArrayList(f64)) !void {
    switch (value) {
        .array => |array| for (array.items) |item| try appendJsonNumbers(item, values),
        else => try values.append(allocator, jsonNumber(value) orelse return error.UnsupportedConstant),
    }
}

fn optionValue(options: ?std.json.Value, name: []const u8) ?std.json.Value {
    const value = options orelse return null;
    return switch (value) { .object => |object| object.get(name), else => null };
}

fn optionUsize(options: ?std.json.Value, name: []const u8) ?usize {
    const value = optionValue(options, name) orelse return null;
    return switch (value) { .integer => |number| std.math.cast(usize, number), else => null };
}

fn optionIsize(options: ?std.json.Value, name: []const u8) ?isize {
    const value = optionValue(options, name) orelse return null;
    return switch (value) { .integer => |number| std.math.cast(isize, number), else => null };
}

fn optionF64(options: ?std.json.Value, name: []const u8) ?f64 {
    const value = optionValue(options, name) orelse return null;
    return jsonNumber(value);
}

fn optionBool(options: ?std.json.Value, name: []const u8, fallback: bool) bool {
    const value = optionValue(options, name) orelse return fallback;
    return switch (value) { .bool => |boolean| boolean, else => fallback };
}

fn stateId(node: JsonNode) compute.StateId {
    const identity = node.provenance orelse node.name orelse "state";
    return .fromInt(std.hash.Wyhash.hash(0xaff0_5700, identity));
}

fn addJsonOperation(builder: *compute.ProgramBuilder, node: JsonNode, mapped: []const compute.ValueId, nodes: []const JsonNode) !compute.ValueId {
    const op = node.op orelse return error.MissingOperation;
    const operand_ids = try allocator.alloc(compute.ValueId, (node.operands orelse return error.MissingOperands).len);
    defer allocator.free(operand_ids);
    for (node.operands.?, operand_ids) |source, *destination| {
        if (source >= mapped.len) return error.InvalidValueReference;
        destination.* = mapped[source];
    }
    const tag: compute.OpTag = if (std.mem.eql(u8, op, "add")) .add
        else if (std.mem.eql(u8, op, "sub")) .sub
        else if (std.mem.eql(u8, op, "mul")) .mul
        else if (std.mem.eql(u8, op, "div")) .div
        else if (std.mem.eql(u8, op, "eq")) .eq
        else if (std.mem.eql(u8, op, "lt")) .lt
        else if (std.mem.eql(u8, op, "gt")) .gt
        else if (std.mem.eql(u8, op, "matmul")) .matmul
        else if (std.mem.eql(u8, op, "dot")) .dot
        else if (std.mem.eql(u8, op, "abs")) .abs
        else if (std.mem.eql(u8, op, "neg")) .neg
        else if (std.mem.eql(u8, op, "exp")) .exp
        else if (std.mem.eql(u8, op, "log")) .log
        else if (std.mem.eql(u8, op, "sqrt")) .sqrt
        else if (std.mem.eql(u8, op, "sign")) .sign
        else if (std.mem.eql(u8, op, "relu")) .relu
        else if (std.mem.eql(u8, op, "sigmoid")) .sigmoid
        else if (std.mem.eql(u8, op, "silu")) .silu
        else if (std.mem.eql(u8, op, "tanh")) .tanh
        else if (std.mem.eql(u8, op, "erf")) .erf
        else if (std.mem.eql(u8, op, "gelu")) .gelu
        else if (std.mem.eql(u8, op, "clamp")) .clamp
        else if (std.mem.eql(u8, op, "where")) .where
        else if (std.mem.eql(u8, op, "softmax")) .softmax
        else if (std.mem.eql(u8, op, "sum")) if (optionUsize(node.options, "axis") == null) .sum_all else .sum_axis
        else if (std.mem.eql(u8, op, "mean")) if (optionUsize(node.options, "axis") == null) .mean_all else .mean_axis
        else if (std.mem.eql(u8, op, "min")) if (optionUsize(node.options, "axis") == null) .min_all else .min_axis
        else if (std.mem.eql(u8, op, "max")) if (optionUsize(node.options, "axis") == null) .max_all else .max_axis
        else if (std.mem.eql(u8, op, "variance")) if (optionUsize(node.options, "axis") == null) .variance_all else .variance_axis
        else if (std.mem.eql(u8, op, "std")) if (optionUsize(node.options, "axis") == null) .std_all else .std_axis
        else if (std.mem.eql(u8, op, "argmin")) if (optionUsize(node.options, "axis") == null) .argmin_all else .argmin_axis
        else if (std.mem.eql(u8, op, "argmax")) if (optionUsize(node.options, "axis") == null) .argmax_all else .argmax_axis
        else if (std.mem.eql(u8, op, "reshape")) .reshape
        else if (std.mem.eql(u8, op, "contiguous")) .contiguous
        else if (std.mem.eql(u8, op, "slice")) .slice
        else if (std.mem.eql(u8, op, "squeeze")) .squeeze
        else if (std.mem.eql(u8, op, "unsqueeze")) .unsqueeze
        else if (std.mem.eql(u8, op, "transpose")) .transpose
        else if (std.mem.eql(u8, op, "cast")) .cast
        else if (std.mem.eql(u8, op, "masked_fill")) .masked_fill
        else if (std.mem.eql(u8, op, "cat")) .cat
        else if (std.mem.eql(u8, op, "stack")) .stack
        else if (std.mem.eql(u8, op, "layer_norm")) .layer_norm
        else if (std.mem.eql(u8, op, "embedding")) .embedding
        else if (std.mem.eql(u8, op, "index_select")) .index_select
        else if (std.mem.eql(u8, op, "gather")) .gather
        else if (std.mem.eql(u8, op, "one_hot")) .one_hot
        else if (std.mem.eql(u8, op, "cross_entropy")) .cross_entropy_indexed
        else return error.UnsupportedOperation;

    var permutation_storage: ?[]usize = null;
    defer if (permutation_storage) |values| allocator.free(values);
    var slice_storage: ?[]compute.SliceRange = null;
    defer if (slice_storage) |values| allocator.free(values);
    const options: compute.OpOptions = switch (tag) {
        .softmax => .{ .softmax = .{ .axis = optionUsize(node.options, "axis") orelse return error.MissingOption } },
        .sum_all, .mean_all, .min_all, .max_all, .variance_all, .std_all, .argmin_all, .argmax_all => .{ .reduce_all = .{ .keepdim = optionBool(node.options, "keep_dims", false) } },
        .sum_axis, .mean_axis, .min_axis, .max_axis, .variance_axis, .std_axis, .argmin_axis, .argmax_axis => .{ .reduce_axis = .{ .axis = optionUsize(node.options, "axis") orelse return error.MissingOption, .keepdim = optionBool(node.options, "keep_dims", false) } },
        .reshape => .{ .reshape = .{ .shape = node.spec.shape } },
        .slice => blk: {
            const raw = optionValue(node.options, "ranges") orelse return error.MissingOption;
            if (raw != .array) return error.InvalidOption;
            const ranges = try allocator.alloc(compute.SliceRange, raw.array.items.len);
            slice_storage = ranges;
            for (raw.array.items, ranges) |item, *destination| {
                if (item != .object) return error.InvalidOption;
                const start_value = item.object.get("start") orelse return error.InvalidOption;
                const stop_value = item.object.get("stop") orelse return error.InvalidOption;
                if (start_value != .integer or stop_value != .integer) return error.InvalidOption;
                const step_value = item.object.get("step");
                destination.* = .{
                    .start = std.math.cast(usize, start_value.integer) orelse return error.InvalidOption,
                    .stop = std.math.cast(usize, stop_value.integer) orelse return error.InvalidOption,
                    .step = if (step_value) |step| switch (step) { .integer => |number| std.math.cast(isize, number) orelse return error.InvalidOption, else => return error.InvalidOption } else 1,
                };
            }
            break :blk .{ .slice = .{ .ranges = ranges } };
        },
        .squeeze => .{ .squeeze = .{ .axis = optionUsize(node.options, "axis") } },
        .unsqueeze => .{ .unsqueeze = .{ .axis = optionUsize(node.options, "axis") orelse return error.MissingOption } },
        .transpose => blk: {
            const raw = optionValue(node.options, "permutation") orelse break :blk .{ .transpose = .{} };
            if (raw != .array) return error.InvalidOption;
            const values = try allocator.alloc(usize, raw.array.items.len);
            permutation_storage = values;
            for (raw.array.items, values) |item, *destination| destination.* = switch (item) { .integer => |number| std.math.cast(usize, number) orelse return error.InvalidOption, else => return error.InvalidOption };
            break :blk .{ .transpose = .{ .permutation = values } };
        },
        .cast => .{ .cast = .{ .to = parseDType(node.spec.dtype) orelse return error.UnsupportedDType } },
        .masked_fill => .{ .masked_fill = .{ .value = optionF64(node.options, "value") orelse return error.MissingOption } },
        .clamp => .{ .clamp = .{ .min = optionF64(node.options, "min") orelse return error.MissingOption, .max = optionF64(node.options, "max") orelse return error.MissingOption } },
        .cat => .{ .concat = .{ .axis = optionUsize(node.options, "axis") orelse return error.MissingOption } },
        .stack => .{ .stack = .{ .axis = optionUsize(node.options, "axis") orelse return error.MissingOption } },
        .layer_norm => .{ .layer_norm = .{ .axis = optionUsize(node.options, "axis") orelse return error.MissingOption, .eps = optionF64(node.options, "epsilon") orelse return error.MissingOption } },
        .embedding => .{ .embedding = .{} },
        .index_select => .{ .index_select = .{ .axis = optionUsize(node.options, "axis") orelse return error.MissingOption } },
        .gather => .{ .gather = .{ .axis = optionUsize(node.options, "axis") orelse return error.MissingOption } },
        .one_hot => .{ .one_hot = .{ .num_classes = optionUsize(node.options, "num_classes") orelse return error.MissingOption } },
        .cross_entropy_indexed => blk: {
            const logits_id = node.operands.?[0];
            if (logits_id >= nodes.len) return error.InvalidValueReference;
            const rank = nodes[logits_id].spec.shape.len;
            const requested = optionIsize(node.options, "class_axis") orelse -1;
            const axis: usize = if (requested < 0)
                std.math.cast(usize, @as(isize, @intCast(rank)) + requested) orelse return error.InvalidOption
            else
                std.math.cast(usize, requested) orelse return error.InvalidOption;
            if (axis >= rank) return error.InvalidOption;
            break :blk .{ .cross_entropy_indexed = .{ .axis = axis } };
        },
        else => .{ .none = {} },
    };
    const outputs = try builder.add(tag, options, operand_ids);
    if (outputs.len != 1) return error.UnsupportedMultipleOutputs;
    return outputs[0];
}

const BuiltProgram = struct {
    value: *compute.Program,
    mapped: []compute.ValueId,
    fn deinit(self: *BuiltProgram) void { self.value.deinit(); allocator.free(self.mapped); }
};

fn buildJsonProgram(source: JsonProgram, output_ids: []const usize) !BuiltProgram {
    var builder = compute.ProgramBuilder.init(allocator);
    defer builder.deinit();
    const mapped = try allocator.alloc(compute.ValueId, source.nodes.len);
    errdefer allocator.free(mapped);
    for (source.nodes) |node| {
        if (node.id >= mapped.len) return error.InvalidNodeId;
        if (std.mem.eql(u8, node.kind, "gradient")) continue;
        const dtype = parseDType(node.spec.dtype) orelse return error.UnsupportedDType;
        var spec = try compute.TensorSpec.init(allocator, dtype, node.spec.shape, node.spec.axes);
        defer spec.deinit();
        if (node.role) |role| {
            if (std.mem.eql(u8, role, "argument")) mapped[node.id] = try builder.addInput(spec)
            else if (std.mem.eql(u8, role, "parameter")) mapped[node.id] = try builder.addStateInput(spec, .parameter, stateId(node))
            else if (std.mem.eql(u8, role, "state")) mapped[node.id] = try builder.addStateInput(spec, .model_state, stateId(node))
            else if (std.mem.eql(u8, role, "constant")) {
                const value = node.value orelse return error.MissingConstant;
                if (jsonNumber(value)) |number| {
                    mapped[node.id] = switch (dtype) {
                        .f32 => try builder.addConstant(spec, .{ .scalar = .{ .f32 = @floatCast(number) } }),
                        .f64 => try builder.addConstant(spec, .{ .scalar = .{ .f64 = number } }),
                        .i64 => try builder.addConstant(spec, .{ .scalar = .{ .i64 = @intFromFloat(number) } }),
                    };
                } else {
                    var numbers: std.ArrayList(f64) = .empty;
                    defer numbers.deinit(allocator);
                    try appendJsonNumbers(value, &numbers);
                    if (numbers.items.len != elementCount(node.spec.shape)) return error.InvalidConstantShape;
                    mapped[node.id] = switch (dtype) {
                        .f32 => blk: {
                            const values = try allocator.alloc(f32, numbers.items.len);
                            defer allocator.free(values);
                            for (numbers.items, values) |source_number, *destination| destination.* = @floatCast(source_number);
                            break :blk try builder.addConstant(spec, .{ .bytes = std.mem.sliceAsBytes(values) });
                        },
                        .f64 => try builder.addConstant(spec, .{ .bytes = std.mem.sliceAsBytes(numbers.items) }),
                        .i64 => blk: {
                            const values = try allocator.alloc(i64, numbers.items.len);
                            defer allocator.free(values);
                            for (numbers.items, values) |source_number, *destination| destination.* = @intFromFloat(source_number);
                            break :blk try builder.addConstant(spec, .{ .bytes = std.mem.sliceAsBytes(values) });
                        },
                    };
                }
            } else return error.UnsupportedRole;
        } else mapped[node.id] = try addJsonOperation(&builder, node, mapped, source.nodes);
    }
    const outputs = try allocator.alloc(compute.ValueId, output_ids.len);
    defer allocator.free(outputs);
    for (output_ids, outputs) |source_id, *destination| {
        if (source_id >= mapped.len) return error.InvalidNodeId;
        destination.* = mapped[source_id];
    }
    return .{ .value = try builder.finish(outputs), .mapped = mapped };
}

fn createExecutableObject(ctx: abi.JSContext, value: *compute.Executable, owner: *SessionObject, needs_seed: bool, explanation_json: []u8) abi.JSValue {
    const object = allocator.create(ExecutableObject) catch { allocator.free(explanation_json); value.deinit(); return errorValue(ctx, "out of memory"); };
    retainSession(owner);
    object.* = .{ .value = value, .owner = owner, .needs_seed = needs_seed, .explanation_json = explanation_json };
    const result = abi.createJSHostObject(ctx, executable_type_id, @intCast(@intFromPtr(object)), executableFinalizer);
    if (abi.jsIsException(result)) { executableFinalizer(executable_type_id, @intCast(@intFromPtr(object))); return result; }
    if (abi.jsSetFunction(ctx, result, "dispose", jsDisposeExecutable, 0) < 0) return errorValue(ctx, "failed to create Executable object");
    return result;
}

const JsonSessionTelemetry = struct {
    backendTiming: bool = false,
    hardwareMetrics: bool = false,
};
const JsonSessionOptions = struct {
    determinism: []const u8 = "strict",
    telemetry: ?JsonSessionTelemetry = JsonSessionTelemetry{},
};
const JsonCompileOptions = struct {
    optimizationLevel: []const u8 = "safe",
    numericalPolicy: []const u8 = "backend_equivalent",
    optimizationGoal: []const u8 = "balanced",
    explanationLevel: []const u8 = "summary",
    residualPolicy: []const u8 = "retain",
};

fn compileOptionsFromJson(bytes: []const u8) !compute.CompileOptions {
    const parsed = try std.json.parseFromSlice(JsonCompileOptions, allocator, bytes, .{ .ignore_unknown_fields = false });
    defer parsed.deinit();
    return .{
        .optimization_level = if (std.mem.eql(u8, parsed.value.optimizationLevel, "none")) .none else if (std.mem.eql(u8, parsed.value.optimizationLevel, "safe")) .safe else return error.InvalidOptimizationLevel,
        .numerical_policy = if (std.mem.eql(u8, parsed.value.numericalPolicy, "exact_only")) .exact_only else if (std.mem.eql(u8, parsed.value.numericalPolicy, "backend_equivalent")) .backend_equivalent else if (std.mem.eql(u8, parsed.value.numericalPolicy, "approximate")) .approximate else return error.InvalidNumericalPolicy,
        .optimization_goal = if (std.mem.eql(u8, parsed.value.optimizationGoal, "balanced")) .balanced else if (std.mem.eql(u8, parsed.value.optimizationGoal, "latency")) .latency else if (std.mem.eql(u8, parsed.value.optimizationGoal, "throughput")) .throughput else if (std.mem.eql(u8, parsed.value.optimizationGoal, "memory")) .memory else return error.InvalidOptimizationGoal,
        .explanation_level = if (std.mem.eql(u8, parsed.value.explanationLevel, "summary")) .summary else if (std.mem.eql(u8, parsed.value.explanationLevel, "detailed")) .detailed else return error.InvalidExplanationLevel,
        .residual_policy = if (std.mem.eql(u8, parsed.value.residualPolicy, "automatic")) .automatic else if (std.mem.eql(u8, parsed.value.residualPolicy, "retain")) .retain else if (std.mem.eql(u8, parsed.value.residualPolicy, "recompute")) .recompute else return error.InvalidResidualPolicy,
    };
}

fn jsCreateSession(ctx: abi.JSContext, _: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    if (argc < 1) return typeError(ctx, "createSession expects a device");
    const text = abi.jsStringAlloc(ctx, argv[0], allocator) catch return typeError(ctx, "device must be a string");
    defer allocator.free(text);
    const backend = parseBackend(text) orelse return typeError(ctx, "unsupported device");
    if (!compute.backendAvailable(backend)) return errorValue(ctx, "backend unavailable");
    var parsed_options: ?std.json.Parsed(JsonSessionOptions) = null;
    defer if (parsed_options) |*parsed| parsed.deinit();
    var options_json_owned: ?[]u8 = null;
    defer if (options_json_owned) |bytes| allocator.free(bytes);
    if (argc >= 2) {
        options_json_owned = abi.jsStringAlloc(ctx, argv[1], allocator) catch return typeError(ctx, "Session options must be JSON");
        parsed_options = std.json.parseFromSlice(JsonSessionOptions, allocator, options_json_owned.?, .{ .ignore_unknown_fields = false }) catch return typeError(ctx, "invalid Session options");
    }
    const options = if (parsed_options) |parsed| parsed.value else JsonSessionOptions{};
    const determinism: compute.DeterminismPolicy = if (std.mem.eql(u8, options.determinism, "strict")) .strict else if (std.mem.eql(u8, options.determinism, "allow_nondeterministic")) .allow_nondeterministic else {
        var message: [160]u8 = undefined;
        const rendered = std.fmt.bufPrintZ(&message, "unsupported determinism policy: '{s}'", .{options.determinism}) catch return typeError(ctx, "unsupported determinism policy");
        return typeError(ctx, rendered.ptr);
    };
    const object = allocator.create(SessionObject) catch return errorValue(ctx, "out of memory");
    object.* = .{ .value = null, .backend = backend };
    const telemetry_options = options.telemetry orelse JsonSessionTelemetry{};
    const value = compute.createConfiguredSession(allocator, backend, .{
        .determinism = determinism,
        .evidence_sink = if (options.telemetry == null) null else object.telemetry_sink.sink(),
        .observation = .{ .backend_timing = telemetry_options.backendTiming, .hardware_metrics = telemetry_options.hardwareMetrics },
    }) catch {
        allocator.destroy(object);
        return errorValue(ctx, "session creation failed");
    };
    object.value = value;
    const result = abi.createJSHostObject(ctx, session_type_id, @intCast(@intFromPtr(object)), sessionFinalizer);
    if (abi.jsIsException(result)) { sessionFinalizer(session_type_id, @intCast(@intFromPtr(object))); return result; }
    if (abi.jsSetFunction(ctx, result, "dispose", jsDisposeSession, 0) < 0) return errorValue(ctx, "failed to create Session object");
    return result;
}

fn jsDefaultDevice(ctx: abi.JSContext, _: abi.JSValueConst, _: c_int, _: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    const name: [:0]const u8 = switch (config.getDefaultDevice()) { .cpu => "cpu", .metal => "metal", .cuda => "cuda" };
    return abi.jsString(ctx, name);
}

fn jsDisposeSession(ctx: abi.JSContext, this_value: abi.JSValueConst, _: c_int, _: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    const object = sessionObject(ctx, this_value) orelse return typeError(ctx, "invalid Session");
    object.disposed = true;
    maybeReleaseSession(object);
    return abi.jsUndefined(ctx);
}

fn jsSessionTensor(ctx: abi.JSContext, _: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    if (argc < 3) return typeError(ctx, "sessionTensor expects Session, values, and dtype");
    const owner = sessionObject(ctx, argv[0]) orelse return typeError(ctx, "invalid Session");
    if (owner.disposed) return typeError(ctx, "Session has been disposed");
    const dtype_text = abi.jsStringAlloc(ctx, argv[2], allocator) catch return typeError(ctx, "dtype must be a string");
    defer allocator.free(dtype_text);
    const dtype = parseDType(dtype_text) orelse return typeError(ctx, "unsupported dtype");
    var flat = Flat{}; defer flat.deinit();
    flatten(ctx, argv[1], 0, &flat) catch return typeError(ctx, "tensor expects a rectangular numeric value");
    var requested_shape: std.ArrayList(usize) = .empty;
    defer requested_shape.deinit(allocator);
    if (argc >= 4) {
        if (!abi.jsIsArray(ctx, argv[3])) return typeError(ctx, "shape must be an array");
        const length_value = abi.jsGetProperty(ctx, argv[3], "length"); defer abi.jsFreeValue(ctx, length_value);
        var length: i32 = 0;
        if (abi.jsToInt32(ctx, &length, length_value) < 0 or length < 0) return typeError(ctx, "invalid shape");
        for (0..@as(usize, @intCast(length))) |index| {
            const item = abi.jsGetArrayElement(ctx, argv[3], @intCast(index)); defer abi.jsFreeValue(ctx, item);
            var dimension: i64 = 0;
            if (abi.jsToInt64(ctx, &dimension, item) < 0 or dimension < 0) return typeError(ctx, "shape must contain non-negative integers");
            requested_shape.append(allocator, @intCast(dimension)) catch return errorValue(ctx, "out of memory");
        }
    }
    const shape = if (argc >= 4) requested_shape.items else flat.shape.items;
    if (elementCount(shape) != flat.values.items.len) return typeError(ctx, "shape does not match tensor values");
    var spec = compute.TensorSpec.init(allocator, dtype, shape, null) catch return errorValue(ctx, "tensor spec failed"); defer spec.deinit();
    const bytes = switch (dtype) {
        .f32 => blk: { const values = allocator.alloc(f32, flat.values.items.len) catch return errorValue(ctx, "out of memory"); defer allocator.free(values); for (flat.values.items, values) |source, *destination| destination.* = @floatCast(source); break :blk owner.value.?.createTensor(spec, std.mem.sliceAsBytes(values)) catch return errorValue(ctx, "tensor creation failed"); },
        .f64 => owner.value.?.createTensor(spec, std.mem.sliceAsBytes(flat.values.items)) catch return errorValue(ctx, "tensor creation failed"),
        .i64 => blk: { const values = allocator.alloc(i64, flat.values.items.len) catch return errorValue(ctx, "out of memory"); defer allocator.free(values); for (flat.values.items, values) |source, *destination| destination.* = @intFromFloat(source); break :blk owner.value.?.createTensor(spec, std.mem.sliceAsBytes(values)) catch return errorValue(ctx, "tensor creation failed"); },
    };
    return createTensorObjectOwned(ctx, bytes, owner.backend, owner);
}

fn jsSessionFull(ctx: abi.JSContext, _: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    if (argc < 4) return typeError(ctx, "sessionFull expects Session, shape, value, and dtype");
    const owner = sessionObject(ctx, argv[0]) orelse return typeError(ctx, "invalid Session");
    if (owner.disposed) return typeError(ctx, "Session has been disposed");
    if (!abi.jsIsArray(ctx, argv[1])) return typeError(ctx, "shape must be an array");
    var shape: std.ArrayList(usize) = .empty;
    defer shape.deinit(allocator);
    const length_value = abi.jsGetProperty(ctx, argv[1], "length"); defer abi.jsFreeValue(ctx, length_value);
    var length: i32 = 0;
    if (abi.jsToInt32(ctx, &length, length_value) < 0 or length < 0) return typeError(ctx, "invalid shape");
    for (0..@as(usize, @intCast(length))) |index| {
        const item = abi.jsGetArrayElement(ctx, argv[1], @intCast(index)); defer abi.jsFreeValue(ctx, item);
        var dimension: i64 = 0;
        if (abi.jsToInt64(ctx, &dimension, item) < 0 or dimension < 0) return typeError(ctx, "shape must contain non-negative integers");
        shape.append(allocator, @intCast(dimension)) catch return errorValue(ctx, "out of memory");
    }
    var fill: f64 = 0;
    if (abi.jsToFloat64(ctx, &fill, argv[2]) < 0 or !std.math.isFinite(fill)) return typeError(ctx, "fill value must be finite");
    const dtype_text = abi.jsStringAlloc(ctx, argv[3], allocator) catch return typeError(ctx, "dtype must be a string");
    defer allocator.free(dtype_text);
    const dtype = parseDType(dtype_text) orelse return typeError(ctx, "unsupported dtype");
    var spec = compute.TensorSpec.init(allocator, dtype, shape.items, null) catch return errorValue(ctx, "tensor spec failed");
    defer spec.deinit();
    const count = elementCount(shape.items);
    const byte_count = std.math.mul(usize, count, dtype.size()) catch return errorValue(ctx, "tensor is too large");
    const bytes = allocator.alloc(u8, byte_count) catch return errorValue(ctx, "out of memory");
    defer allocator.free(bytes);
    switch (dtype) {
        .f32 => {
            for (std.mem.bytesAsSlice(f32, @as([]align(4) u8, @alignCast(bytes)))) |*item| item.* = @floatCast(fill);
        },
        .f64 => {
            for (std.mem.bytesAsSlice(f64, @as([]align(8) u8, @alignCast(bytes)))) |*item| item.* = fill;
        },
        .i64 => {
            for (std.mem.bytesAsSlice(i64, @as([]align(8) u8, @alignCast(bytes)))) |*item| item.* = @intFromFloat(fill);
        },
    }
    const value = owner.value.?.createTensor(spec, bytes) catch return errorValue(ctx, "tensor creation failed");
    return createTensorObjectOwned(ctx, value, owner.backend, owner);
}

fn compileNativeProgram(ctx: abi.JSContext, owner: *SessionObject, program_value: *const compute.Program, needs_seed: bool, options: compute.CompileOptions) abi.JSValue {
    // Fusion can remove instruction boundaries referenced by automatic residual
    // recomputation. Retaining residuals keeps the public Program path valid
    // while preserving the rest of the safe optimization profile.
    var compilation = owner.value.?.compile(program_value, options) catch return errorValue(ctx, "Program compilation failed");
    if (compilation == .diagnostics) {
        const entries = compilation.diagnostics.entries();
        var message: [256]u8 = undefined;
        const rendered = if (entries.len == 0)
            "Program compilation diagnostics"
        else
            std.fmt.bufPrintZ(&message, "Program compilation failed: {s}: {s}", .{ @tagName(entries[0].code()), entries[0].message() }) catch "Program compilation diagnostics";
        compilation.deinit();
        return errorValue(ctx, rendered.ptr);
    }
    const executable = compilation.executable;
    var explanation = compute.createExplanation(allocator, program_value, program_value, executable) catch { compilation.deinit(); return errorValue(ctx, "Program explanation failed"); };
    defer explanation.deinit();
    const explanation_json = explanation.serialize(allocator) catch { compilation.deinit(); return errorValue(ctx, "Program explanation serialization failed"); };
    executable.retain();
    compilation.deinit();
    return createExecutableObject(ctx, executable, owner, needs_seed, explanation_json);
}

fn jsCompileProgram(ctx: abi.JSContext, _: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    if (argc < 2) return typeError(ctx, "compileProgram expects Session and Program JSON");
    const owner = sessionObject(ctx, argv[0]) orelse return typeError(ctx, "invalid Session");
    if (owner.disposed) return typeError(ctx, "Session has been disposed");
    const json = abi.jsStringAlloc(ctx, argv[1], allocator) catch return typeError(ctx, "Program must be JSON");
    defer allocator.free(json);
    var options: compute.CompileOptions = .{ .residual_policy = .retain };
    if (argc >= 3) {
        const options_json = abi.jsStringAlloc(ctx, argv[2], allocator) catch return typeError(ctx, "compile options must be JSON");
        defer allocator.free(options_json);
        options = compileOptionsFromJson(options_json) catch return typeError(ctx, "invalid compile options");
    }
    var parsed = std.json.parseFromSlice(JsonProgram, allocator, json, .{ .ignore_unknown_fields = true }) catch return typeError(ctx, "invalid Program JSON");
    defer parsed.deinit();
    const source = parsed.value;
    if (std.mem.eql(u8, source.kind, "gradient")) {
        var gradient_nodes: std.ArrayList(JsonNode) = .empty;
        defer gradient_nodes.deinit(allocator);
        for (source.nodes) |node| if (std.mem.eql(u8, node.kind, "gradient")) gradient_nodes.append(allocator, node) catch return errorValue(ctx, "out of memory");
        if (gradient_nodes.items.len == 0) return typeError(ctx, "gradient Program has no derivatives");
        const loss_id = gradient_nodes.items[0].operands.?[0];
        var built = buildJsonProgram(source, &.{loss_id}) catch return errorValue(ctx, "Program construction failed");
        defer built.deinit();
        const variables = allocator.alloc(compute.ValueId, gradient_nodes.items.len) catch return errorValue(ctx, "out of memory");
        defer allocator.free(variables);
        for (gradient_nodes.items, variables) |node, *destination| {
            const provenance_value = optionValue(node.options, "with_respect_to") orelse return typeError(ctx, "gradient is missing provenance");
            if (provenance_value != .string) return typeError(ctx, "invalid gradient provenance");
            var found: ?usize = null;
            for (source.nodes) |candidate| if (candidate.provenance != null and std.mem.eql(u8, candidate.provenance.?, provenance_value.string)) { found = candidate.id; break; };
            destination.* = built.mapped[found orelse return typeError(ctx, "unknown gradient provenance")];
        }
        var derivative = compute.differentiate(allocator, built.value, .{ .output = built.mapped[loss_id], .with_respect_to = variables }) catch return errorValue(ctx, "Program differentiation failed");
        defer derivative.deinit();
        return compileNativeProgram(ctx, owner, derivative.program(), true, options);
    }
    if (std.mem.eql(u8, source.kind, "optimize")) return typeError(ctx, "optimize Program native lowering is not implemented");
    var built = buildJsonProgram(source, source.outputs) catch |err| {
        var message: [160]u8 = undefined;
        const rendered = std.fmt.bufPrintZ(&message, "Program construction failed: {s}", .{@errorName(err)}) catch return errorValue(ctx, "Program construction failed");
        return errorValue(ctx, rendered.ptr);
    };
    defer built.deinit();
    return compileNativeProgram(ctx, owner, built.value, false, options);
}

fn jsExplainExecutable(ctx: abi.JSContext, _: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    if (argc < 1) return typeError(ctx, "explainExecutable expects Executable");
    const object = executableObject(ctx, argv[0]) orelse return typeError(ctx, "invalid Executable");
    if (object.value == null) return typeError(ctx, "Executable has been disposed");
    return abi.jsString(ctx, object.explanation_json);
}

fn jsDisposeExecutable(ctx: abi.JSContext, this_value: abi.JSValueConst, _: c_int, _: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    const object = executableObject(ctx, this_value) orelse return typeError(ctx, "invalid Executable");
    if (object.value) |value| { value.deinit(); object.value = null; }
    if (!object.owner_released) {
        releaseSession(object.owner);
        object.owner_released = true;
    }
    return abi.jsUndefined(ctx);
}

fn jsRunExecutable(ctx: abi.JSContext, _: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    if (argc < 2 or !abi.jsIsArray(ctx, argv[1])) return typeError(ctx, "runExecutable expects Executable and Tensor array");
    const object = executableObject(ctx, argv[0]) orelse return typeError(ctx, "invalid Executable");
    const executable = object.value orelse return typeError(ctx, "Executable has been disposed");
    if (object.owner.disposed) return typeError(ctx, "Session has been disposed");
    const length_value = abi.jsGetProperty(ctx, argv[1], "length"); defer abi.jsFreeValue(ctx, length_value);
    var length: i32 = 0; if (abi.jsToInt32(ctx, &length, length_value) < 0 or length < 0) return typeError(ctx, "invalid input array");
    const extra: usize = if (object.needs_seed) 1 else 0;
    const inputs = allocator.alloc(*compute.Tensor, @as(usize, @intCast(length)) + extra) catch return errorValue(ctx, "out of memory"); defer allocator.free(inputs);
    for (0..@as(usize, @intCast(length))) |index| {
        const item = abi.jsGetArrayElement(ctx, argv[1], @intCast(index)); defer abi.jsFreeValue(ctx, item);
        const tensor = tensorObject(ctx, item) orelse return typeError(ctx, "inputs must be Tensors");
        if (tensor.owner != object.owner) return typeError(ctx, "Tensor belongs to a different Session");
        inputs[index] = tensor.value orelse return typeError(ctx, "Tensor has been disposed");
    }
    var seed: ?*compute.Tensor = null;
    defer if (seed) |value| value.deinit();
    if (object.needs_seed) {
        const expected = executable.inputs()[executable.inputs().len - 1].spec();
        seed = switch (expected.dtype()) {
            .f32 => blk: { const one: f32 = 1; break :blk object.owner.value.?.createTensor(expected.*, std.mem.asBytes(&one)) catch return errorValue(ctx, "seed creation failed"); },
            .f64 => blk: { const one: f64 = 1; break :blk object.owner.value.?.createTensor(expected.*, std.mem.asBytes(&one)) catch return errorValue(ctx, "seed creation failed"); },
            .i64 => return typeError(ctx, "gradient loss must use a floating dtype"),
        };
        inputs[inputs.len - 1] = seed.?;
    }
    const outputs = object.owner.value.?.run(executable, inputs) catch |err| {
        var message: [160]u8 = undefined;
        const rendered = std.fmt.bufPrintZ(&message, "Program execution failed: {s}", .{@errorName(err)}) catch return errorValue(ctx, "Program execution failed");
        return errorValue(ctx, rendered.ptr);
    };
    defer allocator.free(outputs);
    const result = abi.jsNewArray(ctx);
    for (outputs, 0..) |output, index| if (abi.jsSetArrayElement(ctx, result, @intCast(index), createTensorObjectOwned(ctx, output, object.owner.backend, object.owner)) < 0) return errorValue(ctx, "failed to create output array");
    return result;
}

fn denseTensor(tensor: *const compute.Tensor) bool {
    var stride: isize = 1;
    var axis = tensor.spec().dimensions().len;
    while (axis > 0) {
        axis -= 1;
        if (tensor.strides()[axis] != stride) return false;
        stride *= @intCast(tensor.spec().dimensions()[axis]);
    }
    return tensor.offsetBytes() == 0;
}

const FFT = struct {
    n: usize,
    re: []f64,
    im: []f64,
    scratch_re: []f64,
    scratch_im: []f64,
    cosine: []f64,
    sine: []f64,
    fn run(self: FFT, input: []const f64, offset: usize, stride: usize, size: usize, output: usize) void {
        if (size == 1) { self.re[output] = input[offset]; self.im[output] = 0; return; }
        var radix: usize = 2;
        while (size % radix != 0) : (radix += 1) {}
        const block = size / radix;
        for (0..radix) |part| self.run(input, offset + part * stride, stride * radix, block, output + part * block);
        for (0..size) |frequency| {
            var real: f64 = 0;
            var imaginary: f64 = 0;
            for (0..radix) |part| {
                const index = output + part * block + frequency % block;
                const angle = ((part * frequency) % size) * (self.n / size);
                real += self.re[index] * self.cosine[angle] - self.im[index] * self.sine[angle];
                imaginary += self.re[index] * self.sine[angle] + self.im[index] * self.cosine[angle];
            }
            self.scratch_re[output + frequency] = real;
            self.scratch_im[output + frequency] = imaginary;
        }
        @memcpy(self.re[output..][0..size], self.scratch_re[output..][0..size]);
        @memcpy(self.im[output..][0..size], self.scratch_im[output..][0..size]);
    }
};

fn positiveInteger(ctx: abi.JSContext, value: abi.JSValueConst) ?usize {
    var integer: i64 = 0;
    if (abi.jsToInt64(ctx, &integer, value) < 0 or integer <= 0) return null;
    return std.math.cast(usize, integer);
}

fn jsStftPower(ctx: abi.JSContext, _: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    if (argc != 5) return typeError(ctx, "stftPower expects signal, window, hop, padded length, and frame count");
    const signal_object = tensorObject(ctx, argv[0]) orelse return typeError(ctx, "signal must be a Tensor");
    const window_object = tensorObject(ctx, argv[1]) orelse return typeError(ctx, "window must be a Tensor");
    const signal = signal_object.value orelse return typeError(ctx, "signal has been disposed");
    const window = window_object.value orelse return typeError(ctx, "window has been disposed");
    const hop = positiveInteger(ctx, argv[2]) orelse return typeError(ctx, "invalid STFT hop");
    const padded_length = positiveInteger(ctx, argv[3]) orelse return typeError(ctx, "invalid STFT padded length");
    const frames = positiveInteger(ctx, argv[4]) orelse return typeError(ctx, "invalid STFT frame count");
    if (signal_object.backend != .cpu or window_object.backend != .cpu or signal.spec().dtype() != .f32 or window.spec().dtype() != .f64 or signal.spec().dimensions().len != 1 or window.spec().dimensions().len != 1 or !denseTensor(signal) or !denseTensor(window)) return typeError(ctx, "stftPower requires contiguous CPU f32 signal and f64 window");
    const samples = std.mem.bytesAsSlice(f32, @as([]align(4) const u8, @alignCast(signal.bytes())));
    const weights = std.mem.bytesAsSlice(f64, @as([]align(8) const u8, @alignCast(window.bytes())));
    const n = weights.len;
    if (n < 2 or n > 4096 or padded_length < samples.len or padded_length <= n / 2 or frames > 1 + padded_length / hop) return typeError(ctx, "invalid STFT dimensions");
    const bins = n / 2 + 1;
    var spec = compute.TensorSpec.init(allocator, .f64, &.{ bins, frames }, null) catch return errorValue(ctx, "STFT spec allocation failed");
    defer spec.deinit();
    const session = sessionFor(.cpu) catch return errorValue(ctx, "STFT CPU session unavailable");
    const output = session.createOutput(spec) catch return errorValue(ctx, "STFT output allocation failed");
    errdefer output.deinit();
    const destination = std.mem.bytesAsSlice(f64, @as([]align(8) u8, @alignCast(output.writableBytes())));
    @memset(destination, 0);
    const workspace = allocator.alloc(f64, n * 7) catch return errorValue(ctx, "STFT workspace allocation failed");
    defer allocator.free(workspace);
    const transform = FFT{ .n = n, .re = workspace[0..n], .im = workspace[n .. 2 * n], .scratch_re = workspace[2 * n .. 3 * n], .scratch_im = workspace[3 * n .. 4 * n], .cosine = workspace[4 * n .. 5 * n], .sine = workspace[5 * n .. 6 * n] };
    const input = workspace[6 * n ..];
    for (0..n) |index| {
        const angle = -2 * std.math.pi * @as(f64, @floatFromInt(index)) / @as(f64, @floatFromInt(n));
        transform.cosine[index] = @cos(angle); transform.sine[index] = @sin(angle);
    }
    for (0..frames) |frame| {
        var nonzero = false;
        for (0..n) |index| {
            var source = @as(i64, @intCast(frame * hop + index)) - @as(i64, @intCast(n / 2));
            if (source < 0) source = -source;
            if (source >= padded_length) source = 2 * @as(i64, @intCast(padded_length)) - source - 2;
            const value: f64 = if (source >= 0 and source < samples.len) samples[@intCast(source)] else 0;
            input[index] = value * weights[index]; nonzero = nonzero or input[index] != 0;
        }
        if (!nonzero) continue;
        transform.run(input, 0, 1, n, 0);
        for (0..bins) |bin| {
            const real: f64 = @as(f32, @floatCast(transform.re[bin]));
            const imaginary: f64 = @as(f32, @floatCast(transform.im[bin]));
            destination[bin * frames + frame] = real * real + imaginary * imaginary;
        }
    }
    return createTensorObject(ctx, output, .cpu);
}

fn jsFilterbank(ctx: abi.JSContext, _: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    if (argc != 2) return typeError(ctx, "filterbank expects spectrum and filters");
    const spectrum_object = tensorObject(ctx, argv[0]) orelse return typeError(ctx, "spectrum must be a Tensor");
    const filters_object = tensorObject(ctx, argv[1]) orelse return typeError(ctx, "filters must be a Tensor");
    const spectrum = spectrum_object.value orelse return typeError(ctx, "spectrum has been disposed");
    const filters = filters_object.value orelse return typeError(ctx, "filters have been disposed");
    if (spectrum_object.backend != .cpu or filters_object.backend != .cpu or spectrum.spec().dtype() != .f64 or filters.spec().dtype() != .f64 or spectrum.spec().dimensions().len != 2 or filters.spec().dimensions().len != 2 or !denseTensor(spectrum) or !denseTensor(filters)) return typeError(ctx, "filterbank requires contiguous CPU f64 matrices");
    const bins = spectrum.spec().dimensions()[0];
    const frames = spectrum.spec().dimensions()[1];
    const bands = filters.spec().dimensions()[0];
    if (filters.spec().dimensions()[1] != bins or bins == 0 or frames == 0 or bands == 0) return typeError(ctx, "invalid filterbank dimensions");
    const values = std.mem.bytesAsSlice(f64, @as([]align(8) const u8, @alignCast(spectrum.bytes())));
    const weights = std.mem.bytesAsSlice(f64, @as([]align(8) const u8, @alignCast(filters.bytes())));
    var spec = compute.TensorSpec.init(allocator, .f64, &.{ bands, frames }, null) catch return errorValue(ctx, "filterbank spec allocation failed");
    defer spec.deinit();
    const session = sessionFor(.cpu) catch return errorValue(ctx, "filterbank CPU session unavailable");
    const output = session.createOutput(spec) catch return errorValue(ctx, "filterbank output allocation failed");
    errdefer output.deinit();
    const destination = std.mem.bytesAsSlice(f64, @as([]align(8) u8, @alignCast(output.writableBytes())));
    @memset(destination, 0);
    for (0..bands) |band| for (0..bins) |bin| {
        const weight = weights[band * bins + bin];
        if (weight == 0) continue;
        for (0..frames) |frame| destination[band * frames + frame] += weight * values[bin * frames + frame];
    };
    return createTensorObject(ctx, output, .cpu);
}

const CheckpointEntry = struct { name: []u8, value: *compute.Tensor };

fn checkpointDTypeName(dtype: compute.DType) []const u8 {
    return switch (dtype) { .f32 => "F32", .f64 => "F64", .i64 => "I64" };
}

fn checkpointDType(text: []const u8) ?compute.DType {
    if (std.mem.eql(u8, text, "F32")) return .f32;
    if (std.mem.eql(u8, text, "F64")) return .f64;
    if (std.mem.eql(u8, text, "I64")) return .i64;
    return null;
}

fn copyLogicalTensorBytes(tensor: *const compute.Tensor, destination: []u8) !void {
    const dtype_size = tensor.spec().dtype().size();
    const elements = elementCount(tensor.spec().dimensions());
    if (destination.len != try std.math.mul(usize, elements, dtype_size)) return error.ByteCountMismatch;
    const source = tensor.bytes();
    for (0..elements) |linear| {
        var remainder = linear;
        var source_element: isize = @intCast(tensor.offsetBytes() / dtype_size);
        var axis = tensor.spec().dimensions().len;
        while (axis > 0) {
            axis -= 1;
            const coordinate = remainder % tensor.spec().dimensions()[axis];
            remainder /= tensor.spec().dimensions()[axis];
            source_element = try std.math.add(isize, source_element, try std.math.mul(isize, @intCast(coordinate), tensor.strides()[axis]));
        }
        const source_index = std.math.cast(usize, source_element) orelse return error.InvalidTensorLayout;
        const source_offset = try std.math.mul(usize, source_index, dtype_size);
        const destination_offset = try std.math.mul(usize, linear, dtype_size);
        if (source_offset + dtype_size > source.len) return error.InvalidTensorLayout;
        @memcpy(destination[destination_offset..][0..dtype_size], source[source_offset..][0..dtype_size]);
    }
}

fn jsSaveNative(ctx: abi.JSContext, _: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    if (argc < 2 or !abi.jsIsArray(ctx, argv[0])) return typeError(ctx, "checkpoint.save expects entries and a path");
    const path = abi.jsStringAlloc(ctx, argv[1], allocator) catch return typeError(ctx, "checkpoint.save expects a string path");
    defer allocator.free(path);
    const length_value = abi.jsGetProperty(ctx, argv[0], "length");
    defer abi.jsFreeValue(ctx, length_value);
    var count: i32 = 0;
    if (abi.jsToInt32(ctx, &count, length_value) < 0 or count <= 0) return errorValue(ctx, "checkpoint.save requires non-empty entries");
    const entries = allocator.alloc(CheckpointEntry, @intCast(count)) catch return errorValue(ctx, "checkpoint.save out of memory");
    var initialized: usize = 0;
    defer {
        for (entries[0..initialized]) |entry| allocator.free(entry.name);
        allocator.free(entries);
    }
    for (entries, 0..) |*entry, index| {
        const item = abi.jsGetArrayElement(ctx, argv[0], @intCast(index));
        defer abi.jsFreeValue(ctx, item);
        const name_value = abi.jsGetProperty(ctx, item, "name");
        defer abi.jsFreeValue(ctx, name_value);
        entry.name = abi.jsStringAlloc(ctx, name_value, allocator) catch return typeError(ctx, "checkpoint entry name must be a string");
        initialized += 1;
        const tensor_value = abi.jsGetProperty(ctx, item, "value");
        defer abi.jsFreeValue(ctx, tensor_value);
        entry.value = liveTensor(ctx, tensor_value) orelse return typeError(ctx, "checkpoint entry value must be a Tensor");
    }
    var header = std.ArrayList(u8).empty;
    defer header.deinit(allocator);
    header.append(allocator, '{') catch return errorValue(ctx, "checkpoint.save out of memory");
    var offset: usize = 0;
    for (entries, 0..) |entry, index| {
        if (index > 0) header.append(allocator, ',') catch return errorValue(ctx, "checkpoint.save out of memory");
        header.append(allocator, '"') catch return errorValue(ctx, "checkpoint.save out of memory");
        header.appendSlice(allocator, entry.name) catch return errorValue(ctx, "checkpoint.save out of memory");
        header.appendSlice(allocator, "\":{\"dtype\":\"") catch return errorValue(ctx, "checkpoint.save out of memory");
        header.appendSlice(allocator, checkpointDTypeName(entry.value.spec().dtype())) catch return errorValue(ctx, "checkpoint.save out of memory");
        header.appendSlice(allocator, "\",\"shape\":[") catch return errorValue(ctx, "checkpoint.save out of memory");
        for (entry.value.spec().dimensions(), 0..) |dimension, dimension_index| {
            if (dimension_index > 0) header.append(allocator, ',') catch return errorValue(ctx, "checkpoint.save out of memory");
            var buffer: [32]u8 = undefined;
            header.appendSlice(allocator, std.fmt.bufPrint(&buffer, "{d}", .{dimension}) catch return errorValue(ctx, "checkpoint.save out of memory")) catch return errorValue(ctx, "checkpoint.save out of memory");
        }
        header.appendSlice(allocator, "],\"data_offsets\":[") catch return errorValue(ctx, "checkpoint.save out of memory");
        var buffer: [32]u8 = undefined;
        header.appendSlice(allocator, std.fmt.bufPrint(&buffer, "{d}", .{offset}) catch return errorValue(ctx, "checkpoint.save out of memory")) catch return errorValue(ctx, "checkpoint.save out of memory");
        offset += elementCount(entry.value.spec().dimensions()) * entry.value.spec().dtype().size();
        header.append(allocator, ',') catch return errorValue(ctx, "checkpoint.save out of memory");
        header.appendSlice(allocator, std.fmt.bufPrint(&buffer, "{d}", .{offset}) catch return errorValue(ctx, "checkpoint.save out of memory")) catch return errorValue(ctx, "checkpoint.save out of memory");
        header.appendSlice(allocator, "]}") catch return errorValue(ctx, "checkpoint.save out of memory");
    }
    header.append(allocator, '}') catch return errorValue(ctx, "checkpoint.save out of memory");
    var output = std.ArrayList(u8).empty;
    defer output.deinit(allocator);
    const header_size_le = std.mem.nativeToLittle(u64, @intCast(header.items.len));
    output.appendSlice(allocator, std.mem.asBytes(&header_size_le)) catch return errorValue(ctx, "checkpoint.save out of memory");
    output.appendSlice(allocator, header.items) catch return errorValue(ctx, "checkpoint.save out of memory");
    for (entries) |entry| {
        const byte_count = elementCount(entry.value.spec().dimensions()) * entry.value.spec().dtype().size();
        const bytes = allocator.alloc(u8, byte_count) catch return errorValue(ctx, "checkpoint.save out of memory");
        defer allocator.free(bytes);
        copyLogicalTensorBytes(entry.value, bytes) catch return errorValue(ctx, "checkpoint.save failed to read Tensor");
        output.appendSlice(allocator, bytes) catch return errorValue(ctx, "checkpoint.save out of memory");
    }
    compat.writeFile(path, output.items) catch return errorValue(ctx, "checkpoint.save failed to write file");
    return abi.jsUndefined(ctx);
}

fn readCheckpointBytes(file: compat.FdFile, bytes: []u8) !void {
    var offset: usize = 0;
    while (offset < bytes.len) {
        const count = try file.read(bytes[offset..@min(bytes.len, offset + 1024 * 1024)]);
        if (count == 0) return error.UnexpectedEndOfFile;
        offset += count;
    }
}

fn jsLoadNative(ctx: abi.JSContext, _: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue { return readCheckpoint(ctx, argc, argv, false); }
fn jsInspectCheckpoint(ctx: abi.JSContext, _: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue { return readCheckpoint(ctx, argc, argv, true); }

fn readCheckpoint(ctx: abi.JSContext, argc: c_int, argv: [*c]abi.JSValueConst, inspect: bool) abi.JSValue {
    if (argc < 1) return typeError(ctx, "checkpoint.load expects a path");
    const path = abi.jsStringAlloc(ctx, argv[0], allocator) catch return typeError(ctx, "checkpoint.load expects a string path");
    defer allocator.free(path);
    const file = compat.FdFile.openRead(path) catch return errorValue(ctx, "checkpoint.load failed to open file");
    defer file.close();
    const file_size = file.size() catch return errorValue(ctx, "checkpoint.load failed to stat file");
    if (file_size < 8) return errorValue(ctx, "checkpoint.load file is too small");
    var prefix: [8]u8 = undefined;
    readCheckpointBytes(file, &prefix) catch return errorValue(ctx, "checkpoint.load truncated header");
    const header_size = std.mem.readInt(u64, &prefix, .little);
    if (header_size > 16 * 1024 * 1024 or header_size > file_size - 8) return errorValue(ctx, "checkpoint.load has an invalid header");
    const header = allocator.alloc(u8, @intCast(header_size)) catch return errorValue(ctx, "checkpoint.load out of memory");
    defer allocator.free(header);
    readCheckpointBytes(file, header) catch return errorValue(ctx, "checkpoint.load truncated header");
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, header, .{}) catch return errorValue(ctx, "checkpoint.load has invalid metadata");
    defer parsed.deinit();
    var selected = std.ArrayList([]const u8).empty;
    defer { for (selected.items) |name| allocator.free(name); selected.deinit(allocator); }
    const selective = !inspect and argc > 1 and !abi.jsIsUndefined(argv[1]);
    if (selective) {
        if (!abi.jsIsArray(ctx, argv[1])) return typeError(ctx, "checkpoint.load names must be an array");
        const length_value = abi.jsGetProperty(ctx, argv[1], "length");
        defer abi.jsFreeValue(ctx, length_value);
        var count: i32 = 0;
        if (abi.jsToInt32(ctx, &count, length_value) < 0 or count < 0) return typeError(ctx, "Invalid checkpoint names");
        for (0..@as(usize, @intCast(count))) |index| {
            const item = abi.jsGetArrayElement(ctx, argv[1], @intCast(index));
            defer abi.jsFreeValue(ctx, item);
            if (!abi.jsIsString(item)) return typeError(ctx, "checkpoint names must be strings");
            const name = abi.jsStringAlloc(ctx, item, allocator) catch return errorValue(ctx, "checkpoint.load out of memory");
            for (selected.items) |previous| if (std.mem.eql(u8, previous, name)) { allocator.free(name); return errorValue(ctx, "Duplicate checkpoint name"); };
            selected.append(allocator, name) catch { allocator.free(name); return errorValue(ctx, "checkpoint.load out of memory"); };
        }
    }
    var matched: usize = 0;
    const result = abi.jsNewObject(ctx);
    if (abi.jsIsException(result)) return result;
    var owns_result = true;
    defer if (owns_result) abi.jsFreeValue(ctx, result);
    const raw_size = file_size - 8 - header_size;
    if (parsed.value != .object) return errorValue(ctx, "checkpoint.load header must be an object");
    for (parsed.value.object.keys(), parsed.value.object.values()) |name, metadata| {
        if (metadata != .object) return errorValue(ctx, "checkpoint.load metadata must be an object");
        const object = metadata.object;
        if (std.mem.eql(u8, name, "__metadata__")) {
            for (object.values()) |entry| if (entry != .string) return errorValue(ctx, "checkpoint.load __metadata__ values must be strings");
            continue;
        }
        const dtype_text = object.get("dtype") orelse return errorValue(ctx, "checkpoint.load metadata is missing dtype");
        if (dtype_text != .string) return errorValue(ctx, "checkpoint.load dtype must be a string");
        const is_bf16 = std.mem.eql(u8, dtype_text.string, "BF16");
        const dtype = if (is_bf16) compute.DType.f32 else checkpointDType(dtype_text.string) orelse return errorValue(ctx, "checkpoint.load has unsupported dtype");
        const shape_value = object.get("shape") orelse return errorValue(ctx, "checkpoint.load metadata is missing shape");
        if (shape_value != .array) return errorValue(ctx, "checkpoint.load shape must be an array");
        const dimensions = allocator.alloc(usize, shape_value.array.items.len) catch return errorValue(ctx, "checkpoint.load out of memory");
        defer allocator.free(dimensions);
        var elements: usize = 1;
        for (dimensions, shape_value.array.items) |*dimension, item| {
            if (item != .integer or item.integer < 0) return errorValue(ctx, "checkpoint.load shape must contain non-negative integers");
            dimension.* = std.math.cast(usize, item.integer) orelse return errorValue(ctx, "checkpoint.load dimension is too large");
            elements = std.math.mul(usize, elements, dimension.*) catch return errorValue(ctx, "checkpoint.load shape is too large");
        }
        const offsets = object.get("data_offsets") orelse return errorValue(ctx, "checkpoint.load metadata is missing offsets");
        if (offsets != .array or offsets.array.items.len != 2) return errorValue(ctx, "checkpoint.load expects two offsets");
        for (offsets.array.items) |item| if (item != .integer or item.integer < 0) return errorValue(ctx, "checkpoint.load offsets must be non-negative integers");
        const start = std.math.cast(usize, offsets.array.items[0].integer) orelse return errorValue(ctx, "checkpoint.load offset is too large");
        const end = std.math.cast(usize, offsets.array.items[1].integer) orelse return errorValue(ctx, "checkpoint.load offset is too large");
        if (end < start or end > raw_size) return errorValue(ctx, "checkpoint.load offsets exceed file size");
        const byte_count = std.math.mul(usize, elements, if (is_bf16) @as(usize, 2) else dtype.size()) catch return errorValue(ctx, "checkpoint.load tensor is too large");
        if (byte_count != end - start) return errorValue(ctx, "checkpoint.load tensor byte length mismatch");
        const name_z = allocator.dupeZ(u8, name) catch return errorValue(ctx, "checkpoint.load out of memory");
        defer allocator.free(name_z);
        if (inspect) {
            const info = abi.jsNewObject(ctx);
            const shape = abi.jsNewArray(ctx);
            for (dimensions, 0..) |dimension, index| if (abi.jsSetArrayElement(ctx, shape, @intCast(index), abi.jsFloat64(ctx, @floatFromInt(dimension))) < 0) return errorValue(ctx, "checkpoint.inspect failed to create shape");
            if (abi.jsSetProperty(ctx, info, "shape", shape) < 0 or abi.jsSetProperty(ctx, info, "dtype", abi.jsString(ctx, dtype_text.string)) < 0 or abi.jsSetProperty(ctx, result, name_z.ptr, info) < 0) return errorValue(ctx, "checkpoint.inspect failed to create result");
            continue;
        }
        if (selective) {
            var wanted = false;
            for (selected.items) |requested| if (std.mem.eql(u8, requested, name)) { wanted = true; break; };
            if (!wanted) continue;
            matched += 1;
        }
        file.seekTo(8 + header_size + start) catch return errorValue(ctx, "checkpoint.load seek failed");
        const output_byte_count = std.math.mul(usize, elements, dtype.size()) catch return errorValue(ctx, "checkpoint.load tensor is too large");
        const bytes = allocator.alloc(u8, output_byte_count) catch return errorValue(ctx, "checkpoint.load out of memory");
        defer allocator.free(bytes);
        if (is_bf16) {
            var buffer: [64 * 1024]u8 = undefined;
            var offset: usize = 0;
            while (offset < elements) {
                const count: usize = @min(buffer.len / 2, elements - offset);
                readCheckpointBytes(file, buffer[0 .. count * 2]) catch return errorValue(ctx, "checkpoint.load truncated tensor");
                for (0..count) |index| {
                    const bits: u32 = @as(u32, std.mem.readInt(u16, buffer[index * 2 ..][0..2], .little)) << 16;
                    @memcpy(bytes[(offset + index) * 4 ..][0..4], std.mem.asBytes(&bits));
                }
                offset += count;
            }
        } else readCheckpointBytes(file, bytes) catch return errorValue(ctx, "checkpoint.load truncated tensor");
        var spec = compute.TensorSpec.init(allocator, dtype, dimensions, null) catch return errorValue(ctx, "checkpoint.load invalid Tensor spec");
        defer spec.deinit();
        const session = sessionFor(.cpu) catch return errorValue(ctx, "checkpoint.load CPU session unavailable");
        const tensor = session.createTensor(spec, bytes) catch return errorValue(ctx, "checkpoint.load failed to create Tensor");
        const js_value = createTensorObject(ctx, tensor, .cpu);
        if (abi.jsIsException(js_value)) return js_value;
        if (abi.jsSetProperty(ctx, result, name_z.ptr, js_value) < 0) return errorValue(ctx, "checkpoint.load failed to create result");
    }
    if (selective and matched != selected.items.len) return errorValue(ctx, "Missing requested checkpoint tensor");
    owns_result = false;
    return result;
}

const functions = [_]abi.JSFunction{
    .{ .name = "defaultDevice", .callback = jsDefaultDevice, .length = 0 },
    .{ .name = "createSession", .callback = jsCreateSession, .length = 2 },
    .{ .name = "sessionTensor", .callback = jsSessionTensor, .length = 4 },
    .{ .name = "sessionFull", .callback = jsSessionFull, .length = 4 },
    .{ .name = "compileProgram", .callback = jsCompileProgram, .length = 3 },
    .{ .name = "explainExecutable", .callback = jsExplainExecutable, .length = 1 },
    .{ .name = "runExecutable", .callback = jsRunExecutable, .length = 2 },
    .{ .name = "stftPower", .callback = jsStftPower, .length = 5 },
    .{ .name = "filterbank", .callback = jsFilterbank, .length = 2 },
    .{ .name = "saveNative", .callback = jsSaveNative, .length = 2 },
    .{ .name = "loadNative", .callback = jsLoadNative, .length = 2 },
    .{ .name = "inspectCheckpoint", .callback = jsInspectCheckpoint, .length = 1 },
};
const function_ptrs = blk: { var pointers: [functions.len]*const abi.JSFunction = undefined; for (&functions, 0..) |*function, index| pointers[index] = function; break :blk pointers; };
pub const specifier: [:0]const u8 = "affon:compute/native";
pub fn load(ctx: ?*anyopaque, module_name: [*c]const u8) ?*anyopaque { return @ptrCast(abi.createJSFunctionModule(allocator, @ptrCast(ctx), module_name, &function_ptrs)); }
