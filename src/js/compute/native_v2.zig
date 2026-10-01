const std = @import("std");
const hao = @import("hao");
const compute = @import("compute");
const config = @import("../../config.zig");

const abi = hao.js.abi;
const allocator = std.heap.c_allocator;
const tensor_type_id: u32 = 1;

const TensorObject = struct {
    value: *compute.Tensor,
    backend: compute.Backend,
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

fn tensorFinalizer(_: u32, handle: u64) callconv(.c) void {
    const object: *TensorObject = @ptrFromInt(@as(usize, @intCast(handle)));
    object.value.deinit();
    allocator.destroy(object);
}

fn errorValue(ctx: abi.JSContext, message: [*:0]const u8) abi.JSValue { return abi.jsThrowError(ctx, message); }
fn typeError(ctx: abi.JSContext, message: [*:0]const u8) abi.JSValue { return abi.jsThrowTypeError(ctx, message); }

fn createTensorObject(ctx: abi.JSContext, value: *compute.Tensor, backend: compute.Backend) abi.JSValue {
    const object = allocator.create(TensorObject) catch { value.deinit(); return errorValue(ctx, "out of memory"); };
    object.* = .{ .value = value, .backend = backend };
    const result = abi.createJSHostObject(ctx, tensor_type_id, @intCast(@intFromPtr(object)), tensorFinalizer);
    if (abi.jsIsException(result)) { value.deinit(); allocator.destroy(object); return result; }
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
        abi.jsSetFunction(ctx, result, "repr", jsToString, 0) < 0)
        return errorValue(ctx, "failed to create Tensor object");
    return result;
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
fn jsToArray(ctx: abi.JSContext, this_value: abi.JSValueConst, _: c_int, _: [*c]abi.JSValueConst) callconv(.c) abi.JSValue { const object = tensorObject(ctx, this_value) orelse return typeError(ctx, "invalid Tensor"); var index: usize = 0; return arrayValue(ctx, object.value, 0, &index); }
fn jsItem(ctx: abi.JSContext, this_value: abi.JSValueConst, _: c_int, _: [*c]abi.JSValueConst) callconv(.c) abi.JSValue { const object = tensorObject(ctx, this_value) orelse return typeError(ctx, "invalid Tensor"); if (elementCount(object.value.spec().dimensions()) != 1) return typeError(ctx, "item requires one element"); return scalar(ctx, object.value, 0); }
fn jsToString(ctx: abi.JSContext, this_value: abi.JSValueConst, _: c_int, _: [*c]abi.JSValueConst) callconv(.c) abi.JSValue { const object = tensorObject(ctx, this_value) orelse return typeError(ctx, "invalid Tensor"); var buffer: [128]u8 = undefined; const text = std.fmt.bufPrint(&buffer, "Tensor(shape={any}, dtype={s})", .{ object.value.spec().dimensions(), @tagName(object.value.spec().dtype()) }) catch return errorValue(ctx, "repr failed"); return abi.jsString(ctx, text); }

const Flat = struct { shape: std.ArrayList(usize) = .empty, values: std.ArrayList(f32) = .empty, fn deinit(self: *Flat) void { self.shape.deinit(allocator); self.values.deinit(allocator); } };
fn flatten(ctx: abi.JSContext, value: abi.JSValueConst, depth: usize, result: *Flat) !void {
    if (abi.jsIsArray(ctx, value)) {
        const length_value = abi.jsGetProperty(ctx, value, "length"); defer abi.jsFreeValue(ctx, length_value);
        var length: i32 = 0; if (abi.jsToInt32(ctx, &length, length_value) < 0 or length < 0) return error.InvalidInput;
        if (result.shape.items.len == depth) try result.shape.append(allocator, @intCast(length)) else if (result.shape.items[depth] != length) return error.JaggedArray;
        for (0..@as(usize, @intCast(length))) |index| { const item = abi.jsGetArrayElement(ctx, value, @intCast(index)); defer abi.jsFreeValue(ctx, item); try flatten(ctx, item, depth + 1, result); }
    } else { if (depth != result.shape.items.len) return error.JaggedArray; var number: f64 = 0; if (abi.jsToFloat64(ctx, &number, value) < 0) return error.InvalidInput; try result.values.append(allocator, @floatCast(number)); }
}
fn jsTensor(ctx: abi.JSContext, _: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    if (argc < 1) return typeError(ctx, "tensor expects values");
    var flat = Flat{}; defer flat.deinit(); flatten(ctx, argv[0], 0, &flat) catch return typeError(ctx, "tensor expects a rectangular numeric value");
    var spec = compute.TensorSpec.init(allocator, .f32, flat.shape.items, null) catch return errorValue(ctx, "tensor spec failed"); defer spec.deinit();
    const backend = defaultBackend(); const session = sessionFor(backend) catch return errorValue(ctx, "backend unavailable");
    const value = session.createTensor(spec, std.mem.sliceAsBytes(flat.values.items)) catch return errorValue(ctx, "tensor creation failed");
    return createTensorObject(ctx, value, backend);
}

fn runBinary(ctx: abi.JSContext, tag: compute.OpTag, lhs: *TensorObject, rhs: *TensorObject) abi.JSValue {
    if (lhs.backend != rhs.backend) return typeError(ctx, "operands must use one backend");
    var builder = compute.ProgramBuilder.init(allocator); defer builder.deinit();
    const a = builder.addInput(lhs.value.spec().*) catch return errorValue(ctx, "program input failed");
    const b = builder.addInput(rhs.value.spec().*) catch return errorValue(ctx, "program input failed");
    const output = builder.add(tag, .{ .none = {} }, &.{ a, b }) catch return errorValue(ctx, "operation validation failed");
    const program = builder.finish(output) catch return errorValue(ctx, "program creation failed"); defer program.deinit();
    const session = sessionFor(lhs.backend) catch return errorValue(ctx, "backend unavailable"); var compilation = session.compile(program, .{}) catch return errorValue(ctx, "compilation failed"); defer compilation.deinit();
    const outputs = session.run(compilation.executable, &.{ lhs.value, rhs.value }) catch return errorValue(ctx, "execution failed");
    if (outputs.len != 1) { session.releaseOutputs(outputs); return errorValue(ctx, "invalid output count"); }
    const result = outputs[0]; allocator.free(outputs); return createTensorObject(ctx, result, lhs.backend);
}
fn binary(tag: compute.OpTag) type { return struct { fn call(ctx: abi.JSContext, _: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue { if (argc != 2) return typeError(ctx, "binary operation expects two Tensors"); const lhs = tensorObject(ctx, argv[0]) orelse return typeError(ctx, "expected Tensor"); const rhs = tensorObject(ctx, argv[1]) orelse return typeError(ctx, "expected Tensor"); return runBinary(ctx, tag, lhs, rhs); } }; }

const functions = [_]abi.JSFunction{
    .{ .name = "tensor", .callback = jsTensor, .length = 1 },
    .{ .name = "add", .callback = binary(.add).call, .length = 2 },
    .{ .name = "sub", .callback = binary(.sub).call, .length = 2 },
    .{ .name = "mul", .callback = binary(.mul).call, .length = 2 },
    .{ .name = "div", .callback = binary(.div).call, .length = 2 },
    .{ .name = "matmul", .callback = binary(.matmul).call, .length = 2 },
    .{ .name = "dot", .callback = binary(.dot).call, .length = 2 },
};
const function_ptrs = blk: { var pointers: [functions.len]*const abi.JSFunction = undefined; for (&functions, 0..) |*function, index| pointers[index] = function; break :blk pointers; };
pub const specifier: [:0]const u8 = "affon:compute/native";
pub fn load(ctx: ?*anyopaque, module_name: [*c]const u8) ?*anyopaque { return @ptrCast(abi.createJSFunctionModule(allocator, @ptrCast(ctx), module_name, &function_ptrs)); }
