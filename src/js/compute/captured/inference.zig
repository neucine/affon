const std = @import("std");
const hao = @import("hao");
const compute = @import("compute");
const engine_api = compute;
const tensor_types = compute.tensor;
const SliceRange = compute.operation.SliceRange;
const semantic = compute.sema;
const lowering = @import("lowering.zig");
const captured_op_info = @import("op_info.zig");

const abi = hao.js.abi;
const allocator = std.heap.c_allocator;

fn errorValue(ctx: abi.JSContext, message: [*:0]const u8) abi.JSValue {
    return abi.jsThrowError(ctx, message);
}

fn typeError(ctx: abi.JSContext, message: [*:0]const u8) abi.JSValue {
    return abi.jsThrowTypeError(ctx, message);
}

fn classifiedErrorValue(ctx: abi.JSContext, message: []const u8, err: anyerror) abi.JSValue {
    return switch (err) {
        error.InvalidArgument,
        error.InvalidAxis,
        error.IndexOutOfBounds,
        error.AxisOutOfBounds,
        error.InvalidSliceStep,
        error.EmptyInput,
        error.TooManySliceDimensions,
        error.UnsupportedDevice,
        error.InvalidInputCount,
        error.InputCountMismatch,
        error.InvalidOpOptions,
        error.InvalidOutputCount,
        error.InvalidClampBounds,
        error.InvalidIndexDType,
        error.InvalidTopK,
        error.UnknownOp,
        error.InvalidNumericValue,
        error.InvalidSliceBound,
        error.NegativeSliceStepNotYetSupported,
        => abi.jsThrowConstructedError(ctx, "AffonError", "invalid_arg", message),
        error.ShapeMismatch, error.IncompatibleShapes, error.SizeMismatch => abi.jsThrowConstructedError(ctx, "AffonError", "shape_mismatch", message),
        error.JaggedArray,
        error.NotContiguous,
        error.DimensionNotSingleton,
        error.MatmulRequires2D,
        error.DotRequires1D,
        error.UnsupportedNDim,
        error.UnsupportedShape,
        => abi.jsThrowConstructedError(ctx, "AffonError", "invalid_shape", message),
        error.DTypeMismatch, error.TypeMismatch, error.UnsupportedDType => abi.jsThrowConstructedError(ctx, "AffonError", "invalid_dtype", message),
        error.DeviceMismatch, error.MixedDeviceUnsupported => abi.jsThrowConstructedError(ctx, "AffonError", "device_mismatch", message),
        error.MetalKernelFailed,
        error.MetalTransferFailed,
        error.MetalUnavailable,
        error.MetalKernelLaunchFailed,
        => abi.jsThrowConstructedError(ctx, "AffonError", "device_error", message),
        error.ExecutionNotImplemented,
        error.InvalidExecutionPlan,
        error.InvalidGraphPlan,
        error.UnsupportedGraphLowering,
        error.UnboundGraphInput,
        error.UnboundGraphOutput,
        error.InputNotMaterialized,
        error.InputNotContiguous,
        error.MultiOutputRequiresExecuteAll,
        error.OpNotImplemented,
        => abi.jsThrowConstructedError(ctx, "AffonError", "invalid_state", message),
        error.OutOfMemory => abi.jsThrowConstructedError(ctx, "AffonError", "out_of_memory", message),
        else => abi.jsThrowConstructedError(ctx, "AffonError", "internal", message),
    };
}

fn parseDType(name: []const u8) !engine_api.DType {
    if (std.mem.eql(u8, name, "f32")) return .f32;
    if (std.mem.eql(u8, name, "f64")) return .f64;
    if (std.mem.eql(u8, name, "i64")) return .i64;
    return error.InvalidDType;
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

fn readRequiredUsizeProperty(ctx: abi.JSContext, object: abi.JSValueConst, name: [:0]const u8) !usize {
    const value = abi.jsGetProperty(ctx, object, name);
    defer abi.jsFreeValue(ctx, value);
    if (abi.jsIsUndefined(value) or abi.jsIsNull(value)) return error.MissingOption;
    return integerArgument(ctx, value, name.ptr) orelse error.InvalidOption;
}

fn readOptionalUsizeProperty(ctx: abi.JSContext, object: abi.JSValueConst, name: [:0]const u8) !?usize {
    if (abi.jsIsUndefined(object) or abi.jsIsNull(object)) return null;
    const value = abi.jsGetProperty(ctx, object, name);
    defer abi.jsFreeValue(ctx, value);
    if (abi.jsIsUndefined(value) or abi.jsIsNull(value)) return null;
    return integerArgument(ctx, value, name.ptr) orelse error.InvalidOption;
}

fn readRequiredFloatProperty(ctx: abi.JSContext, object: abi.JSValueConst, name: [:0]const u8) !f64 {
    const value = abi.jsGetProperty(ctx, object, name);
    defer abi.jsFreeValue(ctx, value);
    if (abi.jsIsUndefined(value) or abi.jsIsNull(value)) return error.MissingOption;
    var number: f64 = 0;
    if (abi.jsToFloat64(ctx, &number, value) < 0) return error.InvalidOption;
    return number;
}

fn readOptionalI64Property(ctx: abi.JSContext, object: abi.JSValueConst, name: [:0]const u8) !?i64 {
    if (abi.jsIsUndefined(object) or abi.jsIsNull(object)) return null;
    const value = abi.jsGetProperty(ctx, object, name);
    defer abi.jsFreeValue(ctx, value);
    if (abi.jsIsUndefined(value) or abi.jsIsNull(value)) return null;
    var integer: i32 = 0;
    if (abi.jsToInt32(ctx, &integer, value) < 0) return error.InvalidOption;
    return @intCast(integer);
}

fn readRequiredI64Property(ctx: abi.JSContext, object: abi.JSValueConst, name: [:0]const u8) !i64 {
    return try readOptionalI64Property(ctx, object, name) orelse error.MissingOption;
}

fn readOptionalIsizeProperty(ctx: abi.JSContext, object: abi.JSValueConst, name: [:0]const u8, default: isize) !isize {
    if (try readOptionalI64Property(ctx, object, name)) |value| return @intCast(value);
    return default;
}

fn readOptionalBoolProperty(ctx: abi.JSContext, object: abi.JSValueConst, name: [:0]const u8, default: bool) !bool {
    if (abi.jsIsUndefined(object) or abi.jsIsNull(object)) return default;
    const value = abi.jsGetProperty(ctx, object, name);
    defer abi.jsFreeValue(ctx, value);
    if (abi.jsIsUndefined(value) or abi.jsIsNull(value)) return default;
    var result = default;
    if (abi.jsToBool(ctx, &result, value) < 0) return error.InvalidOption;
    return result;
}

fn readOwnedUsizeArrayProperty(ctx: abi.JSContext, object: abi.JSValueConst, name: [:0]const u8) ![]usize {
    const value = abi.jsGetProperty(ctx, object, name);
    defer abi.jsFreeValue(ctx, value);
    return readShape(ctx, value, name.ptr) orelse error.InvalidOption;
}

fn readOwnedSliceSelectorsProperty(ctx: abi.JSContext, object: abi.JSValueConst, name: [:0]const u8) ![]semantic.slice.SliceSelector {
    const selectors_value = abi.jsGetProperty(ctx, object, name);
    defer abi.jsFreeValue(ctx, selectors_value);
    if (!abi.jsIsArray(ctx, selectors_value)) return error.InvalidOption;
    const length_value = abi.jsGetProperty(ctx, selectors_value, "length");
    defer abi.jsFreeValue(ctx, length_value);
    const length = integerArgument(ctx, length_value, name.ptr) orelse return error.InvalidOption;
    const selectors = try allocator.alloc(semantic.slice.SliceSelector, length);
    errdefer allocator.free(selectors);
    for (selectors, 0..) |*selector, index| {
        const item = abi.jsGetArrayElement(ctx, selectors_value, @intCast(index));
        defer abi.jsFreeValue(ctx, item);
        const kind_value = abi.jsGetProperty(ctx, item, "kind");
        defer abi.jsFreeValue(ctx, kind_value);
        if (abi.jsStringEquals(ctx, kind_value, "all")) {
            selector.* = .all;
        } else if (abi.jsStringEquals(ctx, kind_value, "index")) {
            selector.* = .{ .index = try readRequiredI64Property(ctx, item, "index") };
        } else if (abi.jsStringEquals(ctx, kind_value, "range")) {
            selector.* = .{ .range = .{
                .start = try readOptionalI64Property(ctx, item, "start"),
                .stop = try readOptionalI64Property(ctx, item, "stop"),
                .step = try readOptionalIsizeProperty(ctx, item, "step", 1),
            } };
        } else {
            return error.InvalidOption;
        }
    }
    return selectors;
}

const OptionStorage = struct {
    shape: ?[]usize = null,
    axes: ?[]usize = null,
    ranges: ?[]SliceRange = null,

    fn deinit(self: *OptionStorage) void {
        if (self.shape) |value| allocator.free(value);
        if (self.axes) |value| allocator.free(value);
        if (self.ranges) |value| allocator.free(value);
        self.* = undefined;
    }
};

fn opInfoFromOptions(ctx: abi.JSContext, kind: []const u8, options: abi.JSValueConst, storage: *OptionStorage) !lowering.OpInfo {
    if (captured_op_info.noOptionKind(kind)) |info| return info;
    if (std.mem.eql(u8, kind, "matmul")) return .{ .tag = .matmul, .options = .{ .none = {} }, .execution_metadata = .{} };
    if (std.mem.eql(u8, kind, "transpose")) return captured_op_info.transpose();
    if (std.mem.eql(u8, kind, "reshape")) {
        storage.shape = try readOwnedUsizeArrayProperty(ctx, options, "shape");
        return .{ .tag = .reshape, .options = .{ .reshape = .{ .shape = storage.shape.? } }, .execution_metadata = .{} };
    }
    if (std.mem.eql(u8, kind, "squeeze")) return .{ .tag = .squeeze, .options = .{ .squeeze = .{ .axis = try readOptionalUsizeProperty(ctx, options, "axis") } }, .execution_metadata = .{} };
    if (std.mem.eql(u8, kind, "unsqueeze")) return .{ .tag = .unsqueeze, .options = .{ .unsqueeze = .{ .axis = try readRequiredUsizeProperty(ctx, options, "axis") } }, .execution_metadata = .{} };
    if (std.mem.eql(u8, kind, "permute")) {
        storage.axes = try readOwnedUsizeArrayProperty(ctx, options, "axes");
        return .{ .tag = .permute, .options = .{ .permute = .{ .axes = storage.axes.? } }, .execution_metadata = .{} };
    }
    if (std.mem.eql(u8, kind, "cast")) {
        const dtype_value = abi.jsGetProperty(ctx, options, "dtype");
        defer abi.jsFreeValue(ctx, dtype_value);
        const dtype_name = abi.jsStringAlloc(ctx, dtype_value, allocator) catch return error.InvalidOption;
        defer allocator.free(dtype_name);
        return .{ .tag = .cast, .options = .{ .cast = .{ .to = try parseDType(dtype_name) } }, .execution_metadata = .{} };
    }
    if (std.mem.eql(u8, kind, "clamp")) return .{ .tag = .clamp, .options = .{ .clamp = .{ .min = try readRequiredFloatProperty(ctx, options, "min"), .max = try readRequiredFloatProperty(ctx, options, "max") } }, .execution_metadata = .{} };
    if (std.mem.eql(u8, kind, "masked_fill")) return .{ .tag = .masked_fill, .options = .{ .masked_fill = .{ .value = try readRequiredFloatProperty(ctx, options, "value") } }, .execution_metadata = .{} };
    if (std.mem.eql(u8, kind, "softmax")) return captured_op_info.softmax(try readRequiredUsizeProperty(ctx, options, "dim"));
    if (std.mem.eql(u8, kind, "cross_entropy_indexed")) return captured_op_info.crossEntropyIndexed(try readRequiredUsizeProperty(ctx, options, "axis"));
    if (std.mem.eql(u8, kind, "one_hot")) return .{ .tag = .one_hot, .options = .{ .one_hot = .{ .num_classes = try readRequiredUsizeProperty(ctx, options, "numClasses") } }, .execution_metadata = .{} };
    if (std.mem.eql(u8, kind, "gather")) return captured_op_info.gather(try readRequiredUsizeProperty(ctx, options, "dim"));
    if (std.mem.eql(u8, kind, "index_select")) return captured_op_info.indexSelect(try readRequiredUsizeProperty(ctx, options, "dim"));
    if (std.mem.eql(u8, kind, "topk_values") or std.mem.eql(u8, kind, "topk_indices")) {
        return captured_op_info.topk(
            kind,
            try readRequiredUsizeProperty(ctx, options, "k"),
            try readRequiredUsizeProperty(ctx, options, "dim"),
            try readOptionalBoolProperty(ctx, options, "largest", true),
            try readOptionalBoolProperty(ctx, options, "sorted", true),
        ) orelse error.UnsupportedCapturedInference;
    }
    if (captured_op_info.reduction(kind, try readOptionalUsizeProperty(ctx, options, "axis"), try readOptionalBoolProperty(ctx, options, "keepdim", false))) |info| return info;
    if (std.mem.eql(u8, kind, "cat")) return captured_op_info.concat(try readRequiredUsizeProperty(ctx, options, "dim"));
    if (std.mem.eql(u8, kind, "stack")) return captured_op_info.stack(try readRequiredUsizeProperty(ctx, options, "dim"));
    return error.UnsupportedCapturedInference;
}

fn readTensorSpecs(ctx: abi.JSContext, value: abi.JSValueConst) ![]tensor_types.TensorSpec {
    if (!abi.jsIsArray(ctx, value)) return error.InvalidInputSpec;
    const length_value = abi.jsGetProperty(ctx, value, "length");
    defer abi.jsFreeValue(ctx, length_value);
    const length = integerArgument(ctx, length_value, "inputs length") orelse return error.InvalidInputSpec;
    const specs = try allocator.alloc(tensor_types.TensorSpec, length);
    var initialized: usize = 0;
    errdefer {
        for (specs[0..initialized]) |*spec| {
            spec.layout.deinit();
            spec.shape.deinit();
        }
        allocator.free(specs);
    }
    for (specs, 0..) |*spec, index| {
        const item = abi.jsGetArrayElement(ctx, value, @intCast(index));
        defer abi.jsFreeValue(ctx, item);
        const shape_value = abi.jsGetProperty(ctx, item, "shape");
        defer abi.jsFreeValue(ctx, shape_value);
        const dims = readShape(ctx, shape_value, "input shape") orelse return error.InvalidInputSpec;
        defer allocator.free(dims);
        var shape = try tensor_types.Shape.initCopy(allocator, dims);
        errdefer shape.deinit();
        var layout = try tensor_types.Layout.initContiguous(allocator, shape);
        errdefer layout.deinit();
        const dtype_value = abi.jsGetProperty(ctx, item, "dtype");
        defer abi.jsFreeValue(ctx, dtype_value);
        const dtype_name = abi.jsStringAlloc(ctx, dtype_value, allocator) catch return error.InvalidInputSpec;
        defer allocator.free(dtype_name);
        const device_value = abi.jsGetProperty(ctx, item, "device");
        defer abi.jsFreeValue(ctx, device_value);
        const device: engine_api.Device = if (abi.jsIsUndefined(device_value) or abi.jsIsNull(device_value) or abi.jsStringEquals(ctx, device_value, "cpu"))
            .cpu
        else if (abi.jsStringEquals(ctx, device_value, "metal"))
            .metal
        else
            return error.InvalidInputSpec;
        spec.* = .{
            .shape = shape,
            .dtype = try parseDType(dtype_name),
            .layout = layout,
            .device = device,
            .axes = null,
        };
        initialized += 1;
    }
    return specs;
}

fn freeTensorSpecs(specs: []tensor_types.TensorSpec) void {
    for (specs) |*spec| {
        spec.layout.deinit();
        spec.shape.deinit();
    }
    allocator.free(specs);
}

fn createResult(ctx: abi.JSContext, shape_dims: []const usize, dtype: engine_api.DType, device: engine_api.Device) abi.JSValue {
    const result = abi.jsNewObject(ctx);
    if (abi.jsIsException(result)) return result;
    errdefer abi.jsFreeValue(ctx, result);
    const shape = abi.jsNewArray(ctx);
    if (abi.jsIsException(shape)) return errorValue(ctx, "failed to create inference shape");
    var shape_owned = true;
    defer if (shape_owned) abi.jsFreeValue(ctx, shape);
    for (shape_dims, 0..) |dim, index| {
        if (abi.jsSetArrayElement(ctx, shape, @intCast(index), abi.jsInt32(ctx, @intCast(dim))) < 0) return errorValue(ctx, "failed to create inference shape");
    }
    const dtype_name: [:0]const u8 = switch (dtype) {
        .f32 => "f32",
        .f64 => "f64",
        .i64 => "i64",
    };
    const device_name: [:0]const u8 = switch (device) {
        .cpu => "cpu",
        .metal => "metal",
        .cuda => "cuda",
    };
    if (abi.jsSetProperty(ctx, result, "shape", shape) < 0 or
        abi.jsSetProperty(ctx, result, "dtype", abi.jsString(ctx, dtype_name)) < 0 or
        abi.jsSetProperty(ctx, result, "device", abi.jsString(ctx, device_name)) < 0)
    {
        return errorValue(ctx, "failed to create inference result");
    }
    shape_owned = false;
    return result;
}

pub fn jsInferCapturedOp(ctx: abi.JSContext, _: abi.JSValueConst, argc: c_int, argv: [*c]abi.JSValueConst) callconv(.c) abi.JSValue {
    if (argc < 2 or argc > 3) return typeError(ctx, "inferCapturedOp expects kind, inputs, and optional options");
    const kind = abi.jsStringAlloc(ctx, argv[0], allocator) catch return typeError(ctx, "inferCapturedOp kind must be a string");
    defer allocator.free(kind);
    const specs = readTensorSpecs(ctx, argv[1]) catch return errorValue(ctx, "inferCapturedOp failed to read input specs");
    defer freeTensorSpecs(specs);
    const options_value = if (argc >= 3) argv[2] else abi.jsUndefined(ctx);
    var option_storage: OptionStorage = .{};
    defer option_storage.deinit();
    const op_info: lowering.OpInfo = if (std.mem.eql(u8, kind, "slice")) blk: {
        if (specs.len != 1) return errorValue(ctx, "inferCapturedOp slice expects one input");
        const selectors = readOwnedSliceSelectorsProperty(ctx, options_value, "slice_selectors") catch return errorValue(ctx, "inferCapturedOp failed to read slice selectors");
        defer allocator.free(selectors);
        option_storage.ranges = semantic.slice.normalizeSelectors(allocator, specs[0].shape.dims, selectors) catch |err| return classifiedErrorValue(ctx, "inferCapturedOp failed", err);
        break :blk .{ .tag = .slice, .options = .{ .slice = .{ .ranges = option_storage.ranges.? } }, .execution_metadata = .{} };
    } else opInfoFromOptions(ctx, kind, options_value, &option_storage) catch return errorValue(ctx, "inferCapturedOp failed to read op options");
    var inferred = semantic.inferFromSpecs(allocator, op_info.tag, specs, op_info.options) catch |err| return classifiedErrorValue(ctx, "inferCapturedOp failed", err);
    defer inferred.deinit();
    if (std.mem.eql(u8, kind, "topk_indices")) {
        const secondary = inferred.secondary_output orelse return errorValue(ctx, "inferCapturedOp topk did not produce indices");
        return createResult(ctx, secondary.shape.dims, secondary.dtype, inferred.device);
    }
    return createResult(ctx, inferred.shape.dims, inferred.dtype, inferred.device);
}
