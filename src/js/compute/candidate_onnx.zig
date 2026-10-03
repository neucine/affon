const std = @import("std");
const compute = @import("compute_candidate");

pub const Result = struct {
    values: []f32,
    explanation: []u8,
    onnx: Comparison,
    pytorch: Comparison,
    top1: usize,
    pub fn deinit(self: *Result, allocator: std.mem.Allocator) void {
        allocator.free(self.values);
        allocator.free(self.explanation);
        self.* = undefined;
    }
};

pub const Comparison = struct {
    passed: bool,
    mismatches: usize,
    max_absolute_error: f32,
};

const TensorBytes = struct { shape: []const usize, bytes: []const u8 };

pub fn run(allocator: std.mem.Allocator, io: std.Io, directory: []const u8, backend: compute.Backend, case_index: usize) !Result {
    return runWithOptions(allocator, io, directory, backend, case_index, .{});
}

pub fn runWithOptions(allocator: std.mem.Allocator, io: std.Io, directory: []const u8, backend: compute.Backend, case_index: usize, compile_options: compute.CompileOptions) !Result {
    var scratch_arena = std.heap.ArenaAllocator.init(allocator);
    defer scratch_arena.deinit();
    const scratch = scratch_arena.allocator();
    const graph_path = try std.fmt.allocPrint(allocator, "{s}/graph.json", .{directory});
    defer allocator.free(graph_path);
    const weights_path = try std.fmt.allocPrint(allocator, "{s}/weights.safetensors", .{directory});
    defer allocator.free(weights_path);
    const reference_path = try std.fmt.allocPrint(allocator, "{s}/reference.safetensors", .{directory});
    defer allocator.free(reference_path);
    const graph_bytes = try std.Io.Dir.cwd().readFileAlloc(io, graph_path, allocator, .limited(4 * 1024 * 1024));
    defer allocator.free(graph_bytes);
    const weights_bytes = try std.Io.Dir.cwd().readFileAlloc(io, weights_path, allocator, .limited(64 * 1024 * 1024));
    defer allocator.free(weights_bytes);
    const reference_bytes = try std.Io.Dir.cwd().readFileAlloc(io, reference_path, allocator, .limited(16 * 1024 * 1024));
    defer allocator.free(reference_bytes);
    var graph = try std.json.parseFromSlice(std.json.Value, allocator, graph_bytes, .{});
    defer graph.deinit();
    var weights = try parseSafetensors(scratch, weights_bytes);
    defer weights.deinit();
    var references = try parseSafetensors(scratch, reference_bytes);
    defer references.deinit();

    const root = try object(graph.value);
    if (!std.mem.eql(u8, try string(root.get("format")), "affon-onnx-static/v1")) return error.InvalidManifest;
    const inputs = try object(root.get("inputs"));
    if (inputs.count() != 1) return error.UnsupportedManifest;
    var input_iterator = inputs.iterator();
    const input_entry = input_iterator.next().?;
    const input_shape = try dimensions(scratch, input_entry.value_ptr.*);
    var input_spec = try compute.TensorSpec.init(allocator, .f32, input_shape, null);
    defer input_spec.deinit();
    var builder = compute.ProgramBuilder.init(allocator);
    defer builder.deinit();
    var values = std.StringHashMap(compute.ValueId).init(allocator);
    defer values.deinit();
    var shapes = std.StringHashMap([]const usize).init(allocator);
    defer shapes.deinit();
    try values.put(input_entry.key_ptr.*, try builder.addInput(input_spec));
    try shapes.put(input_entry.key_ptr.*, input_shape);

    const constants = try object(root.get("constants"));
    var constant_iterator = constants.iterator();
    while (constant_iterator.next()) |entry| {
        const expected_shape = try dimensions(scratch, entry.value_ptr.*);
        const tensor = weights.get(entry.key_ptr.*) orelse return error.MissingWeight;
        if (!std.mem.eql(usize, expected_shape, tensor.shape)) return error.WeightShapeMismatch;
        var spec = try compute.TensorSpec.init(allocator, .f32, expected_shape, null);
        defer spec.deinit();
        try values.put(entry.key_ptr.*, try builder.addConstant(spec, .{ .bytes = tensor.bytes }));
        try shapes.put(entry.key_ptr.*, expected_shape);
    }

    const nodes = try array(root.get("nodes"));
    for (nodes.items) |node_value| {
        const node = try object(node_value);
        const op = try string(node.get("op"));
        const name = try string(node.get("name"));
        const output_name = try string(node.get("output"));
        const output_shape = try dimensions(scratch, node.get("shape"));
        const node_inputs = try array(node.get("inputs"));
        const attrs = try object(node.get("attrs"));
        const output = if (std.mem.eql(u8, op, "Pad"))
            try lowerPad(allocator, scratch, &builder, &values, &shapes, node_inputs, attrs)
        else if (std.mem.eql(u8, op, "Conv"))
            try lowerConv(scratch, &builder, &values, &shapes, node_inputs, attrs, output_shape)
        else if (std.mem.eql(u8, op, "Clip"))
            (try builder.add(.clamp, .{ .clamp = .{ .min = try number(attrs.get("min")), .max = try number(attrs.get("max")) } }, &.{try inputId(&values, node_inputs, 0)}))[0]
        else if (std.mem.eql(u8, op, "Add"))
            (try builder.add(.add, .{ .none = {} }, &.{ try inputId(&values, node_inputs, 0), try inputId(&values, node_inputs, 1) }))[0]
        else if (std.mem.eql(u8, op, "GlobalAveragePool")) blk: {
            const width = try builder.add(.mean_axis, .{ .reduce_axis = .{ .axis = 3, .keepdim = true } }, &.{try inputId(&values, node_inputs, 0)});
            break :blk (try builder.add(.mean_axis, .{ .reduce_axis = .{ .axis = 2, .keepdim = true } }, width))[0];
        } else if (std.mem.eql(u8, op, "Flatten"))
            (try builder.add(.reshape, .{ .reshape = .{ .shape = output_shape } }, &.{try inputId(&values, node_inputs, 0)}))[0]
        else if (std.mem.eql(u8, op, "Gemm"))
            try lowerGemm(&builder, &values, node_inputs, attrs)
        else {
            std.log.err("unsupported candidate ONNX node {s}: {s}", .{ name, op });
            return error.UnsupportedOperation;
        };
        try values.put(output_name, output);
        try shapes.put(output_name, output_shape);
    }
    const output_names = try array(root.get("outputs"));
    if (output_names.items.len != 1) return error.UnsupportedManifest;
    const output_id = values.get(try string(output_names.items[0])) orelse return error.MissingOutput;
    const program = try builder.finish(&.{output_id});
    defer program.deinit();
    const session = try compute.createSession(allocator, backend);
    defer session.deinit() catch unreachable;
    const input_key = try std.fmt.allocPrint(allocator, "case_{d}.pixels", .{case_index});
    defer allocator.free(input_key);
    const reference_input = references.get(input_key) orelse return error.MissingReference;
    const input_tensor = try session.createTensor(input_spec, reference_input.bytes);
    defer input_tensor.deinit();
    var compilation = try session.compile(program, compile_options);
    defer compilation.deinit();
    var report = try compute.createExplanation(allocator, program, program, compilation.executable);
    defer report.deinit();
    const explanation = try report.serialize(allocator);
    errdefer allocator.free(explanation);
    const outputs = try session.run(compilation.executable, &.{input_tensor});
    defer session.releaseOutputs(outputs);
    const output_bytes: []align(4) const u8 = @alignCast(outputs[0].bytes());
    const output_values = std.mem.bytesAsSlice(f32, output_bytes);
    const onnx_key = try std.fmt.allocPrint(scratch, "case_{d}.onnx", .{case_index});
    const pytorch_key = try std.fmt.allocPrint(scratch, "case_{d}.pytorch", .{case_index});
    const onnx_reference = references.get(onnx_key) orelse return error.MissingReference;
    const pytorch_reference = references.get(pytorch_key) orelse return error.MissingReference;
    const aligned_onnx_bytes: []align(4) const u8 = @alignCast(onnx_reference.bytes);
    const aligned_pytorch_bytes: []align(4) const u8 = @alignCast(pytorch_reference.bytes);
    const onnx_values = std.mem.bytesAsSlice(f32, aligned_onnx_bytes);
    const pytorch_values = std.mem.bytesAsSlice(f32, aligned_pytorch_bytes);
    const onnx_comparison = try compare(output_values, onnx_values);
    const pytorch_comparison = try compare(output_values, pytorch_values);
    var top1: usize = 0;
    for (output_values[1..], 1..) |value, index| if (value > output_values[top1]) {
        top1 = index;
    };
    return .{
        .values = try allocator.dupe(f32, output_values),
        .explanation = explanation,
        .onnx = onnx_comparison,
        .pytorch = pytorch_comparison,
        .top1 = top1,
    };
}

fn lowerPad(allocator: std.mem.Allocator, scratch: std.mem.Allocator, builder: *compute.ProgramBuilder, values: *std.StringHashMap(compute.ValueId), shapes: *std.StringHashMap([]const usize), inputs: std.json.Array, attrs: std.json.ObjectMap) !compute.ValueId {
    var current = try inputId(values, inputs, 0);
    const source_name = try string(inputs.items[0]);
    const shape = shapes.get(source_name).?;
    if (shape.len != 4) return error.UnsupportedShape;
    const pads = try dimensions(scratch, attrs.get("pads"));
    if (pads.len != 4) return error.InvalidManifest;
    if (pads[0] > 0) current = try catZero(allocator, builder, current, shape, 2, pads[0], true);
    var height = shape[2] + pads[0];
    if (pads[2] > 0) {
        const grown = [_]usize{ shape[0], shape[1], height, shape[3] };
        current = try catZero(allocator, builder, current, &grown, 2, pads[2], false);
        height += pads[2];
    }
    const vertical = [_]usize{ shape[0], shape[1], height, shape[3] };
    if (pads[1] > 0) current = try catZero(allocator, builder, current, &vertical, 3, pads[1], true);
    if (pads[3] > 0) {
        const grown = [_]usize{ shape[0], shape[1], height, shape[3] + pads[1] };
        current = try catZero(allocator, builder, current, &grown, 3, pads[3], false);
    }
    return current;
}

fn catZero(allocator: std.mem.Allocator, builder: *compute.ProgramBuilder, value: compute.ValueId, shape: []const usize, axis: usize, amount: usize, before: bool) !compute.ValueId {
    var zero_shape = try allocator.dupe(usize, shape);
    defer allocator.free(zero_shape);
    zero_shape[axis] = amount;
    var spec = try compute.TensorSpec.init(allocator, .f32, zero_shape, null);
    defer spec.deinit();
    const bytes = try allocator.alloc(u8, elementCount(zero_shape) * 4);
    defer allocator.free(bytes);
    @memset(bytes, 0);
    const zero = try builder.addConstant(spec, .{ .bytes = bytes });
    const operands = if (before) [_]compute.ValueId{ zero, value } else [_]compute.ValueId{ value, zero };
    return (try builder.add(.cat, .{ .concat = .{ .axis = axis } }, &operands))[0];
}

fn lowerConv(scratch: std.mem.Allocator, builder: *compute.ProgramBuilder, values: *std.StringHashMap(compute.ValueId), shapes: *std.StringHashMap([]const usize), inputs: std.json.Array, attrs: std.json.ObjectMap, output_shape: []const usize) !compute.ValueId {
    const input_name = try string(inputs.items[0]);
    const weight_name = try string(inputs.items[1]);
    const x = values.get(input_name) orelse return error.MissingInput;
    const weight = values.get(weight_name) orelse return error.MissingInput;
    const input_shape = shapes.get(input_name).?;
    const weight_shape = shapes.get(weight_name).?;
    const kernel = try dimensions(scratch, attrs.get("kernel"));
    const strides = try dimensions(scratch, attrs.get("strides"));
    const dilations = try dimensions(scratch, attrs.get("dilations"));
    const group: usize = @intCast(try integer(attrs.get("group")));
    if (input_shape.len != 4 or weight_shape.len != 4 or output_shape.len != 4 or kernel.len != 2 or strides.len != 2 or dilations.len != 2) return error.UnsupportedShape;
    const n = input_shape[0];
    const channels = input_shape[1];
    const out_channels = output_shape[1];
    const oh = output_shape[2];
    const ow = output_shape[3];
    var result: ?compute.ValueId = null;
    if (group == channels and out_channels == channels and weight_shape[1] == 1) {
        for (0..kernel[0]) |kh| for (0..kernel[1]) |kw| {
            const ranges = [_]compute.SliceRange{
                .{ .start = 0, .stop = n },                                                                                           .{ .start = 0, .stop = channels },
                .{ .start = kh * dilations[0], .stop = kh * dilations[0] + (oh - 1) * strides[0] + 1, .step = @intCast(strides[0]) }, .{ .start = kw * dilations[1], .stop = kw * dilations[1] + (ow - 1) * strides[1] + 1, .step = @intCast(strides[1]) },
            };
            const patch = (try builder.add(.slice, .{ .slice = .{ .ranges = &ranges } }, &.{x}))[0];
            const wr = [_]compute.SliceRange{ .{ .start = 0, .stop = channels }, .{ .start = 0, .stop = 1 }, .{ .start = kh, .stop = kh + 1 }, .{ .start = kw, .stop = kw + 1 } };
            const coefficient = (try builder.add(.slice, .{ .slice = .{ .ranges = &wr } }, &.{weight}))[0];
            const coefficient_shape = [_]usize{ 1, channels, 1, 1 };
            const reshaped = (try builder.add(.reshape, .{ .reshape = .{ .shape = &coefficient_shape } }, &.{coefficient}))[0];
            const product = (try builder.add(.mul, .{ .none = {} }, &.{ patch, reshaped }))[0];
            result = if (result) |sum| (try builder.add(.add, .{ .none = {} }, &.{ sum, product }))[0] else product;
        };
    } else if (group == 1) {
        const positions = n * oh * ow;
        for (0..kernel[0]) |kh| for (0..kernel[1]) |kw| {
            const ranges = [_]compute.SliceRange{
                .{ .start = 0, .stop = n },                                                                                           .{ .start = 0, .stop = channels },
                .{ .start = kh * dilations[0], .stop = kh * dilations[0] + (oh - 1) * strides[0] + 1, .step = @intCast(strides[0]) }, .{ .start = kw * dilations[1], .stop = kw * dilations[1] + (ow - 1) * strides[1] + 1, .step = @intCast(strides[1]) },
            };
            const patch = (try builder.add(.slice, .{ .slice = .{ .ranges = &ranges } }, &.{x}))[0];
            const permutation = [_]usize{ 0, 2, 3, 1 };
            const ordered = (try builder.add(.permute, .{ .permute = .{ .axes = &permutation } }, &.{patch}))[0];
            const dense = (try builder.add(.contiguous, .{ .none = {} }, &.{ordered}))[0];
            const matrix_shape = [_]usize{ positions, channels };
            const matrix = (try builder.add(.reshape, .{ .reshape = .{ .shape = &matrix_shape } }, &.{dense}))[0];
            const wr = [_]compute.SliceRange{ .{ .start = 0, .stop = out_channels }, .{ .start = 0, .stop = channels }, .{ .start = kh, .stop = kh + 1 }, .{ .start = kw, .stop = kw + 1 } };
            const coefficient = (try builder.add(.slice, .{ .slice = .{ .ranges = &wr } }, &.{weight}))[0];
            const coefficient_shape = [_]usize{ out_channels, channels };
            const coefficient_matrix = (try builder.add(.reshape, .{ .reshape = .{ .shape = &coefficient_shape } }, &.{coefficient}))[0];
            const transposed = (try builder.add(.transpose, .{ .transpose = .{ .permutation = &.{ 1, 0 } } }, &.{coefficient_matrix}))[0];
            const product = (try builder.add(.matmul, .{ .none = {} }, &.{ matrix, transposed }))[0];
            result = if (result) |sum| (try builder.add(.add, .{ .none = {} }, &.{ sum, product }))[0] else product;
        };
        const nhwo = [_]usize{ n, oh, ow, out_channels };
        const shaped = (try builder.add(.reshape, .{ .reshape = .{ .shape = &nhwo } }, &.{result.?}))[0];
        const permutation = [_]usize{ 0, 3, 1, 2 };
        result = (try builder.add(.permute, .{ .permute = .{ .axes = &permutation } }, &.{shaped}))[0];
    } else return error.UnsupportedGroups;
    if (inputs.items.len == 3) {
        const bias = try inputId(values, inputs, 2);
        const bias_shape = [_]usize{ 1, out_channels, 1, 1 };
        const reshaped = (try builder.add(.reshape, .{ .reshape = .{ .shape = &bias_shape } }, &.{bias}))[0];
        result = (try builder.add(.add, .{ .none = {} }, &.{ result.?, reshaped }))[0];
    }
    return result.?;
}

fn lowerGemm(builder: *compute.ProgramBuilder, values: *std.StringHashMap(compute.ValueId), inputs: std.json.Array, attrs: std.json.ObjectMap) !compute.ValueId {
    if (try number(attrs.get("alpha")) != 1 or try number(attrs.get("beta")) != 1) return error.UnsupportedGemmScale;
    var right = try inputId(values, inputs, 1);
    if (try integer(attrs.get("transB")) == 1) right = (try builder.add(.transpose, .{ .transpose = .{ .permutation = &.{ 1, 0 } } }, &.{right}))[0];
    const product = (try builder.add(.matmul, .{ .none = {} }, &.{ try inputId(values, inputs, 0), right }))[0];
    return if (inputs.items.len == 3) (try builder.add(.add, .{ .none = {} }, &.{ product, try inputId(values, inputs, 2) }))[0] else product;
}

fn parseSafetensors(allocator: std.mem.Allocator, file: []const u8) !std.StringHashMap(TensorBytes) {
    if (file.len < 8) return error.InvalidSafetensors;
    const header_len = std.mem.readInt(u64, file[0..8], .little);
    const data_start = 8 + header_len;
    if (data_start > file.len) return error.InvalidSafetensors;
    var header = try std.json.parseFromSlice(std.json.Value, allocator, file[8..data_start], .{});
    defer header.deinit();
    const root = try object(header.value);
    var result = std.StringHashMap(TensorBytes).init(allocator);
    errdefer result.deinit();
    var iterator = root.iterator();
    while (iterator.next()) |entry| {
        if (std.mem.eql(u8, entry.key_ptr.*, "__metadata__")) continue;
        const metadata = try object(entry.value_ptr.*);
        if (!std.mem.eql(u8, try string(metadata.get("dtype")), "F32")) return error.UnsupportedDType;
        const shape_json = try array(metadata.get("shape"));
        const shape = try allocator.alloc(usize, shape_json.items.len);
        for (shape_json.items, shape) |dimension, *target| target.* = @intCast(try integer(dimension));
        const offsets = try array(metadata.get("data_offsets"));
        if (offsets.items.len != 2) return error.InvalidSafetensors;
        const start = data_start + @as(usize, @intCast(try integer(offsets.items[0])));
        const end = data_start + @as(usize, @intCast(try integer(offsets.items[1])));
        if (end > file.len or start > end) return error.InvalidSafetensors;
        try result.put(try allocator.dupe(u8, entry.key_ptr.*), .{ .shape = shape, .bytes = file[start..end] });
    }
    return result;
}

fn inputId(values: *std.StringHashMap(compute.ValueId), inputs: std.json.Array, index: usize) !compute.ValueId {
    return values.get(try string(inputs.items[index])) orelse error.MissingInput;
}
fn object(value: anytype) !std.json.ObjectMap {
    const item = if (@TypeOf(value) == ?std.json.Value) value orelse return error.InvalidManifest else value;
    return if (item == .object) item.object else error.InvalidManifest;
}
fn array(value: anytype) !std.json.Array {
    const item = if (@TypeOf(value) == ?std.json.Value) value orelse return error.InvalidManifest else value;
    return if (item == .array) item.array else error.InvalidManifest;
}
fn string(value: anytype) ![]const u8 {
    const item = if (@TypeOf(value) == ?std.json.Value) value orelse return error.InvalidManifest else value;
    return if (item == .string) item.string else error.InvalidManifest;
}
fn integer(value: anytype) !i64 {
    const item = if (@TypeOf(value) == ?std.json.Value) value orelse return error.InvalidManifest else value;
    return switch (item) {
        .integer => |v| v,
        .number_string => |v| std.fmt.parseInt(i64, v, 10),
        else => error.InvalidManifest,
    };
}
fn number(value: anytype) !f64 {
    const item = if (@TypeOf(value) == ?std.json.Value) value orelse return error.InvalidManifest else value;
    return switch (item) {
        .integer => |v| @floatFromInt(v),
        .float => |v| v,
        .number_string => |v| std.fmt.parseFloat(f64, v),
        else => error.InvalidManifest,
    };
}
fn dimensions(allocator: std.mem.Allocator, value: anytype) ![]const usize {
    const items = (try array(value)).items;
    const result = try allocator.alloc(usize, items.len);
    for (items, result) |item, *target| target.* = @intCast(try integer(item));
    return result;
}
fn compare(actual: []const f32, expected: []const f32) !Comparison {
    if (actual.len != expected.len) return error.ReferenceShapeMismatch;
    var mismatches: usize = 0;
    var max_absolute_error: f32 = 0;
    for (actual, expected) |got, want| {
        const absolute_error = @abs(got - want);
        max_absolute_error = @max(max_absolute_error, absolute_error);
        if (!std.math.isFinite(got) or absolute_error > 1e-4 + 1e-4 * @abs(want)) mismatches += 1;
    }
    return .{ .passed = mismatches == 0, .mismatches = mismatches, .max_absolute_error = max_absolute_error };
}
fn elementCount(shape: []const usize) usize {
    var count: usize = 1;
    for (shape) |dimension| count *= dimension;
    return count;
}
