const std = @import("std");
const compute = @import("compute");
const core = @import("compute_core_cyna.zig");

const no_value = std.math.maxInt(usize);

const DynamicInput = struct {
    dtype: compute.DType,
    dimensions: []const usize,
};

const ConstantInput = struct {
    bytes: []u8,
};

const Input = union(enum) {
    dynamic: DynamicInput,
    constant: ConstantInput,
};

const Output = struct {
    dtype: core.DenseDType,
    dimensions: []usize,
    byte_count: usize,

    fn deinit(self: *Output, allocator: std.mem.Allocator) void {
        allocator.free(self.dimensions);
        self.* = undefined;
    }

    fn spec(self: *const Output, allocator: std.mem.Allocator) compute.TensorSpec {
        return .{
            .allocator = allocator,
            .dtype_value = denseToComputeDType(self.dtype),
            .dimensions_value = self.dimensions,
            .axis_names_value = null,
        };
    }
};

/// A compiled Cyna graph that consumes the same ordered input tensors as a
/// Compute Program. The original Zig executable is retained by the caller as a
/// fallback, so this type only owns the Cyna graph and copied constant bytes.
pub const Executable = struct {
    allocator: std.mem.Allocator,
    handle: core.DenseGraphHandle,
    inputs: []Input,
    outputs: []Output,
    dynamic_byte_count: usize,
    constant_byte_count: usize,
    output_byte_count: usize,

    pub fn deinit(self: *Executable) void {
        self.handle.deinit();
        for (self.inputs) |*input| switch (input.*) {
            .dynamic => |*dynamic| self.allocator.free(dynamic.dimensions),
            .constant => |*constant| self.allocator.free(constant.bytes),
        };
        self.allocator.free(self.inputs);
        for (self.outputs) |*output| output.deinit(self.allocator);
        self.allocator.free(self.outputs);
        self.* = undefined;
    }

    pub fn run(
        self: *const Executable,
        zig_session: *compute.Session,
        dynamic_inputs: []const *compute.Tensor,
    ) ![]*compute.Tensor {
        if (dynamic_inputs.len > self.inputs.len) return error.InvalidShape;
        const total_bytes = std.math.add(usize, self.dynamic_byte_count, self.constant_byte_count) catch
            return error.InvalidShape;
        const input_bytes = try self.allocator.alloc(u8, total_bytes);
        defer self.allocator.free(input_bytes);
        var cursor: usize = 0;
        var dynamic_index: usize = 0;
        for (self.inputs) |input| switch (input) {
            .dynamic => |descriptor| {
                if (dynamic_index >= dynamic_inputs.len) return error.InvalidShape;
                const tensor = dynamic_inputs[dynamic_index];
                dynamic_index += 1;
                if (tensor.spec().dtype() != descriptor.dtype or
                    !std.mem.eql(usize, tensor.spec().dimensions(), descriptor.dimensions))
                    return error.InvalidShape;
                const expected = byteCount(descriptor.dimensions, descriptor.dtype) catch
                    return error.InvalidShape;
                copyTensorBytes(tensor, input_bytes[cursor..][0..expected]) catch
                    return error.InvalidShape;
                cursor += expected;
            },
            .constant => |constant| {
                @memcpy(input_bytes[cursor..][0..constant.bytes.len], constant.bytes);
                cursor += constant.bytes.len;
            },
        };
        if (dynamic_index != dynamic_inputs.len or cursor != input_bytes.len)
            return error.InvalidShape;

        const output_bytes = try self.allocator.alloc(u8, self.output_byte_count);
        defer self.allocator.free(output_bytes);
        try self.handle.run(input_bytes, output_bytes);

        const results = try self.allocator.alloc(*compute.Tensor, self.outputs.len);
        var created: usize = 0;
        errdefer {
            for (results[0..created]) |value| value.deinit();
            self.allocator.free(results);
        }
        cursor = 0;
        for (self.outputs, 0..) |output, index| {
            const bytes = output_bytes[cursor..][0..output.byte_count];
            results[index] = try zig_session.createTensor(output.spec(self.allocator), bytes);
            created += 1;
            cursor += output.byte_count;
        }
        return results;
    }
};

pub fn lower(
    allocator: std.mem.Allocator,
    program: *const compute.Program,
    goal: core.DenseGoal,
) !?Executable {
    // Identity Programs and duplicated binary operands still expose separate
    // Cyna resident-session failures. Keep them on the established Zig path
    // until those runtime contracts have their own parity evidence.
    if (program.instructions().len == 0) return null;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const temporary = arena.allocator();
    const mapping = try temporary.alloc(usize, program.values().len);
    @memset(mapping, no_value);
    const layout_sensitive = try temporary.alloc(bool, program.values().len);
    @memset(layout_sensitive, false);

    var inputs: std.ArrayList(Input) = .empty;
    errdefer inputs.deinit(allocator);
    var input_descriptors: std.ArrayList(core.DenseByteInput) = .empty;
    defer input_descriptors.deinit(temporary);
    var dynamic_byte_count: usize = 0;
    var constant_byte_count: usize = 0;

    for (program.inputs()) |input_id| {
        const index = input_id.int();
        if (index >= program.values().len) return error.InvalidProgram;
        const value = program.value(input_id) orelse return error.InvalidProgram;
        switch (value.kind()) {
            .input => {},
            else => return error.InvalidProgram,
        }
        const dimensions = try allocator.dupe(usize, value.spec().dimensions());
        const width = byteCount(value.spec().dimensions(), value.spec().dtype()) catch |err| {
            allocator.free(dimensions);
            return err;
        };
        const descriptor = DynamicInput{
            .dtype = value.spec().dtype(),
            .dimensions = dimensions,
        };
        const dummy = temporary.alloc(u8, width) catch |err| {
            allocator.free(dimensions);
            return err;
        };
        @memset(dummy, 0);
        mapping[index] = input_descriptors.items.len;
        inputs.append(allocator, .{ .dynamic = descriptor }) catch |err| {
            allocator.free(dimensions);
            return err;
        };
        try input_descriptors.append(temporary, .{
            .dtype = computeToDenseDType(descriptor.dtype),
            .shape = descriptor.dimensions,
            .bytes = dummy,
        });
        dynamic_byte_count = std.math.add(usize, dynamic_byte_count, width) catch
            return error.InvalidProgram;
    }

    for (program.values(), 0..) |value, index| {
        const constant = switch (value.kind()) {
            .constant => |constant| constant,
            else => continue,
        };
        const bytes = try constantBytes(temporary, value.spec(), constant);
        mapping[index] = input_descriptors.items.len;
        const owned_bytes = try allocator.dupe(u8, bytes);
        inputs.append(allocator, .{ .constant = .{ .bytes = owned_bytes } }) catch |err| {
            allocator.free(owned_bytes);
            return err;
        };
        try input_descriptors.append(temporary, .{
            .dtype = computeToDenseDType(value.spec().dtype()),
            .shape = value.spec().dimensions(),
            .bytes = bytes,
        });
        constant_byte_count = std.math.add(usize, constant_byte_count, bytes.len) catch
            return error.InvalidProgram;
    }

    var steps: std.ArrayList(core.DenseStep) = .empty;
    defer steps.deinit(temporary);
    var produced: usize = input_descriptors.items.len;
    for (program.instructions()) |*instruction| {
        const mapped = try mapInstruction(temporary, instruction, mapping, produced, &steps);
        const keeps_layout = switch (instruction.tag()) {
            .reshape, .transpose, .permute, .squeeze, .unsqueeze, .slice => true,
            else => false,
        };
        for (instruction.outputs(), mapped) |output_id, output_graph_id| {
            const index = output_id.int();
            if (index >= mapping.len or mapping[index] != no_value) return error.InvalidProgram;
            mapping[index] = output_graph_id;
            layout_sensitive[index] = keeps_layout;
        }
        produced = std.math.add(usize, produced, instruction.outputs().len) catch
            return error.InvalidProgram;
    }

    const selected = try temporary.alloc(usize, program.outputs().len);
    for (program.outputs(), selected) |output_id, *graph_id| {
        const index = output_id.int();
        if (index >= mapping.len or mapping[index] == no_value) return error.InvalidProgram;
        // These operations can expose views. Cyna currently materializes every
        // graph output as a dense tensor, so retain the established Zig path
        // when one of them is a public output.
        if (layout_sensitive[index]) return null;
        graph_id.* = mapping[index];
    }
    const inferred = core.inferDenseGraphMulti(
        temporary,
        input_descriptors.items,
        steps.items,
        selected,
    ) catch |err| switch (err) {
        error.OutOfMemory => return err,
        else => return null,
    };
    defer inferred.deinit(temporary);
    if (inferred.ranks.len != program.outputs().len) return error.InvalidProgram;

    const outputs = try allocator.alloc(Output, program.outputs().len);
    errdefer allocator.free(outputs);
    var output_count: usize = 0;
    errdefer for (outputs[0..output_count]) |*output| output.deinit(allocator);
    var output_byte_count: usize = 0;
    for (program.outputs(), 0..) |output_id, index| {
        const program_value = program.value(output_id) orelse return error.InvalidProgram;
        const dtype = inferred.dtype(index);
        const shape = inferred.shape(index);
        if (denseToComputeDType(dtype) != program_value.spec().dtype() or
            !std.mem.eql(usize, shape, program_value.spec().dimensions()))
            return error.InvalidProgram;
        const width = byteCount(shape, denseToComputeDType(dtype)) catch
            return error.InvalidProgram;
        outputs[index] = .{
            .dtype = dtype,
            .dimensions = try allocator.dupe(usize, shape),
            .byte_count = width,
        };
        output_count += 1;
        output_byte_count = std.math.add(usize, output_byte_count, width) catch
            return error.InvalidProgram;
    }

    var handle = core.createDenseGraphHandle(
        temporary,
        input_descriptors.items,
        steps.items,
        selected,
        goal,
    ) catch |err| switch (err) {
        error.OutOfMemory => return err,
        else => return null,
    };
    errdefer handle.deinit();
    return .{
        .allocator = allocator,
        .handle = handle,
        .inputs = try inputs.toOwnedSlice(allocator),
        .outputs = outputs,
        .dynamic_byte_count = dynamic_byte_count,
        .constant_byte_count = constant_byte_count,
        .output_byte_count = output_byte_count,
    };
}

fn mapInstruction(
    allocator: std.mem.Allocator,
    instruction: anytype,
    mapping: []const usize,
    produced: usize,
    steps: *std.ArrayList(core.DenseStep),
) ![]usize {
    const operands = try allocator.alloc(usize, instruction.inputs().len);
    for (instruction.inputs(), operands) |value_id, *graph_id| {
        const index = value_id.int();
        if (index >= mapping.len or mapping[index] == no_value) return error.InvalidProgram;
        graph_id.* = mapping[index];
    }
    const output_ids = try allocator.alloc(usize, instruction.outputs().len);
    for (output_ids, 0..) |*graph_id, index| graph_id.* = produced + index;

    const tag = instruction.tag();
    const options = instruction.options().*;
    var step: core.DenseStep = undefined;
    switch (tag) {
        .add, .sub, .mul, .div, .eq, .lt, .gt, .matmul, .dot => {
            if (operands.len != 2 or output_ids.len != 1) return error.InvalidProgram;
            if (operands[0] == operands[1]) return error.InvalidProgram;
            step = .{ .operation = binaryOperation(tag), .lhs = operands[0], .rhs = operands[1] };
        },
        .abs, .relu, .exp, .log, .sqrt, .sign, .sigmoid, .silu, .tanh, .erf, .gelu, .sin, .cos => {
            if (operands.len != 1 or output_ids.len != 1) return error.InvalidProgram;
            step = .{ .operation = unaryOperation(tag), .lhs = operands[0] };
        },
        .contiguous => {
            if (operands.len != 1 or output_ids.len != 1) return error.InvalidProgram;
            step = .{ .operation = .contiguous, .lhs = operands[0] };
        },
        .transpose => {
            if (operands.len != 1 or output_ids.len != 1) return error.InvalidProgram;
            switch (options) {
                .transpose => |value| {
                    if (value.permutation) |axes| {
                        step = .{ .operation = .permute, .lhs = operands[0], .metadata = axes };
                    } else {
                        step = .{ .operation = .transpose, .lhs = operands[0] };
                    }
                },
                else => return error.InvalidProgram,
            }
        },
        .reshape => {
            if (operands.len != 1 or output_ids.len != 1 or options != .reshape)
                return error.InvalidProgram;
            step = .{ .operation = .reshape, .lhs = operands[0], .metadata = options.reshape.shape };
        },
        .permute => {
            if (operands.len != 1 or output_ids.len != 1 or options != .permute)
                return error.InvalidProgram;
            step = .{ .operation = .permute, .lhs = operands[0], .metadata = options.permute.axes };
        },
        .squeeze => {
            if (operands.len != 1 or output_ids.len != 1 or options != .squeeze)
                return error.InvalidProgram;
            step = .{
                .operation = .squeeze,
                .lhs = operands[0],
                .rhs = options.squeeze.axis orelse std.math.maxInt(usize),
            };
        },
        .unsqueeze => {
            if (operands.len != 1 or output_ids.len != 1 or options != .unsqueeze)
                return error.InvalidProgram;
            step = .{ .operation = .unsqueeze, .lhs = operands[0], .rhs = options.unsqueeze.axis };
        },
        .cast => {
            if (operands.len != 1 or output_ids.len != 1 or options != .cast)
                return error.InvalidProgram;
            step = .{
                .operation = .cast,
                .lhs = operands[0],
                .rhs = @intCast(@backingInt(computeToDenseDType(options.cast.to))),
            };
        },
        .where => {
            if (operands.len != 3 or output_ids.len != 1) return error.InvalidProgram;
            step = .{ .operation = .where, .lhs = operands[0], .rhs = operands[1], .third = operands[2] };
        },
        .masked_fill => {
            if (operands.len != 2 or output_ids.len != 1 or options != .masked_fill)
                return error.InvalidProgram;
            const scalar = try allocator.dupe(usize, &core.scalarMetadata(options.masked_fill.value));
            step = .{ .operation = .masked_fill, .lhs = operands[0], .rhs = operands[1], .metadata = scalar };
        },
        .clamp => {
            if (operands.len != 1 or output_ids.len != 1 or options != .clamp)
                return error.InvalidProgram;
            const scalar = try allocator.alloc(usize, 4);
            @memcpy(scalar[0..2], &core.scalarMetadata(options.clamp.min));
            @memcpy(scalar[2..4], &core.scalarMetadata(options.clamp.max));
            step = .{ .operation = .clamp, .lhs = operands[0], .metadata = scalar };
        },
        .softmax, .log_softmax => {
            if (operands.len != 1 or output_ids.len != 1) return error.InvalidProgram;
            const axis = if (tag == .softmax) options.softmax.axis else options.log_softmax.axis;
            step = .{
                .operation = if (tag == .softmax) .softmax else .log_softmax,
                .lhs = operands[0],
                .rhs = axis,
            };
        },
        .sum_all, .mean_all, .min_all, .max_all, .variance_all, .std_all, .argmin_all, .argmax_all => {
            if (operands.len != 1 or output_ids.len != 1 or options != .reduce_all or
                options.reduce_all.keepdim)
                return error.InvalidProgram;
            step = .{ .operation = reductionOperation(tag), .lhs = operands[0] };
        },
        .sum_axis, .mean_axis, .min_axis, .max_axis, .variance_axis, .std_axis, .argmin_axis, .argmax_axis => {
            if (operands.len != 1 or output_ids.len != 1 or options != .reduce_axis or
                options.reduce_axis.keepdim)
                return error.InvalidProgram;
            step = .{
                .operation = reductionOperation(tag),
                .lhs = operands[0],
                .rhs = options.reduce_axis.axis,
            };
        },
        .cat, .stack => {
            if (operands.len < 2 or output_ids.len != 1) return error.InvalidProgram;
            const axis = if (tag == .cat) options.concat.axis else options.stack.axis;
            step = .{
                .operation = if (tag == .cat) .cat else .stack,
                .lhs = operands[0],
                .rhs = axis,
                .metadata = operands[1..],
            };
        },
        .layer_norm, .rms_norm => {
            if (operands.len != 1 or output_ids.len != 1) return error.InvalidProgram;
            const axis: usize, const epsilon: f64 = if (tag == .layer_norm)
                .{ options.layer_norm.axis, options.layer_norm.eps }
            else
                .{ options.rms_norm.axis, options.rms_norm.eps };
            const scalar = try allocator.dupe(usize, &core.scalarMetadata(epsilon));
            step = .{
                .operation = if (tag == .layer_norm) .layer_norm else .rms_norm,
                .lhs = operands[0],
                .rhs = axis,
                .metadata = scalar,
            };
        },
        .one_hot => {
            if (operands.len != 1 or output_ids.len != 1 or options != .one_hot)
                return error.InvalidProgram;
            step = .{
                .operation = .one_hot,
                .lhs = operands[0],
                .rhs = options.one_hot.num_classes,
            };
        },
        .topk => {
            if (operands.len != 1 or output_ids.len != 2 or options != .topk)
                return error.InvalidProgram;
            const metadata = try allocator.alloc(usize, 4);
            metadata[0] = options.topk.axis;
            metadata[1] = options.topk.k;
            metadata[2] = @intFromBool(options.topk.largest);
            metadata[3] = @intFromBool(options.topk.sorted);
            try steps.append(allocator, .{
                .operation = .top_k_values,
                .lhs = operands[0],
                .metadata = metadata,
            });
            try steps.append(allocator, .{
                .operation = .top_k_indices,
                .lhs = operands[0],
                .metadata = metadata,
            });
            return output_ids;
        },
        .slice => {
            if (operands.len != 1 or output_ids.len != 1 or options != .slice)
                return error.InvalidProgram;
            const values = try allocator.alloc(usize, try std.math.mul(usize, options.slice.ranges.len, 3));
            for (options.slice.ranges, 0..) |range, index| {
                if (range.step <= 0) return error.InvalidProgram;
                values[index * 3] = range.start;
                values[index * 3 + 1] = range.stop;
                values[index * 3 + 2] = @intCast(range.step);
            }
            step = .{ .operation = .slice, .lhs = operands[0], .metadata = values };
        },
        else => return error.InvalidProgram,
    }
    try steps.append(allocator, step);
    return output_ids;
}

fn binaryOperation(tag: compute.OpTag) core.DenseOperation {
    return switch (tag) {
        .add => .add,
        .sub => .sub,
        .mul => .mul,
        .div => .div,
        .eq => .eq,
        .lt => .lt,
        .gt => .gt,
        .matmul => .matmul,
        .dot => .dot,
        else => unreachable,
    };
}

fn unaryOperation(tag: compute.OpTag) core.DenseOperation {
    return switch (tag) {
        .abs => .abs,
        .relu => .relu,
        .exp => .exp,
        .log => .log,
        .sqrt => .sqrt,
        .sign => .sign,
        .sigmoid => .sigmoid,
        .silu => .silu,
        .tanh => .tanh,
        .erf => .erf,
        .gelu => .gelu,
        .sin => .sin,
        .cos => .cos,
        else => unreachable,
    };
}

fn reductionOperation(tag: compute.OpTag) core.DenseOperation {
    return switch (tag) {
        .sum_all, .sum_axis => .sum_all,
        .mean_all, .mean_axis => .mean_all,
        .min_all, .min_axis => .min_all,
        .max_all, .max_axis => .max_all,
        .argmin_all, .argmin_axis => .argmin_all,
        .argmax_all, .argmax_axis => .argmax_all,
        .variance_all, .variance_axis => .variance_all,
        .std_all, .std_axis => .std_all,
        else => unreachable,
    };
}

fn constantBytes(
    allocator: std.mem.Allocator,
    spec: *const compute.TensorSpec,
    constant: compute.Constant,
) ![]u8 {
    const count = try elementCount(spec.dimensions());
    const byte_count = try std.math.mul(usize, count, spec.dtype().size());
    return switch (constant) {
        .bytes => |bytes| if (bytes.len == byte_count)
            allocator.dupe(u8, bytes)
        else
            error.InvalidProgram,
        .scalar => |scalar| blk: {
            if (count != 1) return error.InvalidProgram;
            const bytes = try allocator.alloc(u8, byte_count);
            switch (scalar) {
                .bf16 => |value| @memcpy(bytes, std.mem.asBytes(&value)),
                .f32 => |value| @memcpy(bytes, std.mem.asBytes(&value)),
                .f64 => |value| @memcpy(bytes, std.mem.asBytes(&value)),
                .u32 => |value| @memcpy(bytes, std.mem.asBytes(&value)),
                .i64 => |value| @memcpy(bytes, std.mem.asBytes(&value)),
            }
            break :blk bytes;
        },
    };
}

fn copyTensorBytes(tensor: *const compute.Tensor, output: []u8) !void {
    const spec = tensor.spec();
    const width = spec.dtype().size();
    const count = try elementCount(spec.dimensions());
    const expected = try std.math.mul(usize, count, width);
    if (output.len != expected) return error.InvalidShape;
    const source = tensor.bytes();
    if (source.len < expected + tensor.offsetBytes()) return error.InvalidShape;
    var dense_stride: isize = 1;
    var dense = true;
    var reverse = spec.dimensions().len;
    while (reverse > 0) {
        reverse -= 1;
        if (tensor.strides()[reverse] != dense_stride) dense = false;
        dense_stride *= @intCast(spec.dimensions()[reverse]);
    }
    if (dense and tensor.offsetBytes() == 0) {
        @memcpy(output, source[0..expected]);
        return;
    }
    for (0..count) |linear| {
        var remainder = linear;
        var element = tensor.offsetBytes() / width;
        for (0..spec.dimensions().len) |offset| {
            const axis = spec.dimensions().len - 1 - offset;
            const coordinate = remainder % spec.dimensions()[axis];
            remainder /= spec.dimensions()[axis];
            element = @intCast(@as(isize, @intCast(element)) +
                @as(isize, @intCast(coordinate)) * tensor.strides()[axis]);
        }
        const byte_offset = std.math.mul(usize, element, width) catch return error.InvalidShape;
        if (byte_offset > source.len or width > source.len - byte_offset)
            return error.InvalidShape;
        @memcpy(output[linear * width ..][0..width], source[byte_offset..][0..width]);
    }
}

fn elementCount(dimensions: []const usize) !usize {
    var count: usize = 1;
    for (dimensions) |dimension| count = std.math.mul(usize, count, dimension) catch
        return error.InvalidProgram;
    return count;
}

fn byteCount(dimensions: []const usize, dtype: compute.DType) !usize {
    return std.math.mul(usize, try elementCount(dimensions), dtype.size());
}

fn computeToDenseDType(dtype: compute.DType) core.DenseDType {
    return switch (dtype) {
        .bf16 => .bf16,
        .f32 => .f32,
        .f64 => .f64,
        .u32 => .u32,
        .i64 => .i64,
    };
}

fn denseToComputeDType(dtype: core.DenseDType) compute.DType {
    return switch (dtype) {
        .bf16 => .bf16,
        .f32 => .f32,
        .f64 => .f64,
        .u32 => .u32,
        .i64 => .i64,
    };
}
