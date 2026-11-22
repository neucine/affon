const std = @import("std");
const compute = @import("compute");
const core = @import("compute_core_cyna.zig");

const no_value = std.math.maxInt(usize);

pub const LoweringDiagnostic = struct {
    category: []const u8 = "lowering",
    reason: []const u8 = "unsupported",
    operation: ?compute.OpTag = null,
    option: ?[]const u8 = null,
};

const DynamicInput = struct {
    dtype: compute.DType,
    dimensions: []const usize,
};

const Input = union(enum) {
    dynamic: DynamicInput,
    constant: void,
};

const CachedInput = struct {
    bytes: []u8,
    token: core.DenseGraphHandle.TensorToken,
    tensor_id: u64,
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

pub fn profileNow() u64 {
    var timestamp: std.c.timespec = undefined;
    if (std.c.clock_gettime(.MONOTONIC, &timestamp) != 0) return 0;
    return @as(u64, @intCast(timestamp.sec)) * std.time.ns_per_s +
        @as(u64, @intCast(timestamp.nsec));
}

/// A compiled Cyna graph that consumes the same ordered input tensors as a
/// Compute Program. The original Zig executable is retained by the caller as a
/// fallback, so this type only owns the Cyna graph and copied constant bytes.
pub const Executable = struct {
    allocator: std.mem.Allocator,
    handle: core.DenseGraphHandle,
    source_instruction_by_step: []usize,
    inputs: []Input,
    cached_dynamic: []?CachedInput,
    outputs: []Output,
    dense_inputs: bool,
    dynamic_byte_count: usize,
    constant_byte_count: usize,
    output_byte_count: usize,

    pub fn deinit(self: *Executable) void {
        for (self.cached_dynamic) |cached| {
            if (cached) |entry| {
                self.handle.releaseTensor(entry.token) catch {};
                self.allocator.free(entry.bytes);
            }
        }
        self.allocator.free(self.cached_dynamic);
        self.handle.deinit();
        self.allocator.free(self.source_instruction_by_step);
        for (self.inputs) |*input| switch (input.*) {
            .dynamic => |*dynamic| self.allocator.free(dynamic.dimensions),
            .constant => {},
        };
        self.allocator.free(self.inputs);
        for (self.outputs) |*output| output.deinit(self.allocator);
        self.allocator.free(self.outputs);
        self.* = undefined;
    }

    pub fn run(
        self: *Executable,
        zig_session: *compute.Session,
        dynamic_inputs: []const *compute.Tensor,
    ) ![]*compute.Tensor {
        const profiling = std.c.getenv("AFFON_CYNA_PROFILE") != null;
        const start_ns = if (profiling) profileNow() else 0;
        var pack_ns: u64 = 0;
        var create_ns: u64 = 0;
        var graph_ns: u64 = 0;
        var output_ns: u64 = 0;
        if (dynamic_inputs.len > self.inputs.len) return error.InvalidShape;
        const input_bytes = if (self.dense_inputs) null else try self.allocator.alloc(u8, self.dynamic_byte_count);
        defer if (input_bytes) |bytes| self.allocator.free(bytes);
        const pack_start = if (profiling) profileNow() else 0;
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
                if (self.dense_inputs and !denseTensor(tensor)) return error.GuardMismatch;
                const expected = byteCount(descriptor.dimensions, descriptor.dtype) catch
                    return error.InvalidShape;
                if (input_bytes) |buffer| {
                    copyTensorBytes(tensor, buffer[cursor..][0..expected]) catch
                        return error.InvalidShape;
                } else if (tensor.bytes().len < expected) return error.InvalidShape;
                cursor += expected;
            },
            .constant => {},
        };
        if (dynamic_index != dynamic_inputs.len or cursor != self.dynamic_byte_count)
            return error.InvalidShape;

        if (profiling) pack_ns = profileNow() - pack_start;
        const create_start = if (profiling) profileNow() else 0;
        var input_tokens_buffer: [8]core.DenseGraphHandle.TensorToken = undefined;
        const input_tokens = if (dynamic_inputs.len <= input_tokens_buffer.len)
            input_tokens_buffer[0..dynamic_inputs.len]
        else
            try self.allocator.alloc(core.DenseGraphHandle.TensorToken, dynamic_inputs.len);
        defer if (dynamic_inputs.len > input_tokens_buffer.len)
            self.allocator.free(input_tokens);
        cursor = 0;
        dynamic_index = 0;
        for (self.inputs, 0..) |input, index| switch (input) {
            .dynamic => |descriptor| {
                const byte_count = byteCount(descriptor.dimensions, descriptor.dtype) catch
                    return error.InvalidShape;
                const bytes = if (input_bytes) |buffer|
                    buffer[cursor..][0..byte_count]
                else
                    dynamic_inputs[dynamic_index].bytes()[0..byte_count];
                const tensor_id = dynamic_inputs[dynamic_index].id().local;
                dynamic_index += 1;
                if (self.cached_dynamic[index]) |cached| {
                    if (cached.tensor_id == tensor_id or
                        std.mem.eql(u8, cached.bytes, bytes))
                    {
                        input_tokens[index] = cached.token;
                        cursor += byte_count;
                        continue;
                    }
                }
                const owned_bytes = try self.allocator.dupe(u8, bytes);
                const token = self.handle.createInputTensor(index, bytes) catch |err| {
                    self.allocator.free(owned_bytes);
                    return err;
                };
                if (self.cached_dynamic[index]) |cached| {
                    self.handle.releaseTensor(cached.token) catch {};
                    self.allocator.free(cached.bytes);
                }
                self.cached_dynamic[index] = .{
                    .bytes = owned_bytes,
                    .token = token,
                    .tensor_id = tensor_id,
                };
                input_tokens[index] = token;
                cursor += byte_count;
            },
            .constant => {},
        };

        if (profiling) create_ns = profileNow() - create_start;
        const graph_start = if (profiling) profileNow() else 0;
        var output_tokens_buffer: [8]core.DenseGraphHandle.TensorToken = undefined;
        const output_tokens = if (self.outputs.len <= output_tokens_buffer.len)
            output_tokens_buffer[0..self.outputs.len]
        else
            try self.allocator.alloc(core.DenseGraphHandle.TensorToken, self.outputs.len);
        defer if (self.outputs.len > output_tokens_buffer.len)
            self.allocator.free(output_tokens);
        try self.handle.runTensors(self.allocator, input_tokens, output_tokens);
        if (profiling) graph_ns = profileNow() - graph_start;
        const output_start = if (profiling) profileNow() else 0;
        // A successful graph call publishes every output token. Release them
        // all even when converting an earlier output to an Affon Tensor fails.
        var output_tokens_released = false;
        defer if (!output_tokens_released) for (output_tokens) |token| {
            self.handle.releaseTensor(token) catch {};
        };

        const results = try self.allocator.alloc(*compute.Tensor, self.outputs.len);
        var created: usize = 0;
        errdefer {
            for (results[0..created]) |value| value.deinit();
            self.allocator.free(results);
        }
        if (self.outputs.len <= 8) {
            var layout_words_buffer: [8 * core.DenseGraphHandle.tensor_layout_words_per_output]usize = undefined;
            const layout_words = layout_words_buffer[0 .. self.outputs.len * core.DenseGraphHandle.tensor_layout_words_per_output];
            try self.handle.describeTensors(output_tokens, layout_words);

            var all_dense = true;
            for (self.outputs, 0..) |output, index| {
                const words = layout_words[index * core.DenseGraphHandle.tensor_layout_words_per_output ..][0..core.DenseGraphHandle.tensor_layout_words_per_output];
                const rank = words[0];
                const backing_bytes = words[1];
                const offset_bytes = words[2];
                const width = output.spec(self.allocator).dtype().size();
                if (rank != output.dimensions.len or backing_bytes == 0 or
                    backing_bytes % width != 0 or offset_bytes % width != 0 or
                    offset_bytes != 0 or backing_bytes != output.byte_count)
                {
                    all_dense = false;
                    break;
                }
                var expected_stride: usize = 1;
                var axis = output.dimensions.len;
                while (axis > 0) {
                    axis -= 1;
                    if (words[3 + axis] != expected_stride) {
                        all_dense = false;
                        break;
                    }
                    expected_stride = std.math.mul(usize, expected_stride, output.dimensions[axis]) catch {
                        all_dense = false;
                        break;
                    };
                }
                if (!all_dense) break;
            }
            if (all_dense) {
                var destinations_buffer: [8]usize = undefined;
                var byte_counts_buffer: [8]usize = undefined;
                const destinations = destinations_buffer[0..self.outputs.len];
                const byte_counts = byte_counts_buffer[0..self.outputs.len];
                for (self.outputs, 0..) |output, index| {
                    const tensor = try zig_session.createUninitializedOutput(output.spec(self.allocator));
                    results[index] = tensor;
                    created += 1;
                    destinations[index] = @intFromPtr(tensor.writableBytes().ptr);
                    byte_counts[index] = output.byte_count;
                }
                output_tokens_released = true;
                try self.handle.readTensorsAndRelease(output_tokens, destinations, byte_counts);
                if (profiling) {
                    output_ns = profileNow() - output_start;
                    std.debug.print("AFFON_CYNA_PROFILE pack_us={d} create_us={d} graph_us={d} output_us={d} total_us={d} hits={d} misses={d}\n", .{
                        pack_ns / std.time.ns_per_us,                   create_ns / std.time.ns_per_us,
                        graph_ns / std.time.ns_per_us,                  output_ns / std.time.ns_per_us,
                        (profileNow() - start_ns) / std.time.ns_per_us, self.handle.cacheHits(),
                        self.handle.cacheMisses(),
                    });
                }
                return results;
            }
        }
        for (self.outputs, 0..) |output, index| {
            var stride_words_buffer: [8]usize = undefined;
            const stride_words = if (output.dimensions.len <= stride_words_buffer.len)
                stride_words_buffer[0..output.dimensions.len]
            else
                try self.allocator.alloc(usize, output.dimensions.len);
            defer if (output.dimensions.len > stride_words_buffer.len)
                self.allocator.free(stride_words);
            const layout = try self.handle.tensorLayout(output_tokens[index], stride_words);
            if (layout.rank != output.dimensions.len) return error.InvalidShape;

            const width = output.spec(self.allocator).dtype().size();
            if (layout.backing_bytes == 0 or layout.backing_bytes % width != 0 or
                layout.offset_bytes % width != 0)
                return error.InvalidShape;
            const backing_dimensions = [_]usize{layout.backing_bytes / width};
            var dense_output = layout.offset_bytes == 0 and layout.backing_bytes == output.byte_count;
            var expected_stride: usize = 1;
            var axis = output.dimensions.len;
            while (axis > 0) {
                axis -= 1;
                if (stride_words[axis] != expected_stride) dense_output = false;
                expected_stride = std.math.mul(usize, expected_stride, output.dimensions[axis]) catch
                    return error.InvalidShape;
            }
            if (dense_output) {
                const tensor = try zig_session.createUninitializedOutput(output.spec(self.allocator));
                self.handle.readTensorBacking(output_tokens[index], tensor.writableBytes()) catch |err| {
                    tensor.deinit();
                    return err;
                };
                results[index] = tensor;
                created += 1;
                continue;
            }
            var backing_spec = try compute.TensorSpec.init(
                self.allocator,
                output.spec(self.allocator).dtype(),
                &backing_dimensions,
                null,
            );
            defer backing_spec.deinit();
            const backing = try zig_session.createUninitializedOutput(backing_spec);
            defer backing.deinit();
            try self.handle.readTensorBacking(output_tokens[index], backing.writableBytes());

            const layout_strides = try self.allocator.alloc(isize, output.dimensions.len);
            defer self.allocator.free(layout_strides);
            for (stride_words, layout_strides) |stride, *destination|
                destination.* = std.math.cast(isize, stride) orelse return error.InvalidShape;
            results[index] = try backing.createLayoutView(
                output.spec(self.allocator),
                layout_strides,
                layout.offset_bytes,
            );
            created += 1;
        }
        if (profiling) {
            output_ns = profileNow() - output_start;
            std.debug.print("AFFON_CYNA_PROFILE pack_us={d} create_us={d} graph_us={d} output_us={d} total_us={d} hits={d} misses={d}\n", .{
                pack_ns / std.time.ns_per_us,                   create_ns / std.time.ns_per_us,
                graph_ns / std.time.ns_per_us,                  output_ns / std.time.ns_per_us,
                (profileNow() - start_ns) / std.time.ns_per_us, self.handle.cacheHits(),
                self.handle.cacheMisses(),
            });
        }
        return results;
    }
};

pub fn lower(
    allocator: std.mem.Allocator,
    program: *const compute.Program,
    goal: core.DenseGoal,
    dense_inputs: bool,
    residual_policy: compute.ResidualPolicy,
    search_limits: compute.SearchLimits,
    diagnostic: *LoweringDiagnostic,
) !?Executable {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const temporary = arena.allocator();
    const mapping = try temporary.alloc(usize, program.values().len);
    @memset(mapping, no_value);

    var inputs: std.ArrayList(Input) = .empty;
    errdefer {
        for (inputs.items) |input| switch (input) {
            .dynamic => |dynamic| allocator.free(dynamic.dimensions),
            .constant => {},
        };
        inputs.deinit(allocator);
    }
    var input_descriptors: std.ArrayList(core.DenseByteInput) = .empty;
    defer input_descriptors.deinit(temporary);
    var dynamic_byte_count: usize = 0;
    var constant_byte_count: usize = 0;
    var constant_data: std.ArrayList(u8) = .empty;
    defer constant_data.deinit(temporary);

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
        try constant_data.appendSlice(temporary, bytes);
        mapping[index] = input_descriptors.items.len;
        try inputs.append(allocator, .{ .constant = {} });
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
    var source_instruction_by_step: std.ArrayList(usize) = .empty;
    defer source_instruction_by_step.deinit(temporary);
    var produced: usize = input_descriptors.items.len;
    const instruction_mapping = try temporary.alloc(usize, program.instructions().len);
    @memset(instruction_mapping, no_value);
    for (program.instructions(), 0..) |*instruction, instruction_index| {
        const first_step = steps.items.len;
        const mapped = try mapInstruction(
            temporary,
            instruction,
            mapping,
            produced,
            &steps,
            diagnostic,
        );
        for (first_step..steps.items.len) |_| try source_instruction_by_step.append(temporary, instruction_index);
        for (instruction.outputs(), mapped) |output_id, output_graph_id| {
            const index = output_id.int();
            if (index >= mapping.len or mapping[index] != no_value) return error.InvalidProgram;
            mapping[index] = output_graph_id;
        }
        if (steps.items.len == first_step + 1)
            instruction_mapping[instruction_index] = first_step;
        produced = std.math.add(usize, input_descriptors.items.len, steps.items.len) catch
            return error.InvalidProgram;
    }

    const requirements = try temporary.alloc(core.DenseResidualRequirement, program.residualRequirements().len);
    for (program.residualRequirements(), requirements) |requirement, *mapped| {
        const value = requirement.value().int();
        if (value >= mapping.len or mapping[value] == no_value) return error.InvalidProgram;
        var release_after: ?usize = null;
        var recompute_before: ?usize = null;
        if (requirement.releaseAfter()) |anchor| {
            const index = anchor.int();
            if (index >= instruction_mapping.len or instruction_mapping[index] == no_value) return error.InvalidProgram;
            release_after = instruction_mapping[index];
        }
        if (requirement.recomputeBefore()) |anchor| {
            const index = anchor.int();
            if (index >= instruction_mapping.len or instruction_mapping[index] == no_value) return error.InvalidProgram;
            recompute_before = instruction_mapping[index];
        }
        mapped.* = .{
            .value = mapping[value],
            .recomputable_view = requirement.eligibility() == .recomputable_view,
            .release_after = release_after,
            .recompute_before = recompute_before,
        };
    }
    var residual_use_count: usize = 0;
    for (program.residualNeeds()) |need| residual_use_count += need.uses.len;
    const residual_uses = try temporary.alloc(core.DenseResidualUse, residual_use_count);
    var residual_use_index: usize = 0;
    for (program.residualNeeds()) |need| {
        const value = need.value.int();
        if (value >= mapping.len or mapping[value] == no_value) return error.InvalidProgram;
        for (need.uses) |use| {
            const index = use.instruction.int();
            if (index >= instruction_mapping.len or instruction_mapping[index] == no_value) return error.InvalidProgram;
            residual_uses[residual_use_index] = .{
                .value = mapping[value],
                .instruction = instruction_mapping[index],
                .input_index = use.input_index,
            };
            residual_use_index += 1;
        }
    }

    const selected = try temporary.alloc(usize, program.outputs().len);
    for (program.outputs(), selected) |output_id, *graph_id| {
        const index = output_id.int();
        if (index >= mapping.len or mapping[index] == no_value) return error.InvalidProgram;
        graph_id.* = mapping[index];
    }
    const inferred = core.inferDenseGraphMulti(
        temporary,
        input_descriptors.items,
        steps.items,
        selected,
    ) catch |err| switch (err) {
        error.OutOfMemory => return err,
        else => {
            diagnostic.* = .{
                .category = "inference",
                .reason = @errorName(err),
            };
            return null;
        },
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

    var handle = core.createDenseGraphHandleWithConstantsAndSearch(
        temporary,
        input_descriptors.items,
        program.inputs().len,
        constant_data.items,
        steps.items,
        selected,
        goal,
        dense_inputs,
        requirements,
        residual_uses,
        switch (residual_policy) {
            .retain => .retain,
            .automatic => .automatic,
            .recompute => .recompute,
        },
        .{
            .structural_states = search_limits.structural_states,
            .structural_expansions = search_limits.structural_expansions,
            .complete_evaluations = search_limits.complete_evaluations,
            .physical_work = search_limits.physical_work,
            .ready_work = search_limits.ready_work,
            .max_peak_active_bytes = search_limits.resources.max_peak_active_bytes,
            .max_invocation_scratch_bytes = search_limits.resources.max_invocation_scratch_bytes,
        },
    ) catch |err| switch (err) {
        error.OutOfMemory => return err,
        else => {
            diagnostic.* = .{
                .category = "compile",
                .reason = @errorName(err),
            };
            return null;
        },
    };
    errdefer handle.deinit();
    const owned_inputs = try inputs.toOwnedSlice(allocator);
    errdefer {
        for (owned_inputs) |input| switch (input) {
            .dynamic => |dynamic| allocator.free(dynamic.dimensions),
            .constant => {},
        };
        allocator.free(owned_inputs);
    }
    const cached_dynamic = try allocator.alloc(?CachedInput, owned_inputs.len);
    errdefer allocator.free(cached_dynamic);
    const owned_step_origins = try allocator.dupe(usize, source_instruction_by_step.items);
    @memset(cached_dynamic, null);
    return .{
        .allocator = allocator,
        .handle = handle,
        .source_instruction_by_step = owned_step_origins,
        .inputs = owned_inputs,
        .cached_dynamic = cached_dynamic,
        .outputs = outputs,
        .dense_inputs = dense_inputs,
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
    diagnostic: *LoweringDiagnostic,
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
    diagnostic.* = .{
        .category = "operation",
        .reason = "unsupported",
        .operation = tag,
        .option = @tagName(std.meta.activeTag(options)),
    };
    var step: core.DenseStep = undefined;
    switch (tag) {
        .add, .sub, .mul, .div, .eq, .lt, .gt, .matmul, .dot => {
            if (operands.len != 2 or output_ids.len != 1) return error.InvalidProgram;
            step = .{ .operation = binaryOperation(tag), .lhs = operands[0], .rhs = operands[1] };
        },
        .quantized_matmul => {
            if (operands.len != 4 or output_ids.len != 1 or options != .quantized_matmul)
                return error.InvalidProgram;
            const metadata = try allocator.alloc(usize, 3);
            metadata[0] = operands[3];
            metadata[1] = options.quantized_matmul.bits;
            metadata[2] = options.quantized_matmul.group_size;
            step = .{
                .operation = .quantized_matmul,
                .lhs = operands[0],
                .rhs = operands[1],
                .third = operands[2],
                .metadata = metadata,
            };
        },
        .abs, .neg, .relu, .exp, .log, .sqrt, .sign, .sigmoid, .silu, .tanh, .erf, .gelu, .sin, .cos => {
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
        .reduce_to_shape => {
            if (operands.len != 1 or output_ids.len != 1 or options != .reduce_to_shape)
                return error.InvalidProgram;
            step = .{
                .operation = .reduce_to_shape,
                .lhs = operands[0],
                .metadata = options.reduce_to_shape.shape,
            };
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
            if (operands.len != 1 or output_ids.len != 1 or options != .reduce_all)
                return error.InvalidProgram;
            try steps.append(allocator, .{
                .operation = reductionOperation(tag),
                .lhs = operands[0],
                .metadata = if (options.reduce_all.keepdim) &.{1} else &.{},
            });
            output_ids[0] = produced;
            return output_ids;
        },
        .sum_axis, .mean_axis, .min_axis, .max_axis, .variance_axis, .std_axis, .argmin_axis, .argmax_axis => {
            if (operands.len != 1 or output_ids.len != 1 or options != .reduce_axis)
                return error.InvalidProgram;
            try steps.append(allocator, .{
                .operation = reductionOperation(tag),
                .lhs = operands[0],
                .rhs = options.reduce_axis.axis,
                .metadata = if (options.reduce_axis.keepdim) &.{1} else &.{},
            });
            output_ids[0] = produced;
            return output_ids;
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
        .cross_entropy_indexed, .cross_entropy, .log_softmax_nll => {
            if (operands.len != 2 or output_ids.len != 1) return error.InvalidProgram;
            const axis = switch (options) {
                .cross_entropy_indexed => |value| value.axis,
                .cross_entropy => |value| value.axis,
                .log_softmax_nll => |value| value.axis,
                else => return error.InvalidProgram,
            };
            step = .{
                .operation = switch (tag) {
                    .cross_entropy_indexed => .cross_entropy_indexed,
                    .cross_entropy => .cross_entropy,
                    .log_softmax_nll => .log_softmax_nll,
                    else => unreachable,
                },
                .lhs = operands[0],
                .rhs = operands[1],
                .third = axis,
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
        .index_select, .gather => {
            if (operands.len != 2 or output_ids.len != 1) return error.InvalidProgram;
            const axis = if (tag == .index_select) options.index_select.axis else options.gather.axis;
            step = .{
                .operation = if (tag == .index_select) .index_select else .gather,
                .lhs = operands[0],
                .rhs = operands[1],
                .third = axis,
            };
        },
        .embedding => {
            if (operands.len != 2 or output_ids.len != 1 or options != .embedding)
                return error.InvalidProgram;
            step = .{ .operation = .embedding, .lhs = operands[0], .rhs = operands[1] };
        },
        .scatter_add => {
            if (operands.len != 3 or output_ids.len != 1 or options != .scatter_add)
                return error.InvalidProgram;
            const axis = try allocator.dupe(usize, &.{options.scatter_add.axis});
            step = .{
                .operation = .scatter_add,
                .lhs = operands[0],
                .rhs = operands[1],
                .third = operands[2],
                .metadata = axis,
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
        .neg => .neg,
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
        .sum_all => .sum_all,
        .sum_axis => .sum_axis,
        .mean_all => .mean_all,
        .mean_axis => .mean_axis,
        .min_all => .min_all,
        .min_axis => .min_axis,
        .max_all => .max_all,
        .max_axis => .max_axis,
        .argmin_all => .argmin_all,
        .argmin_axis => .argmin_axis,
        .argmax_all => .argmax_all,
        .argmax_axis => .argmax_axis,
        .variance_all => .variance_all,
        .variance_axis => .variance_axis,
        .std_all => .std_all,
        .std_axis => .std_axis,
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
