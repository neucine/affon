const std = @import("std");
const compute = @import("compute_candidate");

pub const Backend = compute.Backend;
pub const OptimizationProfile = compute.OptimizationProfile;
pub const backendAvailable = compute.backendAvailable;

/// Candidate-only native boundary used by the T13 canaries. This module imports
/// only compute's stable facade; backend/compiler/runtime implementation modules
/// are deliberately unreachable here.
pub const Runtime = struct {
    allocator: std.mem.Allocator,
    session: *compute.Session,

    pub fn init(allocator: std.mem.Allocator, backend: compute.Backend) !Runtime {
        return .{ .allocator = allocator, .session = try compute.createSession(allocator, backend) };
    }

    pub fn deinit(self: *Runtime) !void {
        try self.session.deinit();
        self.* = undefined;
    }

    /// Immediate calls are convenience wrappers over a one-instruction Program.
    pub fn addF32(self: *Runtime, shape: []const usize, lhs_values: []const f32, rhs_values: []const f32) !*compute.Tensor {
        var spec = try compute.TensorSpec.init(self.allocator, .f32, shape, null);
        defer spec.deinit();
        var builder = compute.ProgramBuilder.init(self.allocator);
        defer builder.deinit();
        const lhs = try builder.addInput(spec);
        const rhs = try builder.addInput(spec);
        const sum = try builder.add(.add, .{ .none = {} }, &.{ lhs, rhs });
        const program = try builder.finish(sum);
        defer program.deinit();
        if (program.instructions().len != 1) return error.InvalidImmediateProgram;
        const lhs_tensor = try self.session.createTensor(spec, std.mem.sliceAsBytes(lhs_values));
        defer lhs_tensor.deinit();
        const rhs_tensor = try self.session.createTensor(spec, std.mem.sliceAsBytes(rhs_values));
        defer rhs_tensor.deinit();
        var compilation = try self.session.compile(program, .{});
        defer compilation.deinit();
        const outputs = try self.session.run(compilation.executable, &.{ lhs_tensor, rhs_tensor });
        if (outputs.len != 1) {
            self.session.releaseOutputs(outputs);
            return error.InvalidImmediateProgram;
        }
        const output = outputs[0];
        self.allocator.free(outputs);
        return output;
    }
};

pub const DecoderProof = struct {
    loss: f32,
    resumed_parameter: [12]f32,
    explanation: []u8,

    pub fn deinit(self: *DecoderProof, allocator: std.mem.Allocator) void {
        allocator.free(self.explanation);
        self.* = undefined;
    }
};

const BlockProof = struct {
    output: *compute.Tensor,
    explanation: []u8,
};

fn runDecoderBlock(allocator: std.mem.Allocator, session: *compute.Session, hidden_spec: compute.TensorSpec, hidden: *compute.Tensor, compile_options: compute.CompileOptions) !BlockProof {
    var projection_spec = try compute.TensorSpec.init(allocator, .f32, &.{ 3, 3 }, null);
    defer projection_spec.deinit();
    var expansion_spec = try compute.TensorSpec.init(allocator, .f32, &.{ 3, 6 }, null);
    defer expansion_spec.deinit();
    var contraction_spec = try compute.TensorSpec.init(allocator, .f32, &.{ 6, 3 }, null);
    defer contraction_spec.deinit();
    var builder = compute.ProgramBuilder.init(allocator);
    defer builder.deinit();
    const hidden_id = try builder.addInput(hidden_spec);
    const query_weight = try builder.addStateInput(projection_spec, .parameter, .fromInt(90));
    const key_weight = try builder.addStateInput(projection_spec, .parameter, .fromInt(91));
    const value_weight = try builder.addStateInput(projection_spec, .parameter, .fromInt(92));
    const expansion_weight = try builder.addStateInput(expansion_spec, .parameter, .fromInt(93));
    const contraction_weight = try builder.addStateInput(contraction_spec, .parameter, .fromInt(94));
    const query = try builder.add(.matmul, .{ .none = {} }, &.{ hidden_id, query_weight });
    const key = try builder.add(.matmul, .{ .none = {} }, &.{ hidden_id, key_weight });
    const value = try builder.add(.matmul, .{ .none = {} }, &.{ hidden_id, value_weight });
    const transposed_key = try builder.add(.transpose, .{ .transpose = .{ .permutation = &.{ 1, 0 } } }, key);
    const scores = try builder.add(.matmul, .{ .none = {} }, &.{ query[0], transposed_key[0] });
    const weights = try builder.add(.softmax, .{ .softmax = .{ .axis = 1 } }, scores);
    const attended = try builder.add(.matmul, .{ .none = {} }, &.{ weights[0], value[0] });
    const residual = try builder.add(.add, .{ .none = {} }, &.{ hidden_id, attended[0] });
    const normalized = try builder.add(.layer_norm, .{ .layer_norm = .{ .axis = 1, .eps = 1e-5 } }, residual);
    const expanded = try builder.add(.matmul, .{ .none = {} }, &.{ normalized[0], expansion_weight });
    const activated = try builder.add(.gelu, .{ .none = {} }, expanded);
    const contracted = try builder.add(.matmul, .{ .none = {} }, &.{ activated[0], contraction_weight });
    const block = try builder.add(.add, .{ .none = {} }, &.{ normalized[0], contracted[0] });
    const program = try builder.finish(block);
    defer program.deinit();

    const identity = [_]f32{ 1, 0, 0, 0, 1, 0, 0, 0, 1 };
    const query_tensor = try session.createTensor(projection_spec, std.mem.asBytes(&identity));
    defer query_tensor.deinit();
    const key_tensor = try session.createTensor(projection_spec, std.mem.asBytes(&identity));
    defer key_tensor.deinit();
    const value_tensor = try session.createTensor(projection_spec, std.mem.asBytes(&identity));
    defer value_tensor.deinit();
    const expansion_values = [_]f32{ 0.2, -0.1, 0.3, 0.1, -0.2, 0.05, -0.3, 0.4, 0.1, 0.2, 0.15, -0.25, 0.1, 0.2, -0.1, 0.3, -0.05, 0.25 };
    const expansion_tensor = try session.createTensor(expansion_spec, std.mem.asBytes(&expansion_values));
    defer expansion_tensor.deinit();
    const contraction_values = [_]f32{ 0.2, 0.1, -0.1, -0.2, 0.3, 0.1, 0.15, -0.1, 0.25, 0.05, 0.2, -0.3, -0.1, 0.4, 0.2, 0.3, -0.2, 0.1 };
    const contraction_tensor = try session.createTensor(contraction_spec, std.mem.asBytes(&contraction_values));
    defer contraction_tensor.deinit();
    var compilation = try session.compile(program, compile_options);
    defer compilation.deinit();
    var report = try compute.createExplanation(allocator, program, program, compilation.executable);
    defer report.deinit();
    const explanation = try report.serialize(allocator);
    errdefer allocator.free(explanation);
    const outputs = try session.run(compilation.executable, &.{ hidden, query_tensor, key_tensor, value_tensor, expansion_tensor, contraction_tensor });
    if (outputs.len != 1) {
        session.releaseOutputs(outputs);
        return error.InvalidDecoderProgram;
    }
    const output = outputs[0];
    allocator.free(outputs);
    return .{ .output = output, .explanation = explanation };
}

/// T13b's native decoder canary uses a deliberately small language-model head:
/// hidden states are projected to vocabulary logits, indexed cross entropy is
/// differentiated, and Adam returns the next parameter and optimizer state.
/// The persistent values are ordinary explicit bindings, so the byte snapshot
/// below is also the checkpoint boundary used to resume in a fresh Session.
pub fn runDecoderProof(allocator: std.mem.Allocator, backend: compute.Backend) !DecoderProof {
    return runDecoderProofWithOptions(allocator, backend, .{});
}

pub fn runDecoderProofWithOptions(allocator: std.mem.Allocator, backend: compute.Backend, compile_options: compute.CompileOptions) !DecoderProof {
    var hidden_spec = try compute.TensorSpec.init(allocator, .f32, &.{ 2, 3 }, null);
    defer hidden_spec.deinit();
    var parameter_spec = try compute.TensorSpec.init(allocator, .f32, &.{ 3, 4 }, null);
    defer parameter_spec.deinit();
    var target_spec = try compute.TensorSpec.init(allocator, .i64, &.{2}, null);
    defer target_spec.deinit();
    var scalar_spec = try compute.TensorSpec.init(allocator, .f32, &.{1}, null);
    defer scalar_spec.deinit();

    var forward_builder = compute.ProgramBuilder.init(allocator);
    defer forward_builder.deinit();
    const hidden_id = try forward_builder.addInput(hidden_spec);
    const parameter_id = try forward_builder.addStateInput(parameter_spec, .parameter, .fromInt(100));
    const target_id = try forward_builder.addInput(target_spec);
    const logits = try forward_builder.add(.matmul, .{ .none = {} }, &.{ hidden_id, parameter_id });
    const loss_id = try forward_builder.add(.cross_entropy_indexed, .{ .cross_entropy_indexed = .{ .axis = 1 } }, &.{ logits[0], target_id });
    const forward = try forward_builder.finish(loss_id);
    defer forward.deinit();
    var derivative = try compute.differentiate(allocator, forward, .{ .output = loss_id[0], .with_respect_to = &.{parameter_id} });
    defer derivative.deinit();

    const session = try compute.createSession(allocator, backend);
    defer session.deinit() catch unreachable;
    const hidden = try session.createTensor(hidden_spec, std.mem.asBytes(&[_]f32{ 1, 0, -1, 0.5, 1, 0.25 }));
    defer hidden.deinit();
    var block = try runDecoderBlock(allocator, session, hidden_spec, hidden, compile_options);
    defer block.output.deinit();
    defer allocator.free(block.explanation);
    const initial_parameter = [_]f32{ 0.2, -0.1, 0.3, 0.0, -0.2, 0.4, 0.1, -0.3, 0.05, 0.2, -0.15, 0.35 };
    const parameter = try session.createTensor(parameter_spec, std.mem.asBytes(&initial_parameter));
    defer parameter.deinit();
    const targets = try session.createTensor(target_spec, std.mem.asBytes(&[_]i64{ 2, 1 }));
    defer targets.deinit();
    const seed = try session.createTensor(scalar_spec, std.mem.asBytes(&[_]f32{1}));
    defer seed.deinit();
    var derivative_compilation = try session.compile(derivative.program(), compile_options);
    defer derivative_compilation.deinit();
    const derivative_outputs = try session.run(derivative_compilation.executable, &.{ block.output, parameter, targets, seed });
    defer session.releaseOutputs(derivative_outputs);
    const loss = f32Values(derivative_outputs[0])[0];
    if (!std.math.isFinite(loss)) return error.NonFiniteDecoderLoss;

    var update_builder = compute.ProgramBuilder.init(allocator);
    defer update_builder.deinit();
    const update_parameter = try update_builder.addStateInput(parameter_spec, .parameter, .fromInt(100));
    const gradient = try update_builder.addInput(parameter_spec);
    const first = try update_builder.addStateInput(parameter_spec, .optimizer_state, .fromInt(101));
    const second = try update_builder.addStateInput(parameter_spec, .optimizer_state, .fromInt(102));
    const update = try compute.addAdamUpdate(&update_builder, update_parameter, gradient, first, second, .{
        .learning_rate = 0.01,
        .bias_correction1 = 0.1,
        .bias_correction2 = 0.001,
    });
    const update_program = try update_builder.finish(&.{ update.parameter, update.first_moment, update.second_moment });
    defer update_program.deinit();
    const zeroes = [_]f32{0} ** 12;
    const first_tensor = try session.createTensor(parameter_spec, std.mem.asBytes(&zeroes));
    defer first_tensor.deinit();
    const second_tensor = try session.createTensor(parameter_spec, std.mem.asBytes(&zeroes));
    defer second_tensor.deinit();
    var update_compilation = try session.compile(update_program, compile_options);
    defer update_compilation.deinit();
    const first_update = try session.run(update_compilation.executable, &.{ parameter, derivative_outputs[1], first_tensor, second_tensor });
    defer session.releaseOutputs(first_update);

    var report = try compute.createExplanation(allocator, update_program, update_program, update_compilation.executable);
    defer report.deinit();
    const optimizer_explanation = try report.serialize(allocator);
    defer allocator.free(optimizer_explanation);
    const explanation = try std.fmt.allocPrint(allocator, "{{\"decoder\":{s},\"optimizer\":{s}}}", .{ block.explanation, optimizer_explanation });
    errdefer allocator.free(explanation);

    // Checkpoint/resume is intentionally session-independent: AFFON persists
    // these three output byte strings and rebinds them in a new Session.
    const resumed = try compute.createSession(allocator, backend);
    defer resumed.deinit() catch unreachable;
    const resumed_hidden = try resumed.createTensor(hidden_spec, hidden.bytes());
    defer resumed_hidden.deinit();
    var resumed_block = try runDecoderBlock(allocator, resumed, hidden_spec, resumed_hidden, compile_options);
    defer resumed_block.output.deinit();
    defer allocator.free(resumed_block.explanation);
    const original_block_values = f32Values(block.output);
    const resumed_block_values = f32Values(resumed_block.output);
    if (original_block_values.len != resumed_block_values.len) return error.CheckpointMismatch;
    for (original_block_values, resumed_block_values) |expected, actual| if (@abs(expected - actual) > 1e-5) return error.CheckpointMismatch;
    const resumed_parameter = try resumed.createTensor(parameter_spec, first_update[0].bytes());
    defer resumed_parameter.deinit();
    const resumed_gradient = try resumed.createTensor(parameter_spec, derivative_outputs[1].bytes());
    defer resumed_gradient.deinit();
    const resumed_first = try resumed.createTensor(parameter_spec, first_update[1].bytes());
    defer resumed_first.deinit();
    const resumed_second = try resumed.createTensor(parameter_spec, first_update[2].bytes());
    defer resumed_second.deinit();
    var resumed_compilation = try resumed.compile(update_program, compile_options);
    defer resumed_compilation.deinit();
    const second_update = try resumed.run(resumed_compilation.executable, &.{ resumed_parameter, resumed_gradient, resumed_first, resumed_second });
    defer resumed.releaseOutputs(second_update);
    var values: [12]f32 = undefined;
    @memcpy(&values, f32Values(second_update[0]));
    return .{ .loss = loss, .resumed_parameter = values, .explanation = explanation };
}

fn f32Values(tensor: *compute.Tensor) []const f32 {
    const bytes: []align(4) const u8 = @alignCast(tensor.bytes());
    return std.mem.bytesAsSlice(f32, bytes);
}

test "T13a candidate binding uses the frozen facade and one-instruction immediate program" {
    var runtime = try Runtime.init(std.testing.allocator, .cpu);
    defer runtime.deinit() catch unreachable;
    const output = try runtime.addF32(&.{2}, &.{ 1, 2 }, &.{ 10, 20 });
    defer output.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 11, 22 }, std.mem.bytesAsSlice(f32, @as([]align(4) u8, @alignCast(@constCast(output.bytes())))));
}

test "T13b candidate decoder proof trains explicit state and resumes a checkpoint on CPU" {
    var proof = try runDecoderProof(std.testing.allocator, .cpu);
    defer proof.deinit(std.testing.allocator);
    try std.testing.expect(std.math.isFinite(proof.loss));
    try std.testing.expect(proof.loss > 0);
    try std.testing.expect(std.mem.indexOf(u8, proof.explanation, "\"role\": \"parameter\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, proof.explanation, "\"role\": \"optimizer_state\"") != null);
    for (proof.resumed_parameter) |value| try std.testing.expect(std.math.isFinite(value));
}

test "T13b candidate decoder proof runs on one available accelerator" {
    const backend: compute.Backend = if (compute.backendAvailable(.metal)) .metal else if (compute.backendAvailable(.cuda)) .cuda else return error.SkipZigTest;
    var cpu = try runDecoderProof(std.testing.allocator, .cpu);
    defer cpu.deinit(std.testing.allocator);
    var accelerated = try runDecoderProof(std.testing.allocator, backend);
    defer accelerated.deinit(std.testing.allocator);
    try std.testing.expectApproxEqAbs(cpu.loss, accelerated.loss, 1e-5);
    for (cpu.resumed_parameter, accelerated.resumed_parameter) |expected, actual| try std.testing.expectApproxEqAbs(expected, actual, 1e-4);
}

test "O00c decoder proof has safe and unoptimized numerical parity" {
    var reference = try runDecoderProofWithOptions(std.testing.allocator, .cpu, .{ .optimization_profile = .off });
    defer reference.deinit(std.testing.allocator);
    var optimized = try runDecoderProofWithOptions(std.testing.allocator, .cpu, .{ .optimization_profile = .safe });
    defer optimized.deinit(std.testing.allocator);
    try std.testing.expectEqual(reference.loss, optimized.loss);
    try std.testing.expectEqualSlices(f32, &reference.resumed_parameter, &optimized.resumed_parameter);
}
