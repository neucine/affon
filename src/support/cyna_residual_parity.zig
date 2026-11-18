const std = @import("std");
const compute = @import("compute");
const cyna = @import("compute_core_cyna.zig");
const cyna_program = @import("cyna_program.zig");

test "Zig and Cyna select the same residual view decision and values" {
    const allocator = std.testing.allocator;
    const Builder = compute.testing.program.testing.Builder;
    var spec = try compute.testing.program.TensorSpec.init(allocator, .f32, &.{ 2, 2 }, null);
    defer spec.deinit();
    var builder = Builder.init(allocator);
    defer builder.deinit();
    const a = try builder.addInput(spec);
    const b = try builder.addInput(spec);
    const view = try builder.add(.transpose, .{ .transpose = .{ .permutation = &.{ 1, 0 } } }, &.{a});
    const early = try builder.add(.add, .{ .none = {} }, &.{ view[0], b });
    const middle = try builder.add(.mul, .{ .none = {} }, &.{ a, b });
    const late = try builder.add(.add, .{ .none = {} }, &.{ view[0], middle[0] });
    const combined = try builder.add(.add, .{ .none = {} }, &.{ early[0], late[0] });
    try builder.requireResidual(view[0], .recomputable_view, null, null);
    try builder.requireResidualUse(view[0], .{ .instruction = compute.testing.program.InstructionId.fromInt(1), .input_index = 0 });
    try builder.requireResidualUse(view[0], .{ .instruction = compute.testing.program.InstructionId.fromInt(3), .input_index = 0 });
    const graph = try builder.finish(&.{ early[0], late[0], combined[0] });
    defer graph.deinit();
    var lowering_diagnostic = cyna_program.LoweringDiagnostic{};
    var lowered = (try cyna_program.lower(allocator, graph, .memory, true, .recompute, .{}, &lowering_diagnostic)) orelse return error.CynaResidualLoweringFailed;
    defer lowered.deinit();
    const lowered_decision = try lowered.handle.describeResidualDecision(0);
    try std.testing.expectEqual(.recompute, lowered_decision.choice);
    try std.testing.expectEqual(@as(?usize, 1), lowered_decision.release_after);
    try std.testing.expectEqual(@as(?usize, 3), lowered_decision.recompute_before);

    const first = [_]f32{ 1, 2, 3, 4 };
    const second = [_]f32{ 10, 20, 30, 40 };
    const inputs = [_]cyna.DenseByteInput{
        .{ .dtype = .f32, .shape = &.{ 2, 2 }, .bytes = std.mem.sliceAsBytes(&first) },
        .{ .dtype = .f32, .shape = &.{ 2, 2 }, .bytes = std.mem.sliceAsBytes(&second) },
    };
    const steps = [_]cyna.DenseStep{
        .{ .operation = .transpose, .lhs = 0 },
        .{ .operation = .add, .lhs = 2, .rhs = 1 },
        .{ .operation = .mul, .lhs = 0, .rhs = 1 },
        .{ .operation = .add, .lhs = 2, .rhs = 4 },
        .{ .operation = .add, .lhs = 3, .rhs = 5 },
    };
    const requirements = [_]cyna.DenseResidualRequirement{.{ .value = 2, .recomputable_view = true }};
    const uses = [_]cyna.DenseResidualUse{
        .{ .value = 2, .instruction = 1, .input_index = 0 },
        .{ .value = 2, .instruction = 3, .input_index = 0 },
    };
    const session = try compute.testing.runtime.Session.init(allocator, .{
        .backend = compute.testing.backend.cpu.backend(),
    });
    defer session.deinit() catch unreachable;
    const ta = try session.createTensor(spec, inputs[0].bytes);
    defer ta.deinit();
    const tb = try session.createTensor(spec, inputs[1].bytes);
    defer tb.deinit();

    const cases = .{
        .{ .zig_policy = compute.testing.compiler.executable.ResidualPolicy.retain, .cyna_policy = cyna.DenseResidualPolicy.retain, .goal = compute.testing.compiler.policy.OptimizationGoal.memory, .cyna_goal = cyna.DenseGoal.memory, .recompute = false },
        .{ .zig_policy = compute.testing.compiler.executable.ResidualPolicy.recompute, .cyna_policy = cyna.DenseResidualPolicy.recompute, .goal = compute.testing.compiler.policy.OptimizationGoal.memory, .cyna_goal = cyna.DenseGoal.memory, .recompute = true },
        .{ .zig_policy = compute.testing.compiler.executable.ResidualPolicy.automatic, .cyna_policy = cyna.DenseResidualPolicy.automatic, .goal = compute.testing.compiler.policy.OptimizationGoal.memory, .cyna_goal = cyna.DenseGoal.memory, .recompute = true },
        .{ .zig_policy = compute.testing.compiler.executable.ResidualPolicy.automatic, .cyna_policy = cyna.DenseResidualPolicy.automatic, .goal = compute.testing.compiler.policy.OptimizationGoal.latency, .cyna_goal = cyna.DenseGoal.latency, .recompute = false },
    };
    inline for (cases) |case| {
        var zig = try session.compile(graph, .{ .optimization_goal = case.goal, .residual_policy = case.zig_policy });
        defer zig.deinit();
        var cy = try cyna.createDenseGraphHandleWithResiduals(allocator, &inputs, &steps, &.{ 3, 5, 6 }, case.cyna_goal, true, &requirements, &uses, case.cyna_policy);
        defer cy.deinit();
        const zig_decision = zig.executable.residualDecisions()[0];
        const cy_decision = try cy.describeResidualDecision(0);
        try std.testing.expectEqual(@as(usize, zig_decision.value().int()), cy_decision.value);
        try std.testing.expectEqual(case.recompute, zig_decision.choice() == .recompute);
        try std.testing.expectEqual(case.recompute, cy_decision.choice == .recompute);
        try std.testing.expectEqual(if (case.recompute) @as(?usize, 1) else null, if (zig_decision.releaseAfter()) |id| id.int() else null);
        try std.testing.expectEqual(if (case.recompute) @as(?usize, 1) else null, cy_decision.release_after);
        try std.testing.expectEqual(if (case.recompute) @as(?usize, 3) else null, if (zig_decision.recomputeBefore()) |id| id.int() else null);
        try std.testing.expectEqual(if (case.recompute) @as(?usize, 3) else null, cy_decision.recompute_before);
        const zig_physical = zig.executable.physicalGraph() orelse return error.MissingPhysicalGraph;
        try std.testing.expectEqual(zig_physical.nodes.items.len, cy.physicalNodeCount());
        const zig_outputs = try session.run(zig.executable, &.{ ta, tb });
        defer session.releaseOutputs(zig_outputs);
        if (case.cyna_policy == .recompute) {
            const lowered_outputs = try lowered.run(session, &.{ ta, tb });
            defer session.releaseOutputs(lowered_outputs);
            for (zig_outputs, lowered_outputs) |expected, actual|
                try std.testing.expectEqualSlices(u8, expected.bytes(), actual.bytes());
        }
        const tokens = [_]cyna.DenseGraphHandle.TensorToken{
            try cy.createInputTensor(0, inputs[0].bytes),
            try cy.createInputTensor(1, inputs[1].bytes),
        };
        defer for (tokens) |token| cy.releaseTensor(token) catch {};
        var cy_outputs: [3]cyna.DenseGraphHandle.TensorToken = undefined;
        try cy.runTensors(allocator, &tokens, &cy_outputs);
        defer for (cy_outputs) |token| cy.releaseTensor(token) catch {};
        for (cy_outputs, zig_outputs) |token, expected| {
            const bytes = try allocator.alloc(u8, expected.bytes().len);
            defer allocator.free(bytes);
            try cy.readTensor(token, bytes);
            try std.testing.expectEqualSlices(u8, expected.bytes(), bytes);
        }
    }
}

test "Zig and Cyna retain a residual view required as dense matmul input" {
    const allocator = std.testing.allocator;
    const Builder = compute.testing.program.testing.Builder;
    var spec = try compute.testing.program.TensorSpec.init(allocator, .f32, &.{ 2, 2 }, null);
    defer spec.deinit();
    var builder = Builder.init(allocator);
    defer builder.deinit();
    const a = try builder.addInput(spec);
    const b = try builder.addInput(spec);
    const view = try builder.add(.transpose, .{ .transpose = .{ .permutation = &.{ 1, 0 } } }, &.{a});
    const early = try builder.add(.matmul, .{ .none = {} }, &.{ view[0], b });
    _ = try builder.add(.mul, .{ .none = {} }, &.{ a, b });
    const late = try builder.add(.matmul, .{ .none = {} }, &.{ view[0], b });
    const combined = try builder.add(.add, .{ .none = {} }, &.{ early[0], late[0] });
    try builder.requireResidual(view[0], .recomputable_view, null, null);
    try builder.requireResidualUse(view[0], .{ .instruction = compute.testing.program.InstructionId.fromInt(1), .input_index = 0 });
    try builder.requireResidualUse(view[0], .{ .instruction = compute.testing.program.InstructionId.fromInt(3), .input_index = 0 });
    const graph = try builder.finish(combined);
    defer graph.deinit();

    const first = [_]f32{ 1, 2, 3, 4 };
    const second = [_]f32{ 10, 20, 30, 40 };
    const inputs = [_]cyna.DenseByteInput{
        .{ .dtype = .f32, .shape = &.{ 2, 2 }, .bytes = std.mem.sliceAsBytes(&first) },
        .{ .dtype = .f32, .shape = &.{ 2, 2 }, .bytes = std.mem.sliceAsBytes(&second) },
    };
    const steps = [_]cyna.DenseStep{
        .{ .operation = .transpose, .lhs = 0 },
        .{ .operation = .matmul, .lhs = 2, .rhs = 1 },
        .{ .operation = .mul, .lhs = 0, .rhs = 1 },
        .{ .operation = .matmul, .lhs = 2, .rhs = 1 },
        .{ .operation = .add, .lhs = 3, .rhs = 5 },
    };
    const requirements = [_]cyna.DenseResidualRequirement{.{ .value = 2, .recomputable_view = true }};
    const uses = [_]cyna.DenseResidualUse{
        .{ .value = 2, .instruction = 1, .input_index = 0 },
        .{ .value = 2, .instruction = 3, .input_index = 0 },
    };
    const session = try compute.testing.runtime.Session.init(allocator, .{
        .backend = compute.testing.backend.cpu.backend(),
    });
    defer session.deinit() catch unreachable;
    var zig = try session.compile(graph, .{ .optimization_goal = .memory, .residual_policy = .recompute });
    defer zig.deinit();
    var cy = try cyna.createDenseGraphHandleWithResiduals(allocator, &inputs, &steps, &.{6}, .memory, true, &requirements, &uses, .recompute);
    defer cy.deinit();
    try std.testing.expectEqual(compute.testing.compiler.executable.ResidualDecision.Choice.retain, zig.executable.residualDecisions()[0].choice());
    try std.testing.expectEqual(.retain, (try cy.describeResidualDecision(0)).choice);
    const ta = try session.createTensor(spec, inputs[0].bytes);
    defer ta.deinit();
    const tb = try session.createTensor(spec, inputs[1].bytes);
    defer tb.deinit();
    const expected = try session.run(zig.executable, &.{ ta, tb });
    defer session.releaseOutputs(expected);
    const tokens = [_]cyna.DenseGraphHandle.TensorToken{
        try cy.createInputTensor(0, inputs[0].bytes),
        try cy.createInputTensor(1, inputs[1].bytes),
    };
    defer for (tokens) |token| cy.releaseTensor(token) catch {};
    var output: [1]cyna.DenseGraphHandle.TensorToken = undefined;
    try cy.runTensors(allocator, &tokens, &output);
    defer cy.releaseTensor(output[0]) catch {};
    var actual: [4]f32 = undefined;
    try cy.readTensor(output[0], std.mem.sliceAsBytes(&actual));
    try std.testing.expectEqualSlices(u8, expected[0].bytes(), std.mem.sliceAsBytes(&actual));
}

test "Zig and Cyna retain a residual that is also a program output" {
    const allocator = std.testing.allocator;
    const Builder = compute.testing.program.testing.Builder;
    var spec = try compute.testing.program.TensorSpec.init(allocator, .f32, &.{ 2, 2 }, null);
    defer spec.deinit();
    var builder = Builder.init(allocator);
    defer builder.deinit();
    const a = try builder.addInput(spec);
    const b = try builder.addInput(spec);
    const view = try builder.add(.transpose, .{ .transpose = .{ .permutation = &.{ 1, 0 } } }, &.{a});
    const early = try builder.add(.add, .{ .none = {} }, &.{ view[0], b });
    const late = try builder.add(.add, .{ .none = {} }, &.{ view[0], early[0] });
    try builder.requireResidual(view[0], .recomputable_view, null, null);
    try builder.requireResidualUse(view[0], .{ .instruction = compute.testing.program.InstructionId.fromInt(1), .input_index = 0 });
    try builder.requireResidualUse(view[0], .{ .instruction = compute.testing.program.InstructionId.fromInt(2), .input_index = 0 });
    const graph = try builder.finish(&.{ view[0], late[0] });
    defer graph.deinit();

    const first = [_]f32{ 1, 2, 3, 4 };
    const second = [_]f32{ 10, 20, 30, 40 };
    const inputs = [_]cyna.DenseByteInput{
        .{ .dtype = .f32, .shape = &.{ 2, 2 }, .bytes = std.mem.sliceAsBytes(&first) },
        .{ .dtype = .f32, .shape = &.{ 2, 2 }, .bytes = std.mem.sliceAsBytes(&second) },
    };
    const steps = [_]cyna.DenseStep{
        .{ .operation = .transpose, .lhs = 0 },
        .{ .operation = .add, .lhs = 2, .rhs = 1 },
        .{ .operation = .add, .lhs = 2, .rhs = 3 },
    };
    const requirements = [_]cyna.DenseResidualRequirement{.{ .value = 2, .recomputable_view = true }};
    const uses = [_]cyna.DenseResidualUse{
        .{ .value = 2, .instruction = 1, .input_index = 0 },
        .{ .value = 2, .instruction = 2, .input_index = 0 },
    };
    const session = try compute.testing.runtime.Session.init(allocator, .{
        .backend = compute.testing.backend.cpu.backend(),
    });
    defer session.deinit() catch unreachable;
    var zig = try session.compile(graph, .{ .optimization_goal = .memory, .residual_policy = .recompute });
    defer zig.deinit();
    var cy = try cyna.createDenseGraphHandleWithResiduals(allocator, &inputs, &steps, &.{ 2, 4 }, .memory, true, &requirements, &uses, .recompute);
    defer cy.deinit();
    try std.testing.expectEqual(compute.testing.compiler.executable.ResidualDecision.Choice.retain, zig.executable.residualDecisions()[0].choice());
    const cy_decision = try cy.describeResidualDecision(0);
    try std.testing.expectEqual(.retain, cy_decision.choice);
    try std.testing.expectEqual(compute.testing.compiler.executable.ResidualDecision.Reason.required_program_output, zig.executable.residualDecisions()[0].reason());
    try std.testing.expectEqual(.required_program_output, cy_decision.reason);
    try std.testing.expectEqual(zig.executable.physicalGraph().?.nodes.items.len, cy.physicalNodeCount());

    const ta = try session.createTensor(spec, inputs[0].bytes);
    defer ta.deinit();
    const tb = try session.createTensor(spec, inputs[1].bytes);
    defer tb.deinit();
    const expected = try session.run(zig.executable, &.{ ta, tb });
    defer session.releaseOutputs(expected);
    const tokens = [_]cyna.DenseGraphHandle.TensorToken{
        try cy.createInputTensor(0, inputs[0].bytes),
        try cy.createInputTensor(1, inputs[1].bytes),
    };
    defer for (tokens) |token| cy.releaseTensor(token) catch {};
    var actual: [2]cyna.DenseGraphHandle.TensorToken = undefined;
    try cy.runTensors(allocator, &tokens, &actual);
    defer for (actual) |token| cy.releaseTensor(token) catch {};
    for (actual, expected, 0..) |token, output, index| {
        const bytes = try allocator.alloc(u8, output.bytes().len);
        defer allocator.free(bytes);
        try cy.readTensor(token, bytes);
        if (index == 0) {
            const logical_view = [_]f32{ 1, 3, 2, 4 };
            try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&logical_view), bytes);
            try std.testing.expectEqualSlices(isize, &.{ 1, 2 }, output.strides());
        } else {
            try std.testing.expectEqualSlices(u8, output.bytes(), bytes);
        }
    }
}

test "Zig and Cyna materialize a transposed value for dense reshape" {
    const allocator = std.testing.allocator;
    const Builder = compute.testing.program.testing.Builder;
    var spec = try compute.testing.program.TensorSpec.init(allocator, .f32, &.{ 2, 2 }, null);
    defer spec.deinit();
    var builder = Builder.init(allocator);
    defer builder.deinit();
    const input = try builder.addInput(spec);
    const view = try builder.add(.transpose, .{ .transpose = .{ .permutation = &.{ 1, 0 } } }, &.{input});
    const flat = try builder.add(.reshape, .{ .reshape = .{ .shape = &.{4} } }, view);
    const graph = try builder.finish(flat);
    defer graph.deinit();

    const values = [_]f32{ 1, 2, 3, 4 };
    const inputs = [_]cyna.DenseByteInput{.{ .dtype = .f32, .shape = &.{ 2, 2 }, .bytes = std.mem.sliceAsBytes(&values) }};
    const steps = [_]cyna.DenseStep{
        .{ .operation = .transpose, .lhs = 0 },
        .{ .operation = .reshape, .lhs = 1, .metadata = &.{4} },
    };
    const session = try compute.testing.runtime.Session.init(allocator, .{
        .backend = compute.testing.backend.cpu.backend(),
    });
    defer session.deinit() catch unreachable;
    var zig = try session.compile(graph, .{ .optimization_goal = .memory });
    defer zig.deinit();
    var cy = try cyna.createDenseGraphHandle(allocator, &inputs, &steps, &.{2}, .memory);
    defer cy.deinit();
    const zig_graph = zig.executable.physicalGraph() orelse return error.MissingPhysicalGraph;
    try std.testing.expectEqual(zig_graph.nodes.items.len, cy.physicalNodeCount());
    var zig_conversions: usize = 0;
    for (zig_graph.nodes.items) |node| switch (node.action) {
        .invoke => |invoke| {
            if (std.mem.eql(u8, invoke.implementation_id, "cpu_contiguous")) zig_conversions += 1;
        },
        .view => {},
    };
    var cyna_conversions: usize = 0;
    for (0..cy.physicalNodeCount()) |index| {
        var implementation: [80]u8 = undefined;
        const node = try cy.describePhysicalNode(index, &implementation);
        if (node.action == .conversion) cyna_conversions += 1;
    }
    try std.testing.expectEqual(@as(usize, 1), zig_conversions);
    try std.testing.expectEqual(@as(usize, 1), cyna_conversions);

    const tensor = try session.createTensor(spec, inputs[0].bytes);
    defer tensor.deinit();
    const expected = try session.run(zig.executable, &.{tensor});
    defer session.releaseOutputs(expected);
    const token = try cy.createInputTensor(0, inputs[0].bytes);
    defer cy.releaseTensor(token) catch {};
    var actual: [1]cyna.DenseGraphHandle.TensorToken = undefined;
    try cy.runTensors(allocator, &.{token}, &actual);
    defer cy.releaseTensor(actual[0]) catch {};
    var bytes: [16]u8 = undefined;
    try cy.readTensor(actual[0], &bytes);
    try std.testing.expectEqualSlices(u8, expected[0].bytes(), &bytes);
}

test "Zig and Cyna schedule the same independent memory branches" {
    const allocator = std.testing.allocator;
    const Builder = compute.testing.program.testing.Builder;
    var spec = try compute.testing.program.TensorSpec.init(allocator, .f32, &.{4}, null);
    defer spec.deinit();
    var builder = Builder.init(allocator);
    defer builder.deinit();
    const a = try builder.addInput(spec);
    const b = try builder.addInput(spec);
    const left = try builder.add(.neg, .{ .none = {} }, &.{a});
    const right = try builder.add(.neg, .{ .none = {} }, &.{b});
    const left_sum = try builder.add(.sum_all, .{ .reduce_all = .{} }, left);
    const right_sum = try builder.add(.sum_all, .{ .reduce_all = .{} }, right);
    const total = try builder.add(.add, .{ .none = {} }, &.{ left_sum[0], right_sum[0] });
    const source = try builder.finish(total);
    defer source.deinit();
    const first = [_]f32{ 1, 2, 3, 4 };
    const second = [_]f32{ 5, 6, 7, 8 };
    const inputs = [_]cyna.DenseByteInput{
        .{ .dtype = .f32, .shape = &.{4}, .bytes = std.mem.sliceAsBytes(&first) },
        .{ .dtype = .f32, .shape = &.{4}, .bytes = std.mem.sliceAsBytes(&second) },
    };
    const steps = [_]cyna.DenseStep{
        .{ .operation = .neg, .lhs = 0 },
        .{ .operation = .neg, .lhs = 1 },
        .{ .operation = .sum_all, .lhs = 2 },
        .{ .operation = .sum_all, .lhs = 3 },
        .{ .operation = .add, .lhs = 4, .rhs = 5 },
    };
    const session = try compute.testing.runtime.Session.init(allocator, .{
        .backend = compute.testing.backend.cpu.backend(),
    });
    defer session.deinit() catch unreachable;
    var zig = try session.compile(source, .{ .optimization_goal = .memory });
    defer zig.deinit();
    var cy = try cyna.createDenseGraphHandle(allocator, &inputs, &steps, &.{6}, .memory);
    defer cy.deinit();
    try cy.configureObservation(8);
    const zig_graph = zig.executable.physicalGraph() orelse return error.MissingPhysicalGraph;
    const zig_schedule = zig.executable.physicalSchedule() orelse return error.MissingPhysicalSchedule;
    try std.testing.expectEqual(zig_graph.nodes.items.len, cy.physicalNodeCount());
    const comparison = (try cy.selectionComparison()) orelse return error.MissingSelectionComparison;
    try std.testing.expectEqual(zig_schedule.peak_active_bytes, comparison.baseline_peak_bytes);
    var zig_order: [5]usize = undefined;
    for (zig_schedule.order, 0..) |node_id, index| {
        const node = zig_graph.nodes.items[node_id.int()];
        zig_order[index] = node.origins[0].int();
    }
    var input_bytes: [32]u8 = undefined;
    @memcpy(input_bytes[0..16], inputs[0].bytes);
    @memcpy(input_bytes[16..32], inputs[1].bytes);
    var result: [4]u8 = undefined;
    try cy.run(&input_bytes, &result);
    var dropped: usize = 0;
    try std.testing.expectEqual(@as(usize, 5), cy.observationCount(&dropped));
    var cy_order: [5]usize = undefined;
    for (&cy_order, 0..) |*instruction, index| {
        const event = try cy.readObservation(index);
        var name: [64]u8 = undefined;
        const node = try cy.describePhysicalNode(event.node_id, &name);
        const step = try cy.describeSemanticStep(node.step_index.?);
        instruction.* = step.source_instruction_index;
    }
    try std.testing.expectEqualSlices(usize, &zig_order, &cy_order);
    try std.testing.expectEqualSlices(usize, &.{ 0, 2, 1, 3, 4 }, &cy_order);
}

test "Zig and Cyna select the same add chains across overlapping fusions" {
    const allocator = std.testing.allocator;
    const Builder = compute.testing.program.testing.Builder;
    const values = [_]f32{ 1, 2 };
    const session = try compute.testing.runtime.Session.init(allocator, .{
        .backend = compute.testing.backend.cpu.backend(),
    });
    defer session.deinit() catch unreachable;
    for (2..13) |add_count| {
        var spec = try compute.testing.program.TensorSpec.init(allocator, .f32, &.{2}, null);
        defer spec.deinit();
        var builder = Builder.init(allocator);
        defer builder.deinit();
        const inputs = try allocator.alloc(cyna.DenseByteInput, add_count + 1);
        defer allocator.free(inputs);
        const steps = try allocator.alloc(cyna.DenseStep, add_count);
        defer allocator.free(steps);
        var first_value: ?compute.testing.program.ValueId = null;
        for (inputs, 0..) |*input, index| {
            const value = try builder.addInput(spec);
            if (index == 0) first_value = value;
            input.* = .{ .dtype = .f32, .shape = &.{2}, .bytes = std.mem.sliceAsBytes(&values) };
        }
        var current = first_value.?;
        for (steps, 0..) |*step, index| {
            const right = compute.testing.program.ValueId.fromInt(index + 1);
            current = (try builder.add(.add, .{ .none = {} }, &.{ current, right }))[0];
            step.* = .{ .operation = .add, .lhs = if (index == 0) 0 else add_count + index, .rhs = index + 1 };
        }
        const source = try builder.finish(&.{current});
        defer source.deinit();
        var zig = try session.compile(source, .{ .optimization_goal = .memory });
        defer zig.deinit();
        const selected = [_]usize{add_count * 2};
        var cy = try cyna.createDenseGraphHandle(allocator, inputs, steps, &selected, .memory);
        defer cy.deinit();
        const comparison = (try cy.selectionComparison()) orelse return error.MissingSelectionComparison;
        try std.testing.expectEqual(@as(usize, 16), comparison.budget);
        const search = try cy.searchEvidence();
        try std.testing.expectEqual(@as(usize, 16_384), search.limits.complete_evaluations);
        try std.testing.expect(search.evaluated_states <= search.limits.complete_evaluations);
        try std.testing.expect(!search.truncated);
        try std.testing.expect(!comparison.exclusion_search_truncated);
        const zig_graph = zig.executable.physicalGraph() orelse return error.MissingPhysicalGraph;
        try std.testing.expectEqual(zig_graph.nodes.items.len, cy.physicalNodeCount());
        var zig_fused: usize = 0;
        for (zig_graph.nodes.items) |node| switch (node.action) {
            .invoke => |invoke| {
                if (std.mem.eql(u8, invoke.implementation_id, "cpu_fused_add")) zig_fused += 1;
            },
            .view => {},
        };
        var cy_fused: usize = 0;
        for (0..cy.physicalNodeCount()) |index| {
            var name: [64]u8 = undefined;
            const node = try cy.describePhysicalNode(index, &name);
            if (std.mem.eql(u8, node.implementation, "cpu_fused_add")) cy_fused += 1;
        }
        try std.testing.expectEqual(zig_fused, cy_fused);
    }
}
