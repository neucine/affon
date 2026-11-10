const std = @import("std");
const cyna = @import("compute_core_cyna.zig");

test "Affon calls Cyna matmul bias GELU through the C ABI" {
    const lhs = [_]f32{ 1, 2, 3, 4 };
    const rhs = [_]f32{ 5, 6, 7, 8 };
    const bias = [_]f32{ -20, -30 };
    var output = [_]f32{ -1, -1, -1, -1 };
    try cyna.matmulBiasGeluF32(&lhs, &rhs, &bias, &output, 2, 2, 2);
    try std.testing.expectApproxEqAbs(@as(f32, -0.15880801), output[0], 0.00001);
    try std.testing.expectApproxEqAbs(@as(f32, 0), output[1], 0.00001);
    try std.testing.expectEqual(@as(f32, 23), output[2]);
    try std.testing.expectEqual(@as(f32, 20), output[3]);
}

test "invalid Affon shape leaves caller output intact" {
    const lhs = [_]f32{ 1, 2, 3, 4 };
    const rhs = [_]f32{ 5, 6, 7, 8 };
    const short_bias = [_]f32{-20};
    var output = [_]f32{ -1, -1, -1, -1 };
    try std.testing.expectError(error.InvalidShape, cyna.matmulBiasGeluF32(&lhs, &rhs, &short_bias, &output, 2, 2, 2));
    try std.testing.expectEqualSlices(f32, &.{ -1, -1, -1, -1 }, &output);
    try std.testing.expectError(error.InvalidShape, cyna.matmulBiasGeluF32(&lhs, &rhs, &[_]f32{ -20, -30 }, &output, std.math.maxInt(usize), 2, 2));
    try std.testing.expectEqualSlices(f32, &.{ -1, -1, -1, -1 }, &output);
}

test "Affon calls Cyna quantized matmul through the C ABI" {
    const activation = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 2, 3, 4, 5, 6, 7, 8, 9 };
    const weights = [_]u32{ 0x01010101, 0x01010101, 0x02020202, 0x02020202, 0x03030303, 0x03030303 };
    const scales = [_]f32{ 1, 1, 1, 1, 1, 1 };
    const biases = [_]f32{ 0, 0, 0, 0, 0, 0 };
    var output = [_]f32{ -1, -1, -1, -1, -1, -1 };
    try cyna.quantizedMatmulF32(&activation, &weights, &scales, &biases, &output, 2, 8, 3, 8, 4);
    try std.testing.expectEqualSlices(f32, &.{ 36, 72, 108, 44, 88, 132 }, &output);

    output = @splat(-1);
    try std.testing.expectError(error.InvalidShape, cyna.quantizedMatmulF32(&activation, &weights, &scales, &biases, &output, 2, 8, 3, 3, 4));
    try std.testing.expectError(error.InvalidShape, cyna.quantizedMatmulF32(&activation, &weights, &scales, &biases, &output, 2, 8, 3, 8, 0));
    try std.testing.expectError(error.InvalidShape, cyna.quantizedMatmulF32(&activation, &weights, &scales, &biases, &output, std.math.maxInt(usize), 8, 3, 8, 4));
    try std.testing.expectEqualSlices(f32, &.{ -1, -1, -1, -1, -1, -1 }, &output);
}

test "Affon calls Cyna rank-two reductions through the C ABI" {
    const input = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var whole = [_]f32{-1};
    try cyna.reduceF32(.sum, &input, &whole, 2, 3, .all);
    try std.testing.expectEqual(@as(f32, 21), whole[0]);

    var columns = [_]f32{ -1, -1 };
    try cyna.reduceF32(.mean, &input, &columns, 2, 3, .columns);
    try std.testing.expectEqualSlices(f32, &.{ 2, 5 }, &columns);

    var rows = [_]f32{ -1, -1, -1 };
    try cyna.reduceF32(.variance, &input, &rows, 2, 3, .rows);
    try std.testing.expectEqualSlices(f32, &.{ 2.25, 2.25, 2.25 }, &rows);

    whole[0] = -1;
    try std.testing.expectError(error.InvalidShape, cyna.reduceF32(.sum, input[0..5], &whole, 2, 3, .all));
    try std.testing.expectError(error.InvalidShape, cyna.reduceF32(.sum, &input, &whole, std.math.maxInt(usize), 3, .all));
    try std.testing.expectEqual(@as(f32, -1), whole[0]);
}

test "Affon submits a shaped Cyna graph with mixed selected outputs" {
    const left = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const right = [_]f32{ 7, 8, 9, 10, 11, 12 };
    const inputs = [_]cyna.DenseInput{
        .{ .shape = &.{ 2, 3 }, .values = &left },
        .{ .shape = &.{ 3, 2 }, .values = &right },
    };
    const steps = [_]cyna.DenseStep{
        .{ .operation = .matmul, .lhs = 0, .rhs = 1 },
        .{ .operation = .sum_all, .lhs = 2 },
        .{ .operation = .mean_axis, .lhs = 2, .rhs = 1 },
    };
    const selected = [_]usize{ 2, 3, 4 };
    var output = [_]f32{ -1, -1, -1, -1, -1, -1, -1 };
    try cyna.runDenseGraphF32(std.testing.allocator, &inputs, &steps, &selected, &output);
    try std.testing.expectEqualSlices(f32, &.{ 58, 64, 139, 154, 415, 61, 146.5 }, &output);

    output = @splat(-1);
    const invalid = [_]cyna.DenseStep{.{ .operation = .matmul, .lhs = 0, .rhs = 2 }};
    try std.testing.expectError(error.InvalidShape, cyna.runDenseGraphF32(std.testing.allocator, &inputs, &invalid, &selected, &output));
    try std.testing.expectEqualSlices(f32, &.{ -1, -1, -1, -1, -1, -1, -1 }, &output);

    try std.testing.expectError(error.InvalidShape, cyna.runDenseGraphF32(std.testing.allocator, &inputs, &steps, &selected, output[0..6]));
    try std.testing.expectEqualSlices(f32, &.{ -1, -1, -1, -1, -1, -1, -1 }, &output);

    const no_steps = [_]cyna.DenseStep{};
    const input_selected = [_]usize{ 1, 0 };
    var identity: [12]f32 = @splat(0);
    try cyna.runDenseGraphF32(std.testing.allocator, &inputs, &no_steps, &input_selected, &identity);
    try std.testing.expectEqualSlices(f32, &.{ 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6 }, &identity);
}

test "Affon submits Cyna unary activation steps" {
    const values = [_]f32{0};
    const inputs = [_]cyna.DenseInput{.{ .shape = &.{1}, .values = &values }};
    const steps = [_]cyna.DenseStep{
        .{ .operation = .exp, .lhs = 0 },
        .{ .operation = .sigmoid, .lhs = 0 },
        .{ .operation = .sin, .lhs = 0 },
        .{ .operation = .cos, .lhs = 0 },
    };
    const selected = [_]usize{ 1, 2, 3, 4 };
    var output: [4]f32 = @splat(-1);
    try cyna.runDenseGraphF32(std.testing.allocator, &inputs, &steps, &selected, &output);
    try std.testing.expectEqualSlices(f32, &.{ 1, 0.5, 0, 1 }, &output);
}

test "Affon submits axis softmax and log softmax" {
    const values = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const inputs = [_]cyna.DenseInput{.{ .shape = &.{ 2, 3 }, .values = &values }};
    const steps = [_]cyna.DenseStep{
        .{ .operation = .softmax, .lhs = 0, .rhs = 1 },
        .{ .operation = .log_softmax, .lhs = 0, .rhs = 1 },
    };
    const selected = [_]usize{ 1, 2 };
    var output: [12]f32 = @splat(0);
    try cyna.runDenseGraphF32(std.testing.allocator, &inputs, &steps, &selected, &output);
    const expected_softmax = [_]f32{ 0.09003057, 0.24472848, 0.66524094 };
    const expected_log = [_]f32{ -2.407606, -1.407606, -0.407606 };
    for (0..6) |index| {
        try std.testing.expectApproxEqAbs(expected_softmax[index % 3], output[index], 0.00001);
        try std.testing.expectApproxEqAbs(expected_log[index % 3], output[index + 6], 0.00001);
    }
    const bad_steps = [_]cyna.DenseStep{.{ .operation = .softmax, .lhs = 0, .rhs = 2 }};
    var invalid_output: [6]f32 = @splat(-1);
    const one_selected = [_]usize{1};
    try std.testing.expectError(error.InvalidShape,
        cyna.runDenseGraphF32(std.testing.allocator, &inputs, &bad_steps, &one_selected, &invalid_output));
    try std.testing.expectEqualSlices(f32, &.{ -1, -1, -1, -1, -1, -1 }, &invalid_output);
}

test "Affon submits rank-zero Cyna graph inputs" {
    const left = [_]f32{3};
    const right = [_]f32{2};
    const inputs = [_]cyna.DenseInput{
        .{ .shape = &.{}, .values = &left },
        .{ .shape = &.{}, .values = &right },
    };
    const steps = [_]cyna.DenseStep{
        .{ .operation = .add, .lhs = 0, .rhs = 1 },
        .{ .operation = .neg, .lhs = 2 },
    };
    const selected = [_]usize{ 2, 3 };
    var output = [_]f32{ -1, -1 };
    try cyna.runDenseGraphF32(std.testing.allocator, &inputs, &steps, &selected, &output);
    try std.testing.expectEqualSlices(f32, &.{ 5, -5 }, &output);
    output = @splat(-1);
    try std.testing.expectError(error.InvalidShape,
        cyna.runDenseGraphF32(std.testing.allocator, &inputs, &steps, &selected, output[0..1]));
    try std.testing.expectEqualSlices(f32, &.{ -1, -1 }, &output);
}
