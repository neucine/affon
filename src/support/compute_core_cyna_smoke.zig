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
    try std.testing.expectError(error.InvalidShape,
        cyna.matmulBiasGeluF32(&lhs, &rhs, &short_bias, &output, 2, 2, 2));
    try std.testing.expectEqualSlices(f32, &.{ -1, -1, -1, -1 }, &output);
    try std.testing.expectError(error.InvalidShape,
        cyna.matmulBiasGeluF32(&lhs, &rhs, &[_]f32{ -20, -30 },
            &output, std.math.maxInt(usize), 2, 2));
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
    try std.testing.expectError(error.InvalidShape,
        cyna.quantizedMatmulF32(&activation, &weights, &scales, &biases, &output, 2, 8, 3, 3, 4));
    try std.testing.expectError(error.InvalidShape,
        cyna.quantizedMatmulF32(&activation, &weights, &scales, &biases, &output, 2, 8, 3, 8, 0));
    try std.testing.expectError(error.InvalidShape,
        cyna.quantizedMatmulF32(&activation, &weights, &scales, &biases, &output, std.math.maxInt(usize), 8, 3, 8, 4));
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
    try std.testing.expectError(error.InvalidShape,
        cyna.reduceF32(.sum, input[0..5], &whole, 2, 3, .all));
    try std.testing.expectError(error.InvalidShape,
        cyna.reduceF32(.sum, &input, &whole, std.math.maxInt(usize), 3, .all));
    try std.testing.expectEqual(@as(f32, -1), whole[0]);
}
