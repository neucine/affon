const std = @import("std");
const cyna = @import("compute_core_cyna.zig");

test "Affon infers mixed dtype selected graph results" {
    const inputs = [_]cyna.DenseInputShape{.{ .shape = &.{ 2, 2 } }};
    const steps = [_]cyna.DenseStep{
        .{ .operation = .argmin_axis, .lhs = 0, .rhs = 1 },
        .{ .operation = .argmax_all, .lhs = 0 },
    };
    const selected = [_]usize{ 0, 1, 2 };
    var ranks = [_]usize{ 99, 99, 99 };
    var elements = [_]usize{ 99, 99, 99 };
    var dimensions = [_]usize{ 99, 99, 99, 99, 99 };
    var dtypes = [_]c_int{ 99, 99, 99 };
    const written = try cyna.inferDenseGraphTyped(std.testing.allocator, .f32, &inputs,
        &steps, &selected, &ranks, &elements, &dimensions, &dtypes);
    try std.testing.expectEqual(@as(usize, 4), written);
    try std.testing.expectEqualSlices(usize, &.{ 2, 1, 1 }, &ranks);
    try std.testing.expectEqualSlices(usize, &.{ 4, 2, 1 }, &elements);
    try std.testing.expectEqualSlices(usize, &.{ 2, 2, 2, 1 }, dimensions[0..written]);
    try std.testing.expectEqualSlices(c_int, &.{ 1, 3, 3 }, &dtypes);
    ranks[0] = 99;
    dtypes[0] = 99;
    try std.testing.expectError(error.InvalidShape,
        cyna.inferDenseGraphTyped(std.testing.allocator, .f32, &inputs,
            &steps, &selected, &ranks, &elements, dimensions[0..3], &dtypes));
    try std.testing.expectEqual(@as(usize, 99), ranks[0]);
    try std.testing.expectEqual(@as(c_int, 99), dtypes[0]);
    try std.testing.expectError(error.InvalidShape,
        cyna.inferDenseGraph(std.testing.allocator, .f32, &inputs,
            &steps, &selected, &ranks, &elements, &dimensions));
}

test "Affon runs Cyna population statistics and integer mean" {
    const statistics = [_]cyna.DenseStep{
        .{ .operation = .variance_all, .lhs = 0 },
        .{ .operation = .std_all, .lhs = 0 },
        .{ .operation = .variance_axis, .lhs = 0, .rhs = 1 },
        .{ .operation = .std_axis, .lhs = 0, .rhs = 1 },
    };
    const float_inputs = [_]cyna.DenseInput{.{ .shape = &.{ 2, 2 }, .values = &.{ 1, 2, 3, 4 } }};
    const result = try cyna.allocDenseGraphF32(std.testing.allocator, &float_inputs, &statistics, &.{ 1, 2, 3, 4 }, .baseline);
    defer result.deinit(std.testing.allocator);
    try std.testing.expectApproxEqAbs(@as(f32, 1.25), result.output(0)[0], 0.00001);
    try std.testing.expectApproxEqAbs(@as(f32, 1.118034), result.output(1)[0], 0.00001);
    try std.testing.expectEqualSlices(f32, &.{ 0.25, 0.25 }, result.output(2));
    try std.testing.expectEqualSlices(f32, &.{ 0.5, 0.5 }, result.output(3));
    try std.testing.expectEqualSlices(usize, &.{2}, result.shape(2));
    const integer_inputs = [_]cyna.DenseInputI64{.{ .shape = &.{ 2, 2 }, .values = &.{ 1, 2, 3, 4 } }};
    const integer_steps = [_]cyna.DenseStep{
        .{ .operation = .mean_all, .lhs = 0 },
        .{ .operation = .mean_axis, .lhs = 0, .rhs = 1 },
        .{ .operation = .variance_all, .lhs = 0 },
    };
    const integers = try cyna.allocDenseGraphI64(std.testing.allocator, &integer_inputs, &integer_steps, &.{ 1, 2, 3 }, .baseline);
    defer integers.deinit(std.testing.allocator);
    try std.testing.expectEqualSlices(i64, &.{ 2, 1, 3, 1 }, integers.values);
    const doubles = [_]cyna.DenseInputF64{.{ .shape = &.{ 2, 2 }, .values = &.{ 1, 2, 3, 4 } }};
    const double_result = try cyna.allocDenseGraphF64(std.testing.allocator, &doubles, &statistics, &.{ 1, 3 }, .baseline);
    defer double_result.deinit(std.testing.allocator);
    try std.testing.expectEqualSlices(f64, &.{ 1.25, 0.25, 0.25 }, double_result.values);
    const unsigned = [_]cyna.DenseInputU32{.{ .shape = &.{ 2, 2 }, .values = &.{ 1, 2, 3, 4 } }};
    const unsigned_result = try cyna.allocDenseGraphU32(std.testing.allocator, &unsigned, &statistics, &.{ 1, 3 }, .baseline);
    defer unsigned_result.deinit(std.testing.allocator);
    try std.testing.expectEqualSlices(u32, &.{ 1, 0, 0 }, unsigned_result.values);
    const bits = [_]cyna.DenseInputBf16Bits{.{ .shape = &.{ 2, 2 }, .values = &.{ 0x3f80, 0x4000, 0x4040, 0x4080 } }};
    const bits_result = try cyna.allocDenseGraphBf16Bits(std.testing.allocator, &bits, &statistics, &.{ 1, 3, 4 }, .baseline);
    defer bits_result.deinit(std.testing.allocator);
    try std.testing.expectEqualSlices(u16, &.{ 0x3fa0, 0x3e80, 0x3e80, 0x3f00, 0x3f00 }, bits_result.values);
}

test "Affon runs Cyna i64 argmin and argmax with first-index ties" {
    const inputs = [_]cyna.DenseInputI64{.{ .shape = &.{ 2, 2 }, .values = &.{ 7, -3, 7, -3 } }};
    const steps = [_]cyna.DenseStep{
        .{ .operation = .argmin_all, .lhs = 0 },
        .{ .operation = .argmax_all, .lhs = 0 },
        .{ .operation = .argmin_axis, .lhs = 0, .rhs = 0 },
        .{ .operation = .argmax_axis, .lhs = 0, .rhs = 1 },
    };
    const result = try cyna.allocDenseGraphI64(std.testing.allocator, &inputs, &steps, &.{ 1, 2, 3, 4 }, .baseline);
    defer result.deinit(std.testing.allocator);
    try std.testing.expectEqualSlices(i64, &.{ 1, 0, 0, 0, 0, 0 }, result.values);
    try std.testing.expectEqualSlices(usize, &.{1}, result.shape(0));
    try std.testing.expectEqualSlices(usize, &.{2}, result.shape(2));
    const float_inputs = [_]cyna.DenseInput{.{ .shape = &.{ 2, 2 }, .values = &.{ 7, -3, 7, -3 } }};
    try std.testing.expectError(error.InvalidShape,
        cyna.allocDenseGraphF32(std.testing.allocator, &float_inputs, &steps, &.{1}, .baseline));
}

test "Affon runs Cyna dense min and max reductions" {
    const steps = [_]cyna.DenseStep{
        .{ .operation = .min_all, .lhs = 0 },
        .{ .operation = .max_all, .lhs = 0 },
        .{ .operation = .min_axis, .lhs = 0, .rhs = 1 },
        .{ .operation = .max_axis, .lhs = 0, .rhs = 0 },
    };
    const selected = [_]usize{ 1, 2, 3, 4 };
    const signed = [_]cyna.DenseInputI64{.{ .shape = &.{ 2, 2 }, .values = &.{ 7, -3, 2, 9 } }};
    const owned = try cyna.allocDenseGraphI64(std.testing.allocator, &signed, &steps, &selected, .baseline);
    defer owned.deinit(std.testing.allocator);
    try std.testing.expectEqualSlices(usize, &.{1}, owned.shape(0));
    try std.testing.expectEqualSlices(usize, &.{2}, owned.shape(2));
    try std.testing.expectEqualSlices(i64, &.{ -3, 9, -3, 2, 7, 9 }, owned.values);
    const unsigned = [_]cyna.DenseInputU32{.{ .shape = &.{ 2, 2 }, .values = &.{ 7, 3, 2, 9 } }};
    const unsigned_result = try cyna.allocDenseGraphU32(std.testing.allocator, &unsigned, &steps, &selected, .baseline);
    defer unsigned_result.deinit(std.testing.allocator);
    try std.testing.expectEqualSlices(u32, &.{ 2, 9, 3, 2, 7, 9 }, unsigned_result.values);
    const single = [_]cyna.DenseInput{.{ .shape = &.{ 2, 2 }, .values = &.{ 7, 3, 2, 9 } }};
    const single_result = try cyna.allocDenseGraphF32(std.testing.allocator, &single, &steps, &selected, .baseline);
    defer single_result.deinit(std.testing.allocator);
    try std.testing.expectEqualSlices(f32, &.{ 2, 9, 3, 2, 7, 9 }, single_result.values);
    const double = [_]cyna.DenseInputF64{.{ .shape = &.{ 2, 2 }, .values = &.{ 7, 3, 2, 9 } }};
    const double_result = try cyna.allocDenseGraphF64(std.testing.allocator, &double, &steps, &selected, .baseline);
    defer double_result.deinit(std.testing.allocator);
    try std.testing.expectEqualSlices(f64, &.{ 2, 9, 3, 2, 7, 9 }, double_result.values);
    const bits = [_]cyna.DenseInputBf16Bits{.{ .shape = &.{ 2, 2 }, .values = &.{ 0x40e0, 0x4040, 0x4000, 0x4110 } }};
    const bits_result = try cyna.allocDenseGraphBf16Bits(std.testing.allocator, &bits, &steps, &selected, .baseline);
    defer bits_result.deinit(std.testing.allocator);
    try std.testing.expectEqualSlices(u16, &.{ 0x4000, 0x4110, 0x4040, 0x4000, 0x40e0, 0x4110 }, bits_result.values);
}

test "Affon infers selected graph shapes for five Cyna dtypes" {
    const inputs = [_]cyna.DenseInputShape{
        .{ .shape = &.{ 2, 2 } },
        .{ .shape = &.{ 2, 2 } },
        .{ .shape = &.{2} },
    };
    const steps = [_]cyna.DenseStep{
        .{ .operation = .matmul, .lhs = 0, .rhs = 1 },
        .{ .operation = .add, .lhs = 3, .rhs = 2 },
        .{ .operation = .sum_axis, .lhs = 4, .rhs = 1 },
    };
    const selected = [_]usize{ 3, 4, 5 };
    inline for (.{ cyna.DenseDType.f32, .f64, .i64, .u32, .bf16 }) |dtype| {
        var ranks = [_]usize{ 99, 99, 99 };
        var elements = [_]usize{ 99, 99, 99 };
        var dimensions = [_]usize{ 99, 99, 99, 99, 99 };
        const written = try cyna.inferDenseGraph(std.testing.allocator, dtype,
            &inputs, &steps, &selected, &ranks, &elements, &dimensions);
        try std.testing.expectEqual(@as(usize, 5), written);
        try std.testing.expectEqualSlices(usize, &.{ 2, 2, 1 }, &ranks);
        try std.testing.expectEqualSlices(usize, &.{ 4, 4, 2 }, &elements);
        try std.testing.expectEqualSlices(usize, &.{ 2, 2, 2, 2, 2 }, &dimensions);
        ranks[0] = 99;
        elements[0] = 99;
        dimensions[0] = 99;
        try std.testing.expectError(error.InvalidShape,
            cyna.inferDenseGraph(std.testing.allocator, dtype, &inputs, &steps,
                &selected, &ranks, &elements, dimensions[0..4]));
        try std.testing.expectEqual(@as(usize, 99), ranks[0]);
        try std.testing.expectEqual(@as(usize, 99), elements[0]);
        try std.testing.expectEqual(@as(usize, 99), dimensions[0]);
    }
}

test "Affon calls Cyna signed i64 matmul through the C ABI" {
    const lhs = [_]i64{ 1, -2, 3, 4 };
    const rhs = [_]i64{ 5, 6, 7, 8 };
    var output = [_]i64{ 0, 0, 0, 0 };
    try cyna.matmulI64(&lhs, &rhs, &output, 2, 2, 2);
    try std.testing.expectEqualSlices(i64, &.{ -9, -10, 43, 50 }, &output);
    var unchanged = [_]i64{ 91, 92, 93, 94 };
    try std.testing.expectError(error.InvalidShape, cyna.matmulI64(lhs[0..3], &rhs, &unchanged, 2, 2, 2));
    try std.testing.expectEqualSlices(i64, &.{ 91, 92, 93, 94 }, &unchanged);
    try std.testing.expectError(error.InvalidShape, cyna.matmulI64(&lhs, &rhs, &unchanged, std.math.maxInt(usize), 2, 2));
    try std.testing.expectEqualSlices(i64, &.{ 91, 92, 93, 94 }, &unchanged);
}

test "Affon submits a shaped signed i64 Cyna graph" {
    const left = [_]i64{ 1, 2, 3, 4 };
    const right = [_]i64{ 5, 6, 7, 8 };
    const bias = [_]i64{ 10, 20 };
    const inputs = [_]cyna.DenseInputI64{
        .{ .shape = &.{ 2, 2 }, .values = &left },
        .{ .shape = &.{ 2, 2 }, .values = &right },
        .{ .shape = &.{2}, .values = &bias },
    };
    const steps = [_]cyna.DenseStep{
        .{ .operation = .matmul, .lhs = 0, .rhs = 1 },
        .{ .operation = .add, .lhs = 3, .rhs = 2 },
        .{ .operation = .sum_axis, .lhs = 4, .rhs = 1 },
    };
    const selected = [_]usize{ 3, 4, 5 };
    const owned = try cyna.allocDenseGraphI64(std.testing.allocator, &inputs, &steps, &selected, .memory);
    defer owned.deinit(std.testing.allocator);
    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, owned.shape(0));
    try std.testing.expectEqualSlices(usize, &.{2}, owned.shape(2));
    try std.testing.expectEqualSlices(i64, &.{ 19, 22, 43, 50 }, owned.output(0));
    try std.testing.expectEqualSlices(i64, &.{ 71, 123 }, owned.output(2));
    const bad_inputs = [_]cyna.DenseInputI64{
        .{ .shape = &.{ 2, 2 }, .values = left[0..3] },
        inputs[1],
        inputs[2],
    };
    try std.testing.expectError(error.InvalidShape,
        cyna.allocDenseGraphI64(std.testing.allocator, &bad_inputs, &steps, &selected, .baseline));
    var output = [_]i64{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    try cyna.runDenseGraphI64(std.testing.allocator, &inputs, &steps, &selected, &output);
    try std.testing.expectEqualSlices(i64, &.{ 19, 22, 43, 50, 29, 42, 53, 70, 71, 123 }, &output);
    try cyna.runDenseGraphWithGoalI64(std.testing.allocator, &inputs, &steps, &selected, &output, .memory);
    try std.testing.expectEqualSlices(i64, &.{ 19, 22, 43, 50, 29, 42, 53, 70, 71, 123 }, &output);
    var unchanged = [_]i64{ 91, 91, 91, 91, 91, 91, 91, 91, 91, 91 };
    try std.testing.expectError(error.InvalidShape,
        cyna.runDenseGraphI64(std.testing.allocator, &inputs, &steps, &selected, unchanged[0..9]));
    try std.testing.expectEqual(@as(i64, 91), unchanged[0]);
    const unsupported = [_]cyna.DenseStep{.{ .operation = .mean_axis, .lhs = 0, .rhs = 9 }};
    try std.testing.expectError(error.InvalidShape,
        cyna.runDenseGraphI64(std.testing.allocator, &inputs, &unsupported, &.{3}, unchanged[0..1]));
    try std.testing.expectEqual(@as(i64, 91), unchanged[0]);
}

test "Affon submits a precise f64 Cyna graph" {
    const left = [_]f64{ 16777217.0, 1.0, 0.0, 2.0 };
    const right = [_]f64{ 1.0, 0.0, 0.0, 1.0 };
    const inputs = [_]cyna.DenseInputF64{
        .{ .shape = &.{ 2, 2 }, .values = &left },
        .{ .shape = &.{ 2, 2 }, .values = &right },
    };
    const steps = [_]cyna.DenseStep{
        .{ .operation = .matmul, .lhs = 0, .rhs = 1 },
        .{ .operation = .sum_all, .lhs = 2 },
    };
    const selected = [_]usize{ 2, 3 };
    var output = [_]f64{ -1.0, -1.0, -1.0, -1.0, -1.0 };
    try cyna.runDenseGraphF64(std.testing.allocator, &inputs, &steps, &selected, &output);
    try std.testing.expectEqualSlices(f64, &.{ 16777217.0, 1.0, 0.0, 2.0, 16777220.0 }, &output);
    try cyna.runDenseGraphWithGoalF64(std.testing.allocator, &inputs, &steps, &selected, &output, .memory);
    try std.testing.expectEqualSlices(f64, &.{ 16777217.0, 1.0, 0.0, 2.0, 16777220.0 }, &output);
    output[0] = -1.0;
    try std.testing.expectError(error.InvalidShape,
        cyna.runDenseGraphF64(std.testing.allocator, &inputs, &steps, &selected, output[0..4]));
    try std.testing.expectEqual(@as(f64, -1.0), output[0]);
}

test "Affon submits a wrapping u32 Cyna graph" {
    const left = [_]u32{ std.math.maxInt(u32), 2, 3, 4 };
    const identity = [_]u32{ 1, 0, 0, 1 };
    const bias = [_]u32{ 2, 1 };
    const inputs = [_]cyna.DenseInputU32{
        .{ .shape = &.{ 2, 2 }, .values = &left },
        .{ .shape = &.{ 2, 2 }, .values = &identity },
        .{ .shape = &.{2}, .values = &bias },
    };
    const steps = [_]cyna.DenseStep{
        .{ .operation = .matmul, .lhs = 0, .rhs = 1 },
        .{ .operation = .add, .lhs = 3, .rhs = 2 },
        .{ .operation = .sum_axis, .lhs = 4, .rhs = 1 },
    };
    const selected = [_]usize{ 3, 4, 5 };
    var output = [_]u32{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    try cyna.runDenseGraphU32(std.testing.allocator, &inputs, &steps, &selected, &output);
    try std.testing.expectEqualSlices(u32, &.{ std.math.maxInt(u32), 2, 3, 4, 1, 3, 5, 5, 4, 10 }, &output);
    try cyna.runDenseGraphWithGoalU32(std.testing.allocator, &inputs, &steps, &selected, &output, .memory);
    try std.testing.expectEqualSlices(u32, &.{ std.math.maxInt(u32), 2, 3, 4, 1, 3, 5, 5, 4, 10 }, &output);
    output[0] = 91;
    try std.testing.expectError(error.InvalidShape,
        cyna.runDenseGraphU32(std.testing.allocator, &inputs, &steps, &selected, output[0..9]));
    try std.testing.expectEqual(@as(u32, 91), output[0]);
}

test "Affon submits exact bf16 bits through a Cyna graph" {
    const left = [_]u16{ 0x3f80, 0x4000, 0x4040, 0x4080 };
    const identity = [_]u16{ 0x3f80, 0, 0, 0x3f80 };
    const bias = [_]u16{ 0x3f80, 0x4000 };
    const inputs = [_]cyna.DenseInputBf16Bits{
        .{ .shape = &.{ 2, 2 }, .values = &left },
        .{ .shape = &.{ 2, 2 }, .values = &identity },
        .{ .shape = &.{2}, .values = &bias },
    };
    const steps = [_]cyna.DenseStep{
        .{ .operation = .matmul, .lhs = 0, .rhs = 1 },
        .{ .operation = .add, .lhs = 3, .rhs = 2 },
        .{ .operation = .sum_axis, .lhs = 4, .rhs = 1 },
    };
    const selected = [_]usize{ 3, 4, 5 };
    const expected = [_]u16{ 0x3f80, 0x4000, 0x4040, 0x4080, 0x4000, 0x4080, 0x4080, 0x40c0, 0x40c0, 0x4120 };
    var output = [_]u16{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    try cyna.runDenseGraphBf16Bits(std.testing.allocator, &inputs, &steps, &selected, &output);
    try std.testing.expectEqualSlices(u16, &expected, &output);
    try cyna.runDenseGraphWithGoalBf16Bits(std.testing.allocator, &inputs, &steps, &selected, &output, .memory);
    try std.testing.expectEqualSlices(u16, &expected, &output);
    output[0] = 0xffff;
    try std.testing.expectError(error.InvalidShape,
        cyna.runDenseGraphBf16Bits(std.testing.allocator, &inputs, &steps, &selected, output[0..9]));
    try std.testing.expectEqual(@as(u16, 0xffff), output[0]);
}

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

test "Affon selects the Cyna memory graph goal" {
    const first = [_]f32{ 1, 2 };
    const second = [_]f32{ 3, 4 };
    const bias = [_]f32{ 5, 6 };
    const inputs = [_]cyna.DenseInput{
        .{ .shape = &.{2}, .values = &first },
        .{ .shape = &.{2}, .values = &second },
        .{ .shape = &.{2}, .values = &bias },
    };
    const steps = [_]cyna.DenseStep{
        .{ .operation = .mul, .lhs = 0, .rhs = 1 },
        .{ .operation = .add, .lhs = 3, .rhs = 2 },
    };
    const selected = [_]usize{4};
    var output = [_]f32{ -1, -1 };
    try cyna.runDenseGraphWithGoalF32(std.testing.allocator, &inputs, &steps, &selected, &output, .memory);
    try std.testing.expectEqualSlices(f32, &.{ 8, 14 }, &output);
}
