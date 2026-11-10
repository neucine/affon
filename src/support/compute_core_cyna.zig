const std = @import("std");

extern fn compute_core_run_matmul_bias_gelu_f32(
    lhs: [*]const f32,
    lhs_count: usize,
    rhs: [*]const f32,
    rhs_count: usize,
    bias: [*]const f32,
    bias_count: usize,
    output: [*]f32,
    output_count: usize,
    rows: usize,
    depth: usize,
    columns: usize,
) c_int;

extern fn compute_core_run_quantized_matmul_f32(
    activation: [*]const f32, activation_count: usize,
    weights: [*]const u32, weight_count: usize,
    scales: [*]const f32, scale_count: usize,
    biases: [*]const f32, bias_count: usize,
    output: [*]f32, output_count: usize,
    rows: usize, depth: usize, columns: usize, bits: usize, group_size: usize,
) c_int;

pub const Error = error{ InvalidShape, CoreFailure };

// First Affon consumer boundary for compute-core. The caller owns every slice.
pub fn matmulBiasGeluF32(
    lhs: []const f32,
    rhs: []const f32,
    bias: []const f32,
    output: []f32,
    rows: usize,
    depth: usize,
    columns: usize,
) Error!void {
    if (rows == 0 or depth == 0 or columns == 0) return error.InvalidShape;
    const lhs_count = std.math.mul(usize, rows, depth) catch return error.InvalidShape;
    const rhs_count = std.math.mul(usize, depth, columns) catch return error.InvalidShape;
    const output_count = std.math.mul(usize, rows, columns) catch return error.InvalidShape;
    if (lhs.len != lhs_count or rhs.len != rhs_count or
        bias.len != columns or output.len != output_count) return error.InvalidShape;

    switch (compute_core_run_matmul_bias_gelu_f32(
        lhs.ptr, lhs.len, rhs.ptr, rhs.len, bias.ptr, bias.len,
        output.ptr, output.len, rows, depth, columns,
    )) {
        0 => {},
        1 => return error.InvalidShape,
        else => return error.CoreFailure,
    }
}

pub fn quantizedMatmulF32(
    activation: []const f32, weights: []const u32,
    scales: []const f32, biases: []const f32, output: []f32,
    rows: usize, depth: usize, columns: usize, bits: usize, group_size: usize,
) Error!void {
    if (rows == 0 or depth == 0 or columns == 0 or group_size == 0 or
        (bits != 2 and bits != 4 and bits != 8)) return error.InvalidShape;
    const per_word = 32 / bits;
    if (depth % per_word != 0 or depth % group_size != 0) return error.InvalidShape;
    const activation_count = std.math.mul(usize, rows, depth) catch return error.InvalidShape;
    const weight_count = std.math.mul(usize, columns, depth / per_word) catch return error.InvalidShape;
    const group_count = std.math.mul(usize, columns, depth / group_size) catch return error.InvalidShape;
    const output_count = std.math.mul(usize, rows, columns) catch return error.InvalidShape;
    if (activation.len != activation_count or weights.len != weight_count or
        scales.len != group_count or biases.len != group_count or
        output.len != output_count) return error.InvalidShape;

    switch (compute_core_run_quantized_matmul_f32(
        activation.ptr, activation.len, weights.ptr, weights.len,
        scales.ptr, scales.len, biases.ptr, biases.len, output.ptr, output.len,
        rows, depth, columns, bits, group_size,
    )) {
        0 => {},
        1 => return error.InvalidShape,
        else => return error.CoreFailure,
    }
}
