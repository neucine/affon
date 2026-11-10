const std = @import("std");

extern fn compute_core_run_matmul_i64(
    lhs: [*]const i64,
    lhs_count: usize,
    rhs: [*]const i64,
    rhs_count: usize,
    output: [*]i64,
    output_count: usize,
    rows: usize,
    depth: usize,
    columns: usize,
) c_int;

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
    activation: [*]const f32,
    activation_count: usize,
    weights: [*]const u32,
    weight_count: usize,
    scales: [*]const f32,
    scale_count: usize,
    biases: [*]const f32,
    bias_count: usize,
    output: [*]f32,
    output_count: usize,
    rows: usize,
    depth: usize,
    columns: usize,
    bits: usize,
    group_size: usize,
) c_int;

extern fn compute_core_run_reduce_f32(
    operation: c_int,
    input: [*]const f32,
    input_count: usize,
    output: [*]f32,
    output_count: usize,
    rows: usize,
    columns: usize,
    axis: c_int,
) c_int;

extern fn compute_core_run_dense_graph_with_goal_f32(
    goal: c_int,
    input_values: [*]const f32,
    input_value_count: usize,
    input_ranks: [*]const usize,
    input_count: usize,
    dimensions: [*]const usize,
    dimension_count: usize,
    operations: [*]const c_int,
    lhs_ids: [*]const usize,
    rhs_ids: [*]const usize,
    instruction_count: usize,
    output_ids: [*]const usize,
    selected_count: usize,
    output: [*]f32,
    output_value_count: usize,
) c_int;

extern fn compute_core_run_dense_graph_i64(
    goal: c_int,
    input_values: [*]const i64,
    input_value_count: usize,
    input_ranks: [*]const usize,
    input_count: usize,
    dimensions: [*]const usize,
    dimension_count: usize,
    operations: [*]const c_int,
    lhs_ids: [*]const usize,
    rhs_ids: [*]const usize,
    instruction_count: usize,
    output_ids: [*]const usize,
    selected_count: usize,
    output: [*]i64,
    output_value_count: usize,
) c_int;

extern fn compute_core_run_dense_graph_f64(
    goal: c_int,
    input_values: [*]const f64,
    input_value_count: usize,
    input_ranks: [*]const usize,
    input_count: usize,
    dimensions: [*]const usize,
    dimension_count: usize,
    operations: [*]const c_int,
    lhs_ids: [*]const usize,
    rhs_ids: [*]const usize,
    instruction_count: usize,
    output_ids: [*]const usize,
    selected_count: usize,
    output: [*]f64,
    output_value_count: usize,
) c_int;

pub const DenseGoal = enum(c_int) { baseline = 0, memory = 1 };

fn denseInput(comptime T: type) type {
    return struct { shape: []const usize, values: []const T };
}

pub const DenseInput = denseInput(f32);
pub const DenseInputI64 = denseInput(i64);
pub const DenseInputF64 = denseInput(f64);
pub const DenseOperation = enum(c_int) {
    add = 0,
    sub = 1,
    mul = 2,
    div = 3,
    abs = 4,
    neg = 5,
    relu = 6,
    matmul = 7,
    sum_all = 8,
    mean_all = 9,
    sum_axis = 10,
    mean_axis = 11,
    sign = 12,
    exp = 13,
    log = 14,
    sqrt = 15,
    sigmoid = 16,
    silu = 17,
    tanh = 18,
    erf = 19,
    gelu = 20,
    sin = 21,
    cos = 22,
    softmax = 23,
    log_softmax = 24,
};
// rhs is a value ID for binary steps and an axis for axis reductions and softmax.
pub const DenseStep = struct { operation: DenseOperation, lhs: usize, rhs: usize = 0 };

pub const Reduction = enum(c_int) { sum = 0, mean = 1, min = 2, max = 3, variance = 4, std = 5 };
pub const ReductionAxis = enum(c_int) { all = -1, rows = 0, columns = 1 };

pub const Error = error{ InvalidShape, CoreFailure };

pub fn matmulI64(lhs: []const i64, rhs: []const i64, output: []i64, rows: usize, depth: usize, columns: usize) Error!void {
    if (rows == 0 or depth == 0 or columns == 0) return error.InvalidShape;
    const lhs_count = std.math.mul(usize, rows, depth) catch return error.InvalidShape;
    const rhs_count = std.math.mul(usize, depth, columns) catch return error.InvalidShape;
    const output_count = std.math.mul(usize, rows, columns) catch return error.InvalidShape;
    if (lhs.len != lhs_count or rhs.len != rhs_count or output.len != output_count) return error.InvalidShape;
    switch (compute_core_run_matmul_i64(lhs.ptr, lhs.len, rhs.ptr, rhs.len, output.ptr, output.len, rows, depth, columns)) {
        0 => {},
        1 => return error.InvalidShape,
        else => return error.CoreFailure,
    }
}

// Isolated Affon consumer boundary for compute-core. The caller owns every slice.
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
        lhs.ptr,
        lhs.len,
        rhs.ptr,
        rhs.len,
        bias.ptr,
        bias.len,
        output.ptr,
        output.len,
        rows,
        depth,
        columns,
    )) {
        0 => {},
        1 => return error.InvalidShape,
        else => return error.CoreFailure,
    }
}

pub fn quantizedMatmulF32(
    activation: []const f32,
    weights: []const u32,
    scales: []const f32,
    biases: []const f32,
    output: []f32,
    rows: usize,
    depth: usize,
    columns: usize,
    bits: usize,
    group_size: usize,
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
        activation.ptr,
        activation.len,
        weights.ptr,
        weights.len,
        scales.ptr,
        scales.len,
        biases.ptr,
        biases.len,
        output.ptr,
        output.len,
        rows,
        depth,
        columns,
        bits,
        group_size,
    )) {
        0 => {},
        1 => return error.InvalidShape,
        else => return error.CoreFailure,
    }
}

pub fn reduceF32(
    operation: Reduction,
    input: []const f32,
    output: []f32,
    rows: usize,
    columns: usize,
    axis: ReductionAxis,
) Error!void {
    if (rows == 0 or columns == 0) return error.InvalidShape;
    const input_count = std.math.mul(usize, rows, columns) catch return error.InvalidShape;
    const output_count: usize = switch (axis) {
        .all => 1,
        .rows => columns,
        .columns => rows,
    };
    if (input.len != input_count or output.len != output_count) return error.InvalidShape;

    switch (compute_core_run_reduce_f32(
        @backingInt(operation),
        input.ptr,
        input.len,
        output.ptr,
        output.len,
        rows,
        columns,
        @backingInt(axis),
    )) {
        0 => {},
        1 => return error.InvalidShape,
        else => return error.CoreFailure,
    }
}

pub fn runDenseGraphF32(
    allocator: std.mem.Allocator,
    inputs: []const DenseInput,
    steps: []const DenseStep,
    selected: []const usize,
    output: []f32,
) (Error || std.mem.Allocator.Error)!void {
    return runDenseGraphWithGoalF32(allocator, inputs, steps, selected, output, .baseline);
}

pub fn runDenseGraphWithGoalF32(
    allocator: std.mem.Allocator,
    inputs: []const DenseInput,
    steps: []const DenseStep,
    selected: []const usize,
    output: []f32,
    goal: DenseGoal,
) (Error || std.mem.Allocator.Error)!void {
    return runDenseGraphWithGoal(f32, allocator, inputs, steps, selected, output, goal);
}

pub fn runDenseGraphI64(
    allocator: std.mem.Allocator,
    inputs: []const DenseInputI64,
    steps: []const DenseStep,
    selected: []const usize,
    output: []i64,
) (Error || std.mem.Allocator.Error)!void {
    return runDenseGraphWithGoal(i64, allocator, inputs, steps, selected, output, .baseline);
}

pub fn runDenseGraphWithGoalI64(
    allocator: std.mem.Allocator,
    inputs: []const DenseInputI64,
    steps: []const DenseStep,
    selected: []const usize,
    output: []i64,
    goal: DenseGoal,
) (Error || std.mem.Allocator.Error)!void {
    return runDenseGraphWithGoal(i64, allocator, inputs, steps, selected, output, goal);
}

pub fn runDenseGraphF64(
    allocator: std.mem.Allocator,
    inputs: []const DenseInputF64,
    steps: []const DenseStep,
    selected: []const usize,
    output: []f64,
) (Error || std.mem.Allocator.Error)!void {
    return runDenseGraphWithGoal(f64, allocator, inputs, steps, selected, output, .baseline);
}

pub fn runDenseGraphWithGoalF64(
    allocator: std.mem.Allocator,
    inputs: []const DenseInputF64,
    steps: []const DenseStep,
    selected: []const usize,
    output: []f64,
    goal: DenseGoal,
) (Error || std.mem.Allocator.Error)!void {
    return runDenseGraphWithGoal(f64, allocator, inputs, steps, selected, output, goal);
}

fn runDenseGraphWithGoal(
    comptime T: type,
    allocator: std.mem.Allocator,
    inputs: []const denseInput(T),
    steps: []const DenseStep,
    selected: []const usize,
    output: []T,
    goal: DenseGoal,
) (Error || std.mem.Allocator.Error)!void {
    if (inputs.len == 0 or selected.len == 0 or
        steps.len > std.math.maxInt(usize) - inputs.len) return error.InvalidShape;
    var dimension_count: usize = 0;
    var value_count: usize = 0;
    for (inputs) |input| {
        if (input.shape.len > 8) return error.InvalidShape;
        dimension_count = std.math.add(usize, dimension_count, input.shape.len) catch return error.InvalidShape;
        var elements: usize = 1;
        for (input.shape) |dimension| {
            if (dimension == 0) return error.InvalidShape;
            elements = std.math.mul(usize, elements, dimension) catch return error.InvalidShape;
        }
        if (elements != input.values.len) return error.InvalidShape;
        value_count = std.math.add(usize, value_count, elements) catch return error.InvalidShape;
    }
    for (steps, 0..) |step, index| {
        const available = inputs.len + index;
        if (T == i64 and switch (step.operation) {
            .add, .sub, .mul, .div, .abs, .neg, .relu, .matmul, .sum_all, .sum_axis, .sign => false,
            else => true,
        }) return error.InvalidShape;
        if (step.lhs >= available or
            (switch (step.operation) {
                .add, .sub, .mul, .div, .matmul => step.rhs >= available,
                else => false,
            }))
            return error.InvalidShape;
    }
    for (selected) |id| if (id >= inputs.len + steps.len) {
        return error.InvalidShape;
    };

    const values = try allocator.alloc(T, value_count);
    defer allocator.free(values);
    const ranks = try allocator.alloc(usize, inputs.len);
    defer allocator.free(ranks);
    const dimensions = try allocator.alloc(usize, dimension_count);
    defer allocator.free(dimensions);
    const operations = try allocator.alloc(c_int, steps.len);
    defer allocator.free(operations);
    const lhs_ids = try allocator.alloc(usize, steps.len);
    defer allocator.free(lhs_ids);
    const rhs_ids = try allocator.alloc(usize, steps.len);
    defer allocator.free(rhs_ids);
    var value_cursor: usize = 0;
    var dimension_cursor: usize = 0;
    for (inputs, 0..) |input, index| {
        ranks[index] = input.shape.len;
        @memcpy(values[value_cursor..][0..input.values.len], input.values);
        @memcpy(dimensions[dimension_cursor..][0..input.shape.len], input.shape);
        value_cursor += input.values.len;
        dimension_cursor += input.shape.len;
    }
    for (steps, 0..) |step, index| {
        operations[index] = @backingInt(step.operation);
        lhs_ids[index] = step.lhs;
        rhs_ids[index] = step.rhs;
    }
    const status = if (T == f32) compute_core_run_dense_graph_with_goal_f32(
        @backingInt(goal),
        values.ptr,
        values.len,
        ranks.ptr,
        ranks.len,
        dimensions.ptr,
        dimensions.len,
        operations.ptr,
        lhs_ids.ptr,
        rhs_ids.ptr,
        steps.len,
        selected.ptr,
        selected.len,
        output.ptr,
        output.len,
    ) else if (T == i64) compute_core_run_dense_graph_i64(
        @backingInt(goal),
        values.ptr,
        values.len,
        ranks.ptr,
        ranks.len,
        dimensions.ptr,
        dimensions.len,
        operations.ptr,
        lhs_ids.ptr,
        rhs_ids.ptr,
        steps.len,
        selected.ptr,
        selected.len,
        output.ptr,
        output.len,
    ) else compute_core_run_dense_graph_f64(
        @backingInt(goal),
        values.ptr,
        values.len,
        ranks.ptr,
        ranks.len,
        dimensions.ptr,
        dimensions.len,
        operations.ptr,
        lhs_ids.ptr,
        rhs_ids.ptr,
        steps.len,
        selected.ptr,
        selected.len,
        output.ptr,
        output.len,
    );
    switch (status) {
        0 => {},
        1 => return error.InvalidShape,
        else => return error.CoreFailure,
    }
}
