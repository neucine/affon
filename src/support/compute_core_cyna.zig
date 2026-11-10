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

extern fn compute_core_run_dense_graph_u32(
    goal: c_int,
    input_values: [*]const u32,
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
    output: [*]u32,
    output_value_count: usize,
) c_int;

extern fn compute_core_run_dense_graph_bf16(
    goal: c_int,
    input_bits: [*]const u16,
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
    output_bits: [*]u16,
    output_value_count: usize,
) c_int;

extern fn compute_core_infer_dense_graph(
    dtype: c_int,
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
    output_ranks: [*]usize,
    output_elements: [*]usize,
    output_dimensions: [*]usize,
    output_dimension_capacity: usize,
) c_int;

extern fn compute_core_infer_dense_graph_typed(
    dtype: c_int,
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
    output_ranks: [*]usize,
    output_elements: [*]usize,
    output_dimensions: [*]usize,
    output_dimension_capacity: usize,
    output_dtypes: [*]c_int,
) c_int;

extern fn compute_core_run_dense_graph_bytes(
    goal: c_int,
    dtype: c_int,
    input_bytes: [*]const u8,
    input_byte_count: usize,
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
    output_bytes: [*]u8,
    output_byte_count: usize,
) c_int;

extern fn compute_core_infer_dense_graph_multi_typed(
    input_dtypes: [*]const c_int,
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
    output_ranks: [*]usize,
    output_elements: [*]usize,
    output_dimensions: [*]usize,
    output_dimension_capacity: usize,
    output_dtypes: [*]c_int,
) c_int;

extern fn compute_core_run_dense_graph_multi_bytes(
    goal: c_int,
    input_dtypes: [*]const c_int,
    input_bytes: [*]const u8,
    input_byte_count: usize,
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
    output_bytes: [*]u8,
    output_byte_count: usize,
) c_int;

extern fn compute_core_infer_dense_graph_multi3_typed(
    input_dtypes: [*]const c_int,
    input_ranks: [*]const usize,
    input_count: usize,
    dimensions: [*]const usize,
    dimension_count: usize,
    operations: [*]const c_int,
    lhs_ids: [*]const usize,
    rhs_ids: [*]const usize,
    third_ids: [*]const usize,
    instruction_count: usize,
    output_ids: [*]const usize,
    selected_count: usize,
    output_ranks: [*]usize,
    output_elements: [*]usize,
    output_dimensions: [*]usize,
    output_dimension_capacity: usize,
    output_dtypes: [*]c_int,
) c_int;

extern fn compute_core_run_dense_graph_multi3_bytes(
    goal: c_int,
    input_dtypes: [*]const c_int,
    input_bytes: [*]const u8,
    input_byte_count: usize,
    input_ranks: [*]const usize,
    input_count: usize,
    dimensions: [*]const usize,
    dimension_count: usize,
    operations: [*]const c_int,
    lhs_ids: [*]const usize,
    rhs_ids: [*]const usize,
    third_ids: [*]const usize,
    instruction_count: usize,
    output_ids: [*]const usize,
    selected_count: usize,
    output_bytes: [*]u8,
    output_byte_count: usize,
) c_int;

extern fn compute_core_infer_dense_graph_multi_layout_typed(
    input_dtypes: [*]const c_int,
    input_ranks: [*]const usize,
    input_count: usize,
    dimensions: [*]const usize,
    dimension_count: usize,
    operations: [*]const c_int,
    lhs_ids: [*]const usize,
    rhs_ids: [*]const usize,
    third_ids: [*]const usize,
    metadata_offsets: [*]const usize,
    metadata_values: [*]const usize,
    metadata_count: usize,
    instruction_count: usize,
    output_ids: [*]const usize,
    selected_count: usize,
    output_ranks: [*]usize,
    output_elements: [*]usize,
    output_dimensions: [*]usize,
    output_dimension_capacity: usize,
    output_dtypes: [*]c_int,
) c_int;

extern fn compute_core_run_dense_graph_multi_layout_bytes(
    goal: c_int,
    input_dtypes: [*]const c_int,
    input_bytes: [*]const u8,
    input_byte_count: usize,
    input_ranks: [*]const usize,
    input_count: usize,
    dimensions: [*]const usize,
    dimension_count: usize,
    operations: [*]const c_int,
    lhs_ids: [*]const usize,
    rhs_ids: [*]const usize,
    third_ids: [*]const usize,
    metadata_offsets: [*]const usize,
    metadata_values: [*]const usize,
    metadata_count: usize,
    instruction_count: usize,
    output_ids: [*]const usize,
    selected_count: usize,
    output_bytes: [*]u8,
    output_byte_count: usize,
) c_int;

pub const DenseGoal = enum(c_int) { baseline = 0, memory = 1 };

fn denseInput(comptime T: type) type {
    return struct { shape: []const usize, values: []const T };
}

pub const DenseInput = denseInput(f32);
pub const DenseInputI64 = denseInput(i64);
pub const DenseInputF64 = denseInput(f64);
pub const DenseInputU32 = denseInput(u32);
pub const DenseInputBf16Bits = denseInput(u16);
pub const DenseInputShape = struct { shape: []const usize };
pub const DenseDType = enum(c_int) { f32 = 1, f64 = 2, i64 = 3, u32 = 4, bf16 = 5 };
pub const DenseByteInput = struct { dtype: DenseDType, shape: []const usize, bytes: []const u8 };

fn denseDTypeWidth(dtype: DenseDType) usize {
    return switch (dtype) { .f64, .i64 => 8, .f32, .u32 => 4, .bf16 => 2 };
}

pub const DenseGraphBytesResult = struct {
    bytes: []u8,
    ranks: []usize,
    elements: []usize,
    dimensions: []usize,
    dtypes: []c_int,

    pub fn deinit(self: @This(), allocator: std.mem.Allocator) void {
        allocator.free(self.bytes);
        allocator.free(self.ranks);
        allocator.free(self.elements);
        allocator.free(self.dimensions);
        allocator.free(self.dtypes);
    }

    pub fn dtype(self: @This(), index: usize) DenseDType {
        std.debug.assert(index < self.dtypes.len);
        return @enumFromInt(self.dtypes[index]);
    }

    pub fn shape(self: @This(), index: usize) []const usize {
        std.debug.assert(index < self.ranks.len);
        var offset: usize = 0;
        for (self.ranks[0..index]) |rank| offset += rank;
        return self.dimensions[offset..][0..self.ranks[index]];
    }

    pub fn outputBytes(self: @This(), index: usize) []const u8 {
        std.debug.assert(index < self.elements.len);
        var offset: usize = 0;
        for (self.elements[0..index], self.dtypes[0..index]) |count, code| {
            offset += count * denseDTypeWidth(@enumFromInt(code));
        }
        return self.bytes[offset..][0 .. self.elements[index] * denseDTypeWidth(self.dtype(index))];
    }
};

pub fn DenseGraphResult(comptime T: type) type {
    return struct {
        values: []T,
        ranks: []usize,
        elements: []usize,
        dimensions: []usize,

        pub fn deinit(self: @This(), allocator: std.mem.Allocator) void {
            allocator.free(self.values);
            allocator.free(self.ranks);
            allocator.free(self.elements);
            allocator.free(self.dimensions);
        }

        pub fn shape(self: @This(), index: usize) []const usize {
            std.debug.assert(index < self.ranks.len);
            var offset: usize = 0;
            for (self.ranks[0..index]) |rank| offset += rank;
            return self.dimensions[offset..][0..self.ranks[index]];
        }

        pub fn output(self: @This(), index: usize) []const T {
            std.debug.assert(index < self.elements.len);
            var offset: usize = 0;
            for (self.elements[0..index]) |count| offset += count;
            return self.values[offset..][0..self.elements[index]];
        }
    };
}

pub const DenseGraphResultF32 = DenseGraphResult(f32);
pub const DenseGraphResultF64 = DenseGraphResult(f64);
pub const DenseGraphResultI64 = DenseGraphResult(i64);
pub const DenseGraphResultU32 = DenseGraphResult(u32);
pub const DenseGraphResultBf16Bits = DenseGraphResult(u16);
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
    min_all = 25,
    max_all = 26,
    min_axis = 27,
    max_axis = 28,
    argmin_all = 29,
    argmax_all = 30,
    argmin_axis = 31,
    argmax_axis = 32,
    variance_all = 33,
    std_all = 34,
    variance_axis = 35,
    std_axis = 36,
    eq = 37,
    lt = 38,
    gt = 39,
    where = 40,
    cast = 41,
    transpose = 42,
    contiguous = 43,
    squeeze = 44,
    unsqueeze = 45,
    swap_axes = 46,
    reshape = 47,
    permute = 48,
    slice = 49,
    one_hot = 50,
    index_select = 51,
    embedding = 52,
    gather = 53,
    scatter_add = 54,
    top_k_values = 55,
    top_k_indices = 56,
    reduce_to_shape = 57,
    cat = 58,
    stack = 59,
};
// rhs is a value ID for binary steps, an axis for reductions and softmax,
// an index ID for indexed steps, a class count for one_hot, a DenseDType code
// for cast, or an axis for squeeze, unsqueeze, and swap_axes.
// Squeeze uses maxInt(usize) to remove all singleton axes. third is where's
// false branch, the second axis for swap_axes, an indexed axis, or updates ID.
// metadata is the target shape for reshape, axis order for permute, or
// start/stop/positive-step triples for each slice axis. Scatter add uses one
// metadata value for its axis. Top-k uses axis, k, and a 0/1 largest flag.
// Reduce-to-shape uses target dimensions; cat and stack use additional value IDs.
pub const DenseStep = struct { operation: DenseOperation, lhs: usize, rhs: usize = 0, third: usize = 0, metadata: []const usize = &.{} };

// Returns the number of flattened output dimensions written. Each selected
// output has one rank and element count; a scalar has rank zero and one element.
pub fn inferDenseGraph(
    allocator: std.mem.Allocator, dtype: DenseDType, inputs: []const DenseInputShape,
    steps: []const DenseStep, selected: []const usize, output_ranks: []usize,
    output_elements: []usize, output_dimensions: []usize,
) (Error || std.mem.Allocator.Error)!usize {
    return inferDenseGraphImpl(allocator, dtype, inputs, steps, selected,
        output_ranks, output_elements, output_dimensions, null);
}

pub fn inferDenseGraphTyped(
    allocator: std.mem.Allocator, dtype: DenseDType, inputs: []const DenseInputShape,
    steps: []const DenseStep, selected: []const usize, output_ranks: []usize,
    output_elements: []usize, output_dimensions: []usize, output_dtypes: []c_int,
) (Error || std.mem.Allocator.Error)!usize {
    return inferDenseGraphImpl(allocator, dtype, inputs, steps, selected,
        output_ranks, output_elements, output_dimensions, output_dtypes);
}

// Input and selected output buffers use little-endian dtype-width bytes.
// Use inferDenseGraphTyped to size and interpret each selected output.
pub fn runDenseGraphBytes(
    allocator: std.mem.Allocator, dtype: DenseDType, inputs: []const DenseInputShape,
    input_bytes: []const u8, steps: []const DenseStep, selected: []const usize,
    output_bytes: []u8, goal: DenseGoal,
) (Error || std.mem.Allocator.Error)!void {
    if (inputs.len == 0 or selected.len == 0 or
        steps.len > std.math.maxInt(usize) - inputs.len) return error.InvalidShape;
    const width = denseDTypeWidth(dtype);
    var dimension_count: usize = 0;
    var input_count: usize = 0;
    for (inputs) |input| {
        if (input.shape.len > 8) return error.InvalidShape;
        dimension_count = std.math.add(usize, dimension_count, input.shape.len) catch return error.InvalidShape;
        var elements: usize = 1;
        for (input.shape) |dimension| {
            if (dimension == 0) return error.InvalidShape;
            elements = std.math.mul(usize, elements, dimension) catch return error.InvalidShape;
        }
        input_count = std.math.add(usize, input_count, elements) catch return error.InvalidShape;
    }
    const expected_bytes = std.math.mul(usize, input_count, width) catch return error.InvalidShape;
    if (expected_bytes != input_bytes.len) return error.InvalidShape;
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
    var cursor: usize = 0;
    for (inputs, 0..) |input, index| {
        ranks[index] = input.shape.len;
        @memcpy(dimensions[cursor..][0..input.shape.len], input.shape);
        cursor += input.shape.len;
    }
    for (steps, 0..) |step, index| {
        operations[index] = @backingInt(step.operation);
        lhs_ids[index] = step.lhs;
        rhs_ids[index] = step.rhs;
    }
    switch (compute_core_run_dense_graph_bytes(@backingInt(goal), @backingInt(dtype),
        input_bytes.ptr, input_bytes.len, ranks.ptr, ranks.len, dimensions.ptr,
        dimensions.len, operations.ptr, lhs_ids.ptr, rhs_ids.ptr, steps.len,
        selected.ptr, selected.len, output_bytes.ptr, output_bytes.len)) {
        0 => {},
        1 => return error.InvalidShape,
        else => return error.CoreFailure,
    }
}

pub fn allocDenseGraphBytes(
    allocator: std.mem.Allocator, dtype: DenseDType, inputs: []const DenseInputShape,
    input_bytes: []const u8, steps: []const DenseStep, selected: []const usize,
    goal: DenseGoal,
) (Error || std.mem.Allocator.Error)!DenseGraphBytesResult {
    const max_dimensions = std.math.mul(usize, selected.len, 8) catch return error.InvalidShape;
    const ranks = try allocator.alloc(usize, selected.len);
    errdefer allocator.free(ranks);
    const elements = try allocator.alloc(usize, selected.len);
    errdefer allocator.free(elements);
    const dtypes = try allocator.alloc(c_int, selected.len);
    errdefer allocator.free(dtypes);
    const dimension_buffer = try allocator.alloc(usize, max_dimensions);
    defer allocator.free(dimension_buffer);
    const written = try inferDenseGraphTyped(allocator, dtype, inputs, steps, selected,
        ranks, elements, dimension_buffer, dtypes);
    const dimensions = try allocator.dupe(usize, dimension_buffer[0..written]);
    errdefer allocator.free(dimensions);
    var total_bytes: usize = 0;
    for (elements, dtypes) |count, code| {
        const output_dtype: DenseDType = switch (code) {
            1 => .f32, 2 => .f64, 3 => .i64, 4 => .u32, 5 => .bf16,
            else => return error.CoreFailure,
        };
        const size = std.math.mul(usize, count, denseDTypeWidth(output_dtype)) catch return error.InvalidShape;
        total_bytes = std.math.add(usize, total_bytes, size) catch return error.InvalidShape;
    }
    const bytes = try allocator.alloc(u8, total_bytes);
    errdefer allocator.free(bytes);
    try runDenseGraphBytes(allocator, dtype, inputs, input_bytes, steps, selected, bytes, goal);
    return .{ .bytes = bytes, .ranks = ranks, .elements = elements,
        .dimensions = dimensions, .dtypes = dtypes };
}

pub fn allocDenseGraphMultiBytes(
    allocator: std.mem.Allocator, inputs: []const DenseByteInput,
    steps: []const DenseStep, selected: []const usize, goal: DenseGoal,
) (Error || std.mem.Allocator.Error)!DenseGraphBytesResult {
    if (inputs.len == 0 or selected.len == 0 or
        steps.len > std.math.maxInt(usize) - inputs.len) return error.InvalidShape;
    var dimension_count: usize = 0;
    var input_byte_count: usize = 0;
    for (inputs) |input| {
        if (input.shape.len > 8) return error.InvalidShape;
        dimension_count = std.math.add(usize, dimension_count, input.shape.len) catch return error.InvalidShape;
        var elements: usize = 1;
        for (input.shape) |dimension| {
            if (dimension == 0) return error.InvalidShape;
            elements = std.math.mul(usize, elements, dimension) catch return error.InvalidShape;
        }
        const size = std.math.mul(usize, elements, denseDTypeWidth(input.dtype)) catch return error.InvalidShape;
        if (size != input.bytes.len) return error.InvalidShape;
        input_byte_count = std.math.add(usize, input_byte_count, size) catch return error.InvalidShape;
    }
    const input_ranks = try allocator.alloc(usize, inputs.len);
    defer allocator.free(input_ranks);
    const input_dimensions = try allocator.alloc(usize, dimension_count);
    defer allocator.free(input_dimensions);
    const input_dtypes = try allocator.alloc(c_int, inputs.len);
    defer allocator.free(input_dtypes);
    const input_bytes = try allocator.alloc(u8, input_byte_count);
    defer allocator.free(input_bytes);
    const operations = try allocator.alloc(c_int, steps.len);
    defer allocator.free(operations);
    const lhs_ids = try allocator.alloc(usize, steps.len);
    defer allocator.free(lhs_ids);
    const rhs_ids = try allocator.alloc(usize, steps.len);
    defer allocator.free(rhs_ids);
    const third_ids = try allocator.alloc(usize, steps.len);
    defer allocator.free(third_ids);
    var metadata_count: usize = 0;
    for (steps) |step| metadata_count = std.math.add(usize, metadata_count, step.metadata.len) catch return error.InvalidShape;
    const metadata_offsets = try allocator.alloc(usize, steps.len + 1);
    defer allocator.free(metadata_offsets);
    const metadata_values = try allocator.alloc(usize, metadata_count);
    defer allocator.free(metadata_values);
    metadata_offsets[0] = 0;
    var metadata_cursor: usize = 0;
    var dimension_cursor: usize = 0;
    var byte_cursor: usize = 0;
    for (inputs, 0..) |input, index| {
        input_ranks[index] = input.shape.len;
        input_dtypes[index] = @backingInt(input.dtype);
        @memcpy(input_dimensions[dimension_cursor..][0..input.shape.len], input.shape);
        @memcpy(input_bytes[byte_cursor..][0..input.bytes.len], input.bytes);
        dimension_cursor += input.shape.len;
        byte_cursor += input.bytes.len;
    }
    for (steps, 0..) |step, index| {
        operations[index] = @backingInt(step.operation);
        lhs_ids[index] = step.lhs;
        rhs_ids[index] = step.rhs;
        third_ids[index] = step.third;
        @memcpy(metadata_values[metadata_cursor..][0..step.metadata.len], step.metadata);
        metadata_cursor += step.metadata.len;
        metadata_offsets[index + 1] = metadata_cursor;
    }
    const max_output_dimensions = std.math.mul(usize, selected.len, 8) catch return error.InvalidShape;
    const ranks = try allocator.alloc(usize, selected.len);
    errdefer allocator.free(ranks);
    const elements = try allocator.alloc(usize, selected.len);
    errdefer allocator.free(elements);
    const dtypes = try allocator.alloc(c_int, selected.len);
    errdefer allocator.free(dtypes);
    const dimension_buffer = try allocator.alloc(usize, max_output_dimensions);
    defer allocator.free(dimension_buffer);
    switch (compute_core_infer_dense_graph_multi_layout_typed(input_dtypes.ptr,
        input_ranks.ptr, inputs.len, input_dimensions.ptr, input_dimensions.len,
        operations.ptr, lhs_ids.ptr, rhs_ids.ptr, third_ids.ptr, metadata_offsets.ptr, metadata_values.ptr,
        metadata_values.len, steps.len, selected.ptr,
        selected.len, ranks.ptr, elements.ptr, dimension_buffer.ptr,
        dimension_buffer.len, dtypes.ptr)) {
        0 => {}, 1 => return error.InvalidShape, else => return error.CoreFailure,
    }
    var written: usize = 0;
    for (ranks) |rank| written = std.math.add(usize, written, rank) catch return error.CoreFailure;
    const dimensions = try allocator.dupe(usize, dimension_buffer[0..written]);
    errdefer allocator.free(dimensions);
    var total_bytes: usize = 0;
    for (elements, dtypes) |count, code| {
        const output_dtype: DenseDType = switch (code) {
            1 => .f32, 2 => .f64, 3 => .i64, 4 => .u32, 5 => .bf16,
            else => return error.CoreFailure,
        };
        const size = std.math.mul(usize, count, denseDTypeWidth(output_dtype)) catch return error.InvalidShape;
        total_bytes = std.math.add(usize, total_bytes, size) catch return error.InvalidShape;
    }
    const bytes = try allocator.alloc(u8, total_bytes);
    errdefer allocator.free(bytes);
    switch (compute_core_run_dense_graph_multi_layout_bytes(@backingInt(goal),
        input_dtypes.ptr, input_bytes.ptr, input_bytes.len, input_ranks.ptr,
        inputs.len, input_dimensions.ptr, input_dimensions.len, operations.ptr,
        lhs_ids.ptr, rhs_ids.ptr, third_ids.ptr, metadata_offsets.ptr, metadata_values.ptr,
        metadata_values.len, steps.len, selected.ptr, selected.len,
        bytes.ptr, bytes.len)) {
        0 => {}, 1 => return error.InvalidShape, else => return error.CoreFailure,
    }
    return .{ .bytes = bytes, .ranks = ranks, .elements = elements,
        .dimensions = dimensions, .dtypes = dtypes };
}

fn inferDenseGraphImpl(
    allocator: std.mem.Allocator,
    dtype: DenseDType,
    inputs: []const DenseInputShape,
    steps: []const DenseStep,
    selected: []const usize,
    output_ranks: []usize,
    output_elements: []usize,
    output_dimensions: []usize,
    output_dtypes: ?[]c_int,
) (Error || std.mem.Allocator.Error)!usize {
    if (inputs.len == 0 or selected.len == 0 or
        output_ranks.len != selected.len or output_elements.len != selected.len or
        (output_dtypes != null and output_dtypes.?.len != selected.len) or
        steps.len > std.math.maxInt(usize) - inputs.len) return error.InvalidShape;
    var dimension_count: usize = 0;
    for (inputs) |input| {
        if (input.shape.len > 8) return error.InvalidShape;
        dimension_count = std.math.add(usize, dimension_count, input.shape.len) catch return error.InvalidShape;
        var elements: usize = 1;
        for (input.shape) |dimension| {
            if (dimension == 0) return error.InvalidShape;
            elements = std.math.mul(usize, elements, dimension) catch return error.InvalidShape;
        }
    }
    for (steps, 0..) |step, index| {
        const available = inputs.len + index;
        if (dtype != .i64 and output_dtypes == null and
            ((@backingInt(step.operation) >= 29 and @backingInt(step.operation) <= 32) or
            @backingInt(step.operation) >= 37)) return error.InvalidShape;
        if (step.lhs >= available or
            (switch (step.operation) {
                .add, .sub, .mul, .div, .matmul, .eq, .lt, .gt => step.rhs >= available,
                else => false,
            })) return error.InvalidShape;
        if (dtype == .i64 and switch (step.operation) {
            .add, .sub, .mul, .div, .abs, .neg, .relu, .matmul,
            .sum_all, .mean_all, .sum_axis, .mean_axis, .sign,
            .min_all, .max_all, .min_axis, .max_axis,
            .argmin_all, .argmax_all, .argmin_axis, .argmax_axis,
            .variance_all, .std_all, .variance_axis, .std_axis,
            .eq, .lt, .gt => false,
            else => true,
        }) return error.InvalidShape;
        if (dtype == .u32 and @backingInt(step.operation) > 12 and @backingInt(step.operation) < 25) return error.InvalidShape;
    }
    for (selected) |id| if (id >= inputs.len + steps.len) return error.InvalidShape;
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
    var cursor: usize = 0;
    for (inputs, 0..) |input, index| {
        ranks[index] = input.shape.len;
        @memcpy(dimensions[cursor..][0..input.shape.len], input.shape);
        cursor += input.shape.len;
    }
    for (steps, 0..) |step, index| {
        operations[index] = @backingInt(step.operation);
        lhs_ids[index] = step.lhs;
        rhs_ids[index] = step.rhs;
    }
    const status = if (output_dtypes) |dtypes|
        compute_core_infer_dense_graph_typed(@backingInt(dtype), ranks.ptr, ranks.len,
            dimensions.ptr, dimensions.len, operations.ptr, lhs_ids.ptr, rhs_ids.ptr,
            steps.len, selected.ptr, selected.len, output_ranks.ptr, output_elements.ptr,
            output_dimensions.ptr, output_dimensions.len, dtypes.ptr)
    else
        compute_core_infer_dense_graph(@backingInt(dtype), ranks.ptr, ranks.len,
            dimensions.ptr, dimensions.len, operations.ptr, lhs_ids.ptr, rhs_ids.ptr,
            steps.len, selected.ptr, selected.len, output_ranks.ptr, output_elements.ptr,
            output_dimensions.ptr, output_dimensions.len);
    switch (status) {
        0 => {},
        1 => return error.InvalidShape,
        else => return error.CoreFailure,
    }
    var written: usize = 0;
    for (output_ranks) |rank| {
        written = std.math.add(usize, written, rank) catch return error.CoreFailure;
    }
    return written;
}

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

pub fn runDenseGraphU32(
    allocator: std.mem.Allocator,
    inputs: []const DenseInputU32,
    steps: []const DenseStep,
    selected: []const usize,
    output: []u32,
) (Error || std.mem.Allocator.Error)!void {
    return runDenseGraphWithGoal(u32, allocator, inputs, steps, selected, output, .baseline);
}

pub fn runDenseGraphWithGoalU32(
    allocator: std.mem.Allocator,
    inputs: []const DenseInputU32,
    steps: []const DenseStep,
    selected: []const usize,
    output: []u32,
    goal: DenseGoal,
) (Error || std.mem.Allocator.Error)!void {
    return runDenseGraphWithGoal(u32, allocator, inputs, steps, selected, output, goal);
}

pub fn runDenseGraphBf16Bits(
    allocator: std.mem.Allocator,
    inputs: []const DenseInputBf16Bits,
    steps: []const DenseStep,
    selected: []const usize,
    output_bits: []u16,
) (Error || std.mem.Allocator.Error)!void {
    return runDenseGraphWithGoal(u16, allocator, inputs, steps, selected, output_bits, .baseline);
}

pub fn runDenseGraphWithGoalBf16Bits(
    allocator: std.mem.Allocator,
    inputs: []const DenseInputBf16Bits,
    steps: []const DenseStep,
    selected: []const usize,
    output_bits: []u16,
    goal: DenseGoal,
) (Error || std.mem.Allocator.Error)!void {
    return runDenseGraphWithGoal(u16, allocator, inputs, steps, selected, output_bits, goal);
}

pub fn allocDenseGraphF32(allocator: std.mem.Allocator, inputs: []const DenseInput, steps: []const DenseStep, selected: []const usize, goal: DenseGoal) (Error || std.mem.Allocator.Error)!DenseGraphResultF32 {
    return allocDenseGraph(f32, .f32, allocator, inputs, steps, selected, goal);
}

pub fn allocDenseGraphF64(allocator: std.mem.Allocator, inputs: []const DenseInputF64, steps: []const DenseStep, selected: []const usize, goal: DenseGoal) (Error || std.mem.Allocator.Error)!DenseGraphResultF64 {
    return allocDenseGraph(f64, .f64, allocator, inputs, steps, selected, goal);
}

pub fn allocDenseGraphI64(allocator: std.mem.Allocator, inputs: []const DenseInputI64, steps: []const DenseStep, selected: []const usize, goal: DenseGoal) (Error || std.mem.Allocator.Error)!DenseGraphResultI64 {
    return allocDenseGraph(i64, .i64, allocator, inputs, steps, selected, goal);
}

pub fn allocDenseGraphU32(allocator: std.mem.Allocator, inputs: []const DenseInputU32, steps: []const DenseStep, selected: []const usize, goal: DenseGoal) (Error || std.mem.Allocator.Error)!DenseGraphResultU32 {
    return allocDenseGraph(u32, .u32, allocator, inputs, steps, selected, goal);
}

pub fn allocDenseGraphBf16Bits(allocator: std.mem.Allocator, inputs: []const DenseInputBf16Bits, steps: []const DenseStep, selected: []const usize, goal: DenseGoal) (Error || std.mem.Allocator.Error)!DenseGraphResultBf16Bits {
    return allocDenseGraph(u16, .bf16, allocator, inputs, steps, selected, goal);
}

fn allocDenseGraph(
    comptime T: type,
    dtype: DenseDType,
    allocator: std.mem.Allocator,
    inputs: []const denseInput(T),
    steps: []const DenseStep,
    selected: []const usize,
    goal: DenseGoal,
) (Error || std.mem.Allocator.Error)!DenseGraphResult(T) {
    const max_dimensions = std.math.mul(usize, selected.len, 8) catch return error.InvalidShape;
    const shapes = try allocator.alloc(DenseInputShape, inputs.len);
    defer allocator.free(shapes);
    for (inputs, 0..) |input, index| shapes[index] = .{ .shape = input.shape };
    const ranks = try allocator.alloc(usize, selected.len);
    errdefer allocator.free(ranks);
    const elements = try allocator.alloc(usize, selected.len);
    errdefer allocator.free(elements);
    const dimension_buffer = try allocator.alloc(usize, max_dimensions);
    defer allocator.free(dimension_buffer);
    const written = try inferDenseGraph(allocator, dtype, shapes, steps, selected, ranks, elements, dimension_buffer);
    const dimensions = try allocator.dupe(usize, dimension_buffer[0..written]);
    errdefer allocator.free(dimensions);
    var total: usize = 0;
    for (elements) |count| total = std.math.add(usize, total, count) catch return error.InvalidShape;
    const values = try allocator.alloc(T, total);
    errdefer allocator.free(values);
    try runDenseGraphWithGoal(T, allocator, inputs, steps, selected, values, goal);
    return .{ .values = values, .ranks = ranks, .elements = elements, .dimensions = dimensions };
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
        if (T != i64 and ((@backingInt(step.operation) >= 29 and @backingInt(step.operation) <= 32) or
            @backingInt(step.operation) >= 37)) return error.InvalidShape;
        if (T == i64 and switch (step.operation) {
            .add, .sub, .mul, .div, .abs, .neg, .relu, .matmul,
            .sum_all, .mean_all, .sum_axis, .mean_axis, .sign,
            .min_all, .max_all, .min_axis, .max_axis,
            .argmin_all, .argmax_all, .argmin_axis, .argmax_axis,
            .variance_all, .std_all, .variance_axis, .std_axis,
            .eq, .lt, .gt => false,
            else => true,
        }) return error.InvalidShape;
        if (T == u32 and @backingInt(step.operation) > 12 and @backingInt(step.operation) < 25) return error.InvalidShape;
        if (step.lhs >= available or
            (switch (step.operation) {
                .add, .sub, .mul, .div, .matmul, .eq, .lt, .gt => step.rhs >= available,
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
    ) else if (T == f64) compute_core_run_dense_graph_f64(
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
    ) else if (T == u32) compute_core_run_dense_graph_u32(
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
    ) else compute_core_run_dense_graph_bf16(
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
