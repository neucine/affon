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

extern fn compute_core_run_dense_graph_resident_f32(
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

extern fn compute_core_run_dense_graph_resident_i64(
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

extern fn compute_core_run_dense_graph_resident_f64(
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

extern fn compute_core_run_dense_graph_resident_u32(
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

extern fn compute_core_run_dense_graph_resident_bf16(
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

extern fn compute_core_run_dense_graph_resident_multi_layout_bytes(
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

extern fn compute_core_create_dense_graph_handle_guarded(
    goal: c_int,
    input_dtypes: [*]const c_int,
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
    dense_inputs: c_int,
) ?*anyopaque;
extern fn compute_core_create_dense_graph_handle_with_residuals(
    goal: c_int,
    input_dtypes: [*]const c_int,
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
    dense_inputs: c_int,
    requirements: [*]const usize,
    requirement_count: usize,
    uses: [*]const usize,
    use_count: usize,
    residual_policy: c_int,
) ?*anyopaque;
extern fn compute_core_create_dense_graph_handle_with_constants(
    goal: c_int,
    input_dtypes: [*]const c_int,
    input_byte_count: usize,
    input_ranks: [*]const usize,
    input_count: usize,
    dynamic_count: usize,
    constant_bytes: [*]const u8,
    constant_byte_count: usize,
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
    dense_inputs: c_int,
    requirements: [*]const usize,
    requirement_count: usize,
    uses: [*]const usize,
    use_count: usize,
    residual_policy: c_int,
) ?*anyopaque;
extern fn compute_core_create_dense_graph_handle_configured(
    goal: c_int,
    input_dtypes: [*]const c_int,
    input_byte_count: usize,
    input_ranks: [*]const usize,
    input_count: usize,
    dynamic_count: usize,
    constant_bytes: [*]const u8,
    constant_byte_count: usize,
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
    dense_inputs: c_int,
    requirements: [*]const usize,
    requirement_count: usize,
    uses: [*]const usize,
    use_count: usize,
    residual_policy: c_int,
    search_limit_words: [*]const usize,
    search_limit_count: usize,
) ?*anyopaque;
extern fn compute_core_run_dense_graph_handle(
    handle: ?*anyopaque,
    input_bytes: [*]const u8,
    input_byte_count: usize,
    output_bytes: [*]u8,
    output_byte_count: usize,
) c_int;
extern fn compute_core_dense_graph_handle_cache_hits(handle: ?*anyopaque) usize;
extern fn compute_core_dense_graph_handle_cache_misses(handle: ?*anyopaque) usize;
extern fn compute_core_dense_graph_handle_cache_bytes(handle: ?*anyopaque) usize;
extern fn compute_core_configure_dense_graph_observation(handle: ?*anyopaque, limit: usize) c_int;
extern fn compute_core_dense_graph_observation_count(handle: ?*anyopaque, dropped: *usize) usize;
extern fn compute_core_read_dense_graph_run_state(handle: ?*anyopaque, output: [*]usize, word_count: usize) c_int;
extern fn compute_core_read_dense_graph_selection(handle: ?*anyopaque, output: [*]usize, word_count: usize) c_int;
extern fn compute_core_read_dense_graph_search(handle: ?*anyopaque, output: [*]usize, word_count: usize) c_int;
extern fn compute_core_read_dense_graph_preparation(handle: ?*anyopaque, output: [*]usize, word_count: usize) c_int;
extern fn compute_core_describe_dense_graph_preparation_failure(handle: ?*anyopaque, index: usize, words: [*]usize, word_count: usize, implementation: [*]u8, implementation_capacity: usize, configuration: [*]u8, configuration_capacity: usize) c_int;
extern fn compute_core_read_dense_graph_resources(handle: ?*anyopaque, output: [*]usize, word_count: usize) c_int;
extern fn compute_core_dense_graph_physical_node_count(handle: ?*anyopaque) usize;
extern fn compute_core_dense_graph_physical_value_count(handle: ?*anyopaque) usize;
extern fn compute_core_read_dense_graph_physical_values(handle: ?*anyopaque, output: [*]usize, word_count: usize) c_int;
extern fn compute_core_read_dense_graph_physical_liveness(handle: ?*anyopaque, output: [*]usize, word_count: usize) c_int;
extern fn compute_core_dense_graph_residual_decision_count(handle: ?*anyopaque) usize;
extern fn compute_core_describe_dense_graph_residual_decision(handle: ?*anyopaque, index: usize, output: [*]usize, word_count: usize) c_int;
extern fn compute_core_describe_dense_graph_physical_node(handle: ?*anyopaque, index: usize, words: [*]usize, word_count: usize, implementation: [*]u8, implementation_capacity: usize, implementation_length: *usize) c_int;
extern fn compute_core_dense_graph_semantic_step_count(handle: ?*anyopaque) usize;
extern fn compute_core_describe_dense_graph_semantic_step(handle: ?*anyopaque, index: usize, output: *[5]usize, word_count: usize) c_int;
extern fn compute_core_read_dense_graph_observation(handle: ?*anyopaque, index: usize, output: *[9]usize, word_count: usize) c_int;
extern fn compute_core_destroy_dense_graph_handle(handle: ?*anyopaque) c_int;

extern fn compute_core_create_dense_graph_tensor(
    handle: ?*anyopaque,
    input_index: usize,
    bytes: [*]const u8,
    byte_count: usize,
    output: *[3]usize,
) c_int;
extern fn compute_core_read_dense_graph_tensor(
    handle: ?*anyopaque,
    owner: usize,
    slot: usize,
    generation: usize,
    output: [*]u8,
    byte_count: usize,
) c_int;
extern fn compute_core_describe_dense_graph_tensor(
    handle: ?*anyopaque,
    owner: usize,
    slot: usize,
    generation: usize,
    rank: *usize,
    backing_bytes: *usize,
    offset_bytes: *usize,
    strides: [*]usize,
    stride_count: usize,
) c_int;
extern fn compute_core_read_dense_graph_tensor_backing(
    handle: ?*anyopaque,
    owner: usize,
    slot: usize,
    generation: usize,
    output: [*]u8,
    byte_count: usize,
) c_int;
extern fn compute_core_run_dense_graph_tensors(
    handle: ?*anyopaque,
    input_tokens: [*]const usize,
    input_count: usize,
    output_tokens: [*]usize,
    output_count: usize,
) c_int;
extern fn compute_core_describe_dense_graph_tensors(
    handle: ?*anyopaque,
    tokens: [*]const usize,
    count: usize,
    words: [*]usize,
    word_count: usize,
) c_int;
extern fn compute_core_read_dense_graph_tensors_and_release(
    handle: ?*anyopaque,
    tokens: [*]const usize,
    count: usize,
    destinations: [*]const usize,
    byte_counts: [*]const usize,
) c_int;
extern fn compute_core_release_dense_graph_tensor(
    handle: ?*anyopaque,
    owner: usize,
    slot: usize,
    generation: usize,
) c_int;

extern fn compute_core_create_cpu_device_handle() ?*anyopaque;
extern fn compute_core_read_device_identity(
    handle: ?*anyopaque,
    output: *[3]usize,
) c_int;
extern fn compute_core_create_device_session(
    handle: ?*anyopaque,
    device_owner: usize,
    device_slot: usize,
    device_generation: usize,
    has_limit: c_int,
    max_resident_bytes: usize,
    output: *[3]usize,
) c_int;
extern fn compute_core_destroy_device_session(
    handle: ?*anyopaque,
    session_owner: usize,
    session_slot: usize,
    session_generation: usize,
) c_int;
extern fn compute_core_create_device_tensor(
    handle: ?*anyopaque,
    session_owner: usize,
    session_slot: usize,
    session_generation: usize,
    dtype: c_int,
    dimensions: [*]const usize,
    rank: usize,
    bytes: [*]const u8,
    byte_count: usize,
    output: *[3]usize,
) c_int;
extern fn compute_core_read_device_tensor(
    handle: ?*anyopaque,
    session_owner: usize,
    session_slot: usize,
    session_generation: usize,
    tensor_owner: usize,
    tensor_slot: usize,
    tensor_generation: usize,
    output: [*]u8,
    byte_count: usize,
) c_int;
extern fn compute_core_release_device_tensor(
    handle: ?*anyopaque,
    session_owner: usize,
    session_slot: usize,
    session_generation: usize,
    tensor_owner: usize,
    tensor_slot: usize,
    tensor_generation: usize,
) c_int;
extern fn compute_core_destroy_device_handle(handle: ?*anyopaque) c_int;

pub const DenseGoal = enum(c_int) { baseline = 0, memory = 1, balanced = 2, latency = 3, throughput = 4 };
pub const SearchLimits = struct {
    structural_states: usize = 4_096,
    structural_expansions: usize = 131_072,
    complete_evaluations: usize = 16_384,
    physical_work: usize = 8_192,
    ready_work: usize = 16_384,
    max_peak_active_bytes: ?usize = null,
    max_invocation_scratch_bytes: ?usize = null,
};

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
    return switch (dtype) {
        .f64, .i64 => 8,
        .f32, .u32 => 4,
        .bf16 => 2,
    };
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
        return @fromBackingInt(@intCast(self.dtypes[index]));
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
            offset += count * denseDTypeWidth(@fromBackingInt(@intCast(code)));
        }
        return self.bytes[offset..][0 .. self.elements[index] * denseDTypeWidth(self.dtype(index))];
    }
};

pub const DenseGraphInferred = struct {
    ranks: []usize,
    elements: []usize,
    dimensions: []usize,
    dtypes: []c_int,

    pub fn deinit(self: @This(), allocator: std.mem.Allocator) void {
        allocator.free(self.ranks);
        allocator.free(self.elements);
        allocator.free(self.dimensions);
        allocator.free(self.dtypes);
    }

    pub fn dtype(self: @This(), index: usize) DenseDType {
        std.debug.assert(index < self.dtypes.len);
        return @fromBackingInt(@intCast(self.dtypes[index]));
    }

    pub fn shape(self: @This(), index: usize) []const usize {
        std.debug.assert(index < self.ranks.len);
        var offset: usize = 0;
        for (self.ranks[0..index]) |rank| offset += rank;
        return self.dimensions[offset..][0..self.ranks[index]];
    }
};

pub const HandleIdentity = struct {
    owner: usize,
    slot: usize,
    generation: usize,
};

/// Persistent CPU device. Destroy its sessions before calling `deinit`.
pub const CpuDeviceHandle = struct {
    raw: ?*anyopaque,
    identity: HandleIdentity,

    pub fn create() Error!CpuDeviceHandle {
        const raw = compute_core_create_cpu_device_handle() orelse return error.CoreFailure;
        var parts: [3]usize = undefined;
        switch (compute_core_read_device_identity(raw, &parts)) {
            0 => return .{
                .raw = raw,
                .identity = .{ .owner = parts[0], .slot = parts[1], .generation = parts[2] },
            },
            1 => return error.InvalidShape,
            else => return error.CoreFailure,
        }
    }

    pub fn deinit(self: *CpuDeviceHandle) void {
        if (self.raw) |raw| {
            const status = compute_core_destroy_device_handle(raw);
            std.debug.assert(status == 0);
            self.raw = null;
        }
    }

    pub fn createSession(
        self: *const CpuDeviceHandle,
        max_resident_bytes: ?usize,
    ) Error!CpuSessionHandle {
        if (self.raw == null) return error.InvalidShape;
        var parts: [3]usize = undefined;
        const has_limit: c_int = if (max_resident_bytes == null) 0 else 1;
        switch (compute_core_create_device_session(
            self.raw,
            self.identity.owner,
            self.identity.slot,
            self.identity.generation,
            has_limit,
            max_resident_bytes orelse 0,
            &parts,
        )) {
            0 => return .{
                .raw = self.raw,
                .device_identity = self.identity,
                .identity = .{ .owner = parts[0], .slot = parts[1], .generation = parts[2] },
            },
            1 => return error.InvalidShape,
            else => return error.CoreFailure,
        }
    }
};

pub const CpuSessionHandle = struct {
    raw: ?*anyopaque,
    device_identity: HandleIdentity,
    identity: HandleIdentity,

    pub fn deinit(self: *CpuSessionHandle) void {
        if (self.raw) |raw| {
            const status = compute_core_destroy_device_session(
                raw,
                self.identity.owner,
                self.identity.slot,
                self.identity.generation,
            );
            std.debug.assert(status == 0);
            self.raw = null;
        }
    }

    pub fn createTensor(
        self: *const CpuSessionHandle,
        dtype: DenseDType,
        shape: []const usize,
        bytes: []const u8,
    ) Error!HandleIdentity {
        if (self.raw == null or shape.len > 8) return error.InvalidShape;
        var expected: usize = 1;
        for (shape) |dimension| {
            if (dimension == 0) return error.InvalidShape;
            expected = std.math.mul(usize, expected, dimension) catch return error.InvalidShape;
        }
        const byte_count = std.math.mul(usize, expected, denseDTypeWidth(dtype)) catch
            return error.InvalidShape;
        if (byte_count != bytes.len) return error.InvalidShape;
        var parts: [3]usize = undefined;
        const scalar_dimensions = [_]usize{0};
        const dimensions: [*]const usize = if (shape.len == 0) &scalar_dimensions else shape.ptr;
        switch (compute_core_create_device_tensor(
            self.raw,
            self.identity.owner,
            self.identity.slot,
            self.identity.generation,
            @backingInt(dtype),
            dimensions,
            shape.len,
            bytes.ptr,
            bytes.len,
            &parts,
        )) {
            0 => return .{ .owner = parts[0], .slot = parts[1], .generation = parts[2] },
            1 => return error.InvalidShape,
            else => return error.CoreFailure,
        }
    }

    pub fn readTensor(
        self: *const CpuSessionHandle,
        token: HandleIdentity,
        output: []u8,
    ) Error!void {
        if (self.raw == null) return error.InvalidShape;
        switch (compute_core_read_device_tensor(
            self.raw,
            self.identity.owner,
            self.identity.slot,
            self.identity.generation,
            token.owner,
            token.slot,
            token.generation,
            output.ptr,
            output.len,
        )) {
            0 => {},
            1 => return error.InvalidShape,
            else => return error.CoreFailure,
        }
    }

    pub fn releaseTensor(
        self: *const CpuSessionHandle,
        token: HandleIdentity,
    ) Error!void {
        if (self.raw == null) return error.InvalidShape;
        switch (compute_core_release_device_tensor(
            self.raw,
            self.identity.owner,
            self.identity.slot,
            self.identity.generation,
            token.owner,
            token.slot,
            token.generation,
        )) {
            0 => {},
            1 => return error.InvalidShape,
            else => return error.CoreFailure,
        }
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
    masked_fill = 60,
    clamp = 61,
    layer_norm = 62,
    rms_norm = 63,
    cross_entropy = 64,
    log_softmax_nll = 65,
    cross_entropy_indexed = 66,
    quantized_matmul = 67,
    dot = 68,
};

// Codes exported by Compute Core's OpTag, in its declared order. Keep this
// table in sync with compute-core/src/program/types.cy when the C ABI changes.
const cyna_operation_names = [_][]const u8{
    "",              "add",           "sub",                   "mul",        "div",           "matmul",           "abs",             "neg",        "sign",            "relu",
    "sum_all",       "mean_all",      "sum_axis",              "mean_axis",  "min_all",       "max_all",          "argmin_all",      "argmax_all", "min_axis",        "max_axis",
    "argmin_axis",   "argmax_axis",   "variance_all",          "std_all",    "variance_axis", "std_axis",         "reshape",         "permute",    "transpose",       "squeeze",
    "unsqueeze",     "contiguous",    "slice",                 "cast",       "scatter_add",   "gather",           "index_select",    "embedding",  "one_hot",         "top_k_values",
    "top_k_indices", "eq",            "lt",                    "gt",         "where",         "masked_fill",      "cat",             "stack",      "clamp",           "exp",
    "log",           "sqrt",          "sigmoid",               "silu",       "tanh",          "erf",              "gelu",            "sin",        "cos",             "softmax",
    "log_softmax",   "cross_entropy", "cross_entropy_indexed", "layer_norm", "rms_norm",      "quantized_matmul", "log_softmax_nll", "dot",        "reduce_to_shape",
};

pub fn operationName(code: ?usize) ?[]const u8 {
    const value = code orelse return null;
    if (value == 0 or value >= cyna_operation_names.len) return null;
    return cyna_operation_names[value];
}
// rhs is a value ID for binary steps, an axis for reductions and softmax,
// an index ID for indexed steps, a class count for one_hot, a DenseDType code
// for cast, or an axis for squeeze, unsqueeze, and swap_axes.
// Squeeze uses maxInt(usize) to remove all singleton axes. third is where's
// false branch, the second axis for swap_axes, an indexed axis, or updates ID.
// metadata is the target shape for reshape, axis order for permute, or
// start/stop/positive-step triples for each slice axis. Scatter add uses one
// metadata value for its axis. Top-k uses axis, k, and a 0/1 largest flag.
// Reduce-to-shape uses target dimensions; cat and stack use additional value IDs.
// Masked fill and clamp store each f64 scalar as low and high 32-bit words.
// Layer and RMS normalization use the same scalar encoding for epsilon;
// loss steps use rhs for targets and third for class axis. Quantized matmul
// uses lhs for activation, rhs for packed weights, third for scales, and
// metadata for bias value ID, bit width, and group size.
pub const DenseStep = struct { operation: DenseOperation, lhs: usize, rhs: usize = 0, third: usize = 0, metadata: []const usize = &.{} };
pub const DenseResidualRequirement = struct {
    value: usize,
    recomputable_view: bool,
    release_after: ?usize = null,
    recompute_before: ?usize = null,
};
pub const DenseResidualUse = struct { value: usize, instruction: usize, input_index: u32 };
pub const DenseResidualPolicy = enum(c_int) { retain = 0, automatic = 1, recompute = 2 };

pub fn scalarMetadata(value: f64) [2]usize {
    const bits: u64 = @bitCast(value);
    return .{ @intCast(bits & 0xffff_ffff), @intCast(bits >> 32) };
}

// Returns the number of flattened output dimensions written. Each selected
// output has one rank and element count; a scalar has rank zero and one element.
pub fn inferDenseGraph(
    allocator: std.mem.Allocator,
    dtype: DenseDType,
    inputs: []const DenseInputShape,
    steps: []const DenseStep,
    selected: []const usize,
    output_ranks: []usize,
    output_elements: []usize,
    output_dimensions: []usize,
) (Error || std.mem.Allocator.Error)!usize {
    return inferDenseGraphImpl(allocator, dtype, inputs, steps, selected, output_ranks, output_elements, output_dimensions, null);
}

pub fn inferDenseGraphTyped(
    allocator: std.mem.Allocator,
    dtype: DenseDType,
    inputs: []const DenseInputShape,
    steps: []const DenseStep,
    selected: []const usize,
    output_ranks: []usize,
    output_elements: []usize,
    output_dimensions: []usize,
    output_dtypes: []c_int,
) (Error || std.mem.Allocator.Error)!usize {
    return inferDenseGraphImpl(allocator, dtype, inputs, steps, selected, output_ranks, output_elements, output_dimensions, output_dtypes);
}

// Input and selected output buffers use little-endian dtype-width bytes.
// Use inferDenseGraphTyped to size and interpret each selected output.
pub fn runDenseGraphBytes(
    allocator: std.mem.Allocator,
    dtype: DenseDType,
    inputs: []const DenseInputShape,
    input_bytes: []const u8,
    steps: []const DenseStep,
    selected: []const usize,
    output_bytes: []u8,
    goal: DenseGoal,
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
    switch (compute_core_run_dense_graph_bytes(@backingInt(goal), @backingInt(dtype), input_bytes.ptr, input_bytes.len, ranks.ptr, ranks.len, dimensions.ptr, dimensions.len, operations.ptr, lhs_ids.ptr, rhs_ids.ptr, steps.len, selected.ptr, selected.len, output_bytes.ptr, output_bytes.len)) {
        0 => {},
        1 => return error.InvalidShape,
        else => return error.CoreFailure,
    }
}

pub fn allocDenseGraphBytes(
    allocator: std.mem.Allocator,
    dtype: DenseDType,
    inputs: []const DenseInputShape,
    input_bytes: []const u8,
    steps: []const DenseStep,
    selected: []const usize,
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
    const written = try inferDenseGraphTyped(allocator, dtype, inputs, steps, selected, ranks, elements, dimension_buffer, dtypes);
    const dimensions = try allocator.dupe(usize, dimension_buffer[0..written]);
    errdefer allocator.free(dimensions);
    var total_bytes: usize = 0;
    for (elements, dtypes) |count, code| {
        const output_dtype: DenseDType = switch (code) {
            1 => .f32,
            2 => .f64,
            3 => .i64,
            4 => .u32,
            5 => .bf16,
            else => return error.CoreFailure,
        };
        const size = std.math.mul(usize, count, denseDTypeWidth(output_dtype)) catch return error.InvalidShape;
        total_bytes = std.math.add(usize, total_bytes, size) catch return error.InvalidShape;
    }
    const bytes = try allocator.alloc(u8, total_bytes);
    errdefer allocator.free(bytes);
    try runDenseGraphBytes(allocator, dtype, inputs, input_bytes, steps, selected, bytes, goal);
    return .{ .bytes = bytes, .ranks = ranks, .elements = elements, .dimensions = dimensions, .dtypes = dtypes };
}

// The graph owns its compiled Cyna runtime. Call deinit once after all runs.
pub const DenseGraphHandle = struct {
    raw: ?*anyopaque,
    input_byte_count: usize,
    input_count: usize,
    output_count: usize,

    pub const NodeObservation = struct {
        run_id: u64,
        node_id: usize,
        schedule_position: usize,
        elapsed_ns: u64,
        timing_available: bool,
        succeeded: bool,
        materialized_view: bool,
        kernel_count: usize,
        path: ExecutionPath,
    };

    pub const ExecutionPath = enum(usize) { unknown = 0, prepared = 1, resident_view = 2, semantic = 3 };
    pub const FallbackReason = enum(usize) { no_fallback = 0, prepared_unsupported = 1, prepared_invalid_graph = 2, no_physical_selection = 3, view_path_selected = 4 };
    pub const RunState = struct { run_id: u64, succeeded: bool, path: ExecutionPath, fallback_reason: FallbackReason };
    pub const PhysicalNodeDescription = struct {
        node_id: usize,
        step_index: ?usize,
        source_operation_code: ?usize,
        action: enum { instruction, conversion },
        source_instruction_count: usize,
        is_view: bool,
        output_bytes: usize,
        implementation: []const u8,
    };
    pub const PhysicalValueDescription = struct {
        origin: enum(usize) { input = 1, constant = 2, node_output = 3 },
        semantic_origin: usize,
        alias_of: ?usize,
        logical_bytes: usize,
        producer_node: ?usize,
    };
    pub const SemanticStepDescription = struct {
        step_index: usize,
        source_operation_code: ?usize,
        source_instruction_index: usize,
        value_id: usize,
        implementation_code: usize,
    };

    pub const ResidualDecision = struct {
        value: usize,
        choice: enum(usize) { retain = 0, recompute = 1 },
        reason: enum(usize) { policy_retain = 0, latency_priority = 1, throughput_priority = 2, recomputable_view = 3, retain_only = 4, required_program_output = 5 },
        release_after: ?usize,
        recompute_before: ?usize,
    };

    pub const SelectionComparison = struct {
        evaluated_states: usize,
        baseline_peak_bytes: usize,
        candidate_peak_bytes: usize,
        candidate_published: bool,
        budget: usize,
        exclusion_search_truncated: bool,
    };

    pub const SearchEvidence = struct {
        limits: SearchLimits,
        evaluated_states: usize,
        structural_expansions: usize,
        complete_evaluations: usize,
        truncated: bool,
        ready_truncated: bool,
    };

    pub const PreparationFailureEvidence = struct {
        node_id: usize,
        phase: enum(usize) {
            persistent_scratch = 1,
            implementation = 2,
        },
        is_conversion: bool,
        step_index: ?usize,
        source_value: ?usize,
        target_layout: ?usize,
        implementation_version: usize,
        error_code: usize,
        implementation_hex: []u8,
        configuration_hex: []u8,

        pub fn deinit(self: *PreparationFailureEvidence, allocator: std.mem.Allocator) void {
            allocator.free(self.implementation_hex);
            allocator.free(self.configuration_hex);
        }
    };

    pub const PreparationEvidence = struct {
        physical_work_checked: usize,
        physical_work_truncated: bool,
        attempts: usize,
        failures: []PreparationFailureEvidence,

        pub fn deinit(self: *PreparationEvidence, allocator: std.mem.Allocator) void {
            for (self.failures) |*failure| failure.deinit(allocator);
            allocator.free(self.failures);
        }
    };

    pub const ResourceEvidence = struct {
        live_bytes: usize,
        peak_live_bytes: usize,
        live_tensors: usize,
        max_resident_bytes: ?usize,
        prepared_cache_entries: usize,
        prepared_cache_bytes: usize,
        invocation_scratch_bytes: usize,
        persistent_scratch_bytes: usize,
        cpu_pool_reserved_bytes: usize,
        cpu_pool_active_bytes: usize,
        cpu_pool_cached_bytes: usize,
        host_readback_bytes: usize,
    };

    pub fn resourceEvidence(self: *const DenseGraphHandle) Error!ResourceEvidence {
        if (self.raw == null) return error.InvalidShape;
        var words: [12]usize = undefined;
        if (compute_core_read_dense_graph_resources(self.raw, &words, words.len) != 0)
            return error.CoreFailure;
        return .{
            .live_bytes = words[0],
            .peak_live_bytes = words[1],
            .live_tensors = words[2],
            .max_resident_bytes = if (words[3] == 0) null else words[3],
            .prepared_cache_entries = words[4],
            .prepared_cache_bytes = words[5],
            .invocation_scratch_bytes = words[6],
            .persistent_scratch_bytes = words[7],
            .cpu_pool_reserved_bytes = words[8],
            .cpu_pool_active_bytes = words[9],
            .cpu_pool_cached_bytes = words[10],
            .host_readback_bytes = words[11],
        };
    }

    pub fn preparationEvidence(
        self: *const DenseGraphHandle,
        allocator: std.mem.Allocator,
    ) !PreparationEvidence {
        if (self.raw == null) return error.InvalidShape;
        var words: [4]usize = undefined;
        if (compute_core_read_dense_graph_preparation(self.raw, &words, words.len) != 0)
            return error.CoreFailure;
        if (words[1] > 1) return error.CoreFailure;
        const failures = try allocator.alloc(PreparationFailureEvidence, words[3]);
        errdefer allocator.free(failures);
        var initialized: usize = 0;
        errdefer {
            for (failures[0..initialized]) |*failure| failure.deinit(allocator);
        }
        for (failures, 0..) |*failure, index| {
            var description: [10]usize = undefined;
            var implementation_buffer: [4096]u8 = undefined;
            var configuration_buffer: [4096]u8 = undefined;
            if (compute_core_describe_dense_graph_preparation_failure(
                self.raw,
                index,
                &description,
                description.len,
                &implementation_buffer,
                implementation_buffer.len,
                &configuration_buffer,
                configuration_buffer.len,
            ) != 0) return error.CoreFailure;
            if (description[1] < 1 or description[1] > 2 or description[2] > 1 or
                description[8] > implementation_buffer.len or
                description[9] > configuration_buffer.len) return error.CoreFailure;
            failure.* = .{
                .node_id = description[0],
                .phase = @fromBackingInt(@intCast(description[1])),
                .is_conversion = description[2] == 1,
                .step_index = if (description[3] == 0) null else description[3] - 1,
                .source_value = if (description[4] == 0) null else description[4] - 1,
                .target_layout = if (description[5] == 0) null else description[5],
                .implementation_version = description[6],
                .error_code = description[7],
                .implementation_hex = try std.fmt.allocPrint(
                    allocator,
                    "{x}",
                    .{implementation_buffer[0..description[8]]},
                ),
                .configuration_hex = try std.fmt.allocPrint(
                    allocator,
                    "{x}",
                    .{configuration_buffer[0..description[9]]},
                ),
            };
            initialized += 1;
        }
        return .{
            .physical_work_checked = words[0],
            .physical_work_truncated = words[1] == 1,
            .attempts = words[2],
            .failures = failures,
        };
    }

    pub fn searchEvidence(self: *const DenseGraphHandle) Error!SearchEvidence {
        if (self.raw == null) return error.InvalidShape;
        var words: [12]usize = undefined;
        if (compute_core_read_dense_graph_search(self.raw, &words, words.len) != 0)
            return error.CoreFailure;
        if (words[8] > 1 or words[9] > 1) return error.CoreFailure;
        const unset = std.math.maxInt(usize);
        return .{
            .limits = .{
                .structural_states = words[0],
                .structural_expansions = words[1],
                .complete_evaluations = words[2],
                .physical_work = words[3],
                .ready_work = words[4],
                .max_peak_active_bytes = if (words[10] == unset) null else words[10],
                .max_invocation_scratch_bytes = if (words[11] == unset) null else words[11],
            },
            .evaluated_states = words[5],
            .structural_expansions = words[6],
            .complete_evaluations = words[7],
            .truncated = words[8] == 1,
            .ready_truncated = words[9] == 1,
        };
    }

    pub fn selectionComparison(self: *const DenseGraphHandle) Error!?SelectionComparison {
        if (self.raw == null) return error.InvalidShape;
        var words: [6]usize = undefined;
        const status = compute_core_read_dense_graph_selection(self.raw, &words, words.len);
        if (status == 1) return null;
        if (status != 0 or words[3] > 1 or words[5] > 1) return error.CoreFailure;
        return .{
            .evaluated_states = words[0],
            .baseline_peak_bytes = words[1],
            .candidate_peak_bytes = words[2],
            .candidate_published = words[3] == 1,
            .budget = words[4],
            .exclusion_search_truncated = words[5] == 1,
        };
    }

    pub fn residualDecisionCount(self: *const DenseGraphHandle) usize {
        return compute_core_dense_graph_residual_decision_count(self.raw);
    }

    pub fn describeResidualDecision(self: *const DenseGraphHandle, index: usize) Error!ResidualDecision {
        if (self.raw == null) return error.InvalidShape;
        var words: [5]usize = undefined;
        if (compute_core_describe_dense_graph_residual_decision(self.raw, index, &words, words.len) != 0)
            return error.CoreFailure;
        return .{
            .value = words[0],
            .choice = switch (words[1]) {
                0 => .retain,
                1 => .recompute,
                else => return error.CoreFailure,
            },
            .reason = switch (words[2]) {
                0 => .policy_retain,
                1 => .latency_priority,
                2 => .throughput_priority,
                3 => .recomputable_view,
                4 => .retain_only,
                5 => .required_program_output,
                else => return error.CoreFailure,
            },
            .release_after = if (words[3] == std.math.maxInt(usize)) null else words[3],
            .recompute_before = if (words[4] == std.math.maxInt(usize)) null else words[4],
        };
    }

    pub fn semanticStepCount(self: *const DenseGraphHandle) usize {
        return compute_core_dense_graph_semantic_step_count(self.raw);
    }

    pub fn describeSemanticStep(self: *const DenseGraphHandle, index: usize) Error!SemanticStepDescription {
        if (self.raw == null) return error.InvalidShape;
        var words: [5]usize = undefined;
        if (compute_core_describe_dense_graph_semantic_step(self.raw, index, &words, words.len) != 0)
            return error.CoreFailure;
        return .{
            .step_index = words[0],
            .source_operation_code = if (words[1] == 0) null else words[1],
            .source_instruction_index = words[2],
            .value_id = words[3],
            .implementation_code = words[4],
        };
    }

    pub fn physicalNodeCount(self: *const DenseGraphHandle) usize {
        return compute_core_dense_graph_physical_node_count(self.raw);
    }

    pub fn physicalValueCount(self: *const DenseGraphHandle) usize {
        return compute_core_dense_graph_physical_value_count(self.raw);
    }

    pub fn readPhysicalValues(self: *const DenseGraphHandle, output: []PhysicalValueDescription, words: []usize) Error!void {
        if (self.raw == null or output.len != self.physicalValueCount() or
            words.len != std.math.mul(usize, output.len, 5) catch return error.InvalidShape)
            return error.InvalidShape;
        if (compute_core_read_dense_graph_physical_values(self.raw, words.ptr, words.len) != 0) return error.CoreFailure;
        for (output, 0..) |*value, index| {
            const row = words[index * 5 ..][0..5];
            if (row[0] < 1 or row[0] > 3) return error.CoreFailure;
            value.* = .{
                .origin = @fromBackingInt(@intCast(row[0])),
                .semantic_origin = row[1],
                .alias_of = if (row[2] == 0) null else row[2] - 1,
                .logical_bytes = row[3],
                .producer_node = if (row[4] == 0) null else row[4] - 1,
            };
        }
    }

    pub fn readPhysicalLiveness(self: *const DenseGraphHandle, words: []usize) Error!void {
        const count = self.physicalNodeCount();
        const row_words = std.math.mul(usize, count, 4) catch return error.InvalidShape;
        const word_count = std.math.add(usize, 3, row_words) catch return error.InvalidShape;
        if (self.raw == null or words.len != word_count)
            return error.InvalidShape;
        if (compute_core_read_dense_graph_physical_liveness(self.raw, words.ptr, words.len) != 0) return error.CoreFailure;
    }

    pub fn describePhysicalNode(self: *const DenseGraphHandle, index: usize, implementation_buffer: []u8) Error!PhysicalNodeDescription {
        if (self.raw == null or implementation_buffer.len == 0) return error.InvalidShape;
        var words: [7]usize = undefined;
        var implementation_length: usize = 0;
        if (compute_core_describe_dense_graph_physical_node(self.raw, index, &words, words.len, implementation_buffer.ptr, implementation_buffer.len, &implementation_length) != 0) return error.CoreFailure;
        if (implementation_length > implementation_buffer.len) return error.CoreFailure;
        return .{
            .node_id = words[0],
            .step_index = if (words[1] == std.math.maxInt(usize)) null else words[1],
            .source_operation_code = if (words[2] == 0) null else words[2],
            .action = if (words[3] == 1) .instruction else .conversion,
            .source_instruction_count = words[4],
            .is_view = words[5] != 0,
            .output_bytes = words[6],
            .implementation = implementation_buffer[0..implementation_length],
        };
    }

    pub fn physicalNodeOrigins(self: *const DenseGraphHandle, allocator: std.mem.Allocator, index: usize, source_instruction_count: usize) ![]usize {
        if (self.raw == null) return error.InvalidShape;
        if (source_instruction_count == 0) return try allocator.alloc(usize, 0);
        const word_count = std.math.add(usize, 7, source_instruction_count) catch return error.InvalidShape;
        const words = try allocator.alloc(usize, word_count);
        defer allocator.free(words);
        var implementation_buffer: [256]u8 = undefined;
        var implementation_length: usize = 0;
        if (compute_core_describe_dense_graph_physical_node(self.raw, index, words.ptr, words.len, &implementation_buffer, implementation_buffer.len, &implementation_length) != 0 or words[0] != index or
            words[4] != source_instruction_count) return error.CoreFailure;
        return try allocator.dupe(usize, words[7..]);
    }

    pub fn runState(self: *const DenseGraphHandle) Error!RunState {
        if (self.raw == null) return error.InvalidShape;
        var words: [4]usize = undefined;
        if (compute_core_read_dense_graph_run_state(self.raw, &words, words.len) != 0)
            return error.CoreFailure;
        if (words[1] == 2) return error.InvalidCommand;
        if (words[2] > 3) return error.CoreFailure;
        if (words[3] > 4) return error.CoreFailure;
        return .{ .run_id = @intCast(words[0]), .succeeded = words[1] == 0, .path = @fromBackingInt(@intCast(words[2])), .fallback_reason = @fromBackingInt(@intCast(words[3])) };
    }

    pub fn configureObservation(self: *const DenseGraphHandle, limit: usize) Error!void {
        if (self.raw == null) return error.InvalidShape;
        if (compute_core_configure_dense_graph_observation(self.raw, limit) != 0) return error.CoreFailure;
    }

    pub fn observationCount(self: *const DenseGraphHandle, dropped: *usize) usize {
        return compute_core_dense_graph_observation_count(self.raw, dropped);
    }

    pub fn readObservation(self: *const DenseGraphHandle, index: usize) Error!NodeObservation {
        if (self.raw == null) return error.InvalidShape;
        var words: [9]usize = undefined;
        if (compute_core_read_dense_graph_observation(self.raw, index, &words, words.len) != 0)
            return error.CoreFailure;
        return .{
            .run_id = @intCast(words[0]),
            .node_id = words[1],
            .schedule_position = words[2],
            .elapsed_ns = @intCast(words[3]),
            .timing_available = words[4] != 0,
            .succeeded = words[5] != 0,
            .materialized_view = words[6] != 0,
            .kernel_count = words[8],
            .path = if (words[7] <= 3) @fromBackingInt(@intCast(words[7])) else .unknown,
        };
    }

    pub fn deinit(self: *DenseGraphHandle) void {
        if (self.raw) |raw| {
            const status = compute_core_destroy_dense_graph_handle(raw);
            std.debug.assert(status == 0);
            self.raw = null;
        }
    }

    pub fn run(self: *const DenseGraphHandle, input_bytes: []const u8, output_bytes: []u8) Error!void {
        if (self.raw == null or input_bytes.len != self.input_byte_count) return error.InvalidShape;
        switch (compute_core_run_dense_graph_handle(self.raw, input_bytes.ptr, input_bytes.len, output_bytes.ptr, output_bytes.len)) {
            0 => {},
            1 => return error.InvalidShape,
            3 => return error.InvalidCommand,
            else => return error.CoreFailure,
        }
    }

    pub const TensorToken = struct { owner: usize, slot: usize, generation: usize };
    pub const TensorLayout = struct {
        rank: usize,
        backing_bytes: usize,
        offset_bytes: usize,
    };
    pub const tensor_layout_words_per_output = 11;

    pub fn createInputTensor(self: *const DenseGraphHandle, input_index: usize, bytes: []const u8) Error!TensorToken {
        if (self.raw == null) return error.InvalidShape;
        var parts: [3]usize = undefined;
        switch (compute_core_create_dense_graph_tensor(self.raw, input_index, bytes.ptr, bytes.len, &parts)) {
            0 => return .{ .owner = parts[0], .slot = parts[1], .generation = parts[2] },
            1 => return error.InvalidShape,
            else => return error.CoreFailure,
        }
    }

    pub fn readTensor(self: *const DenseGraphHandle, token: TensorToken, output: []u8) Error!void {
        if (self.raw == null) return error.InvalidShape;
        switch (compute_core_read_dense_graph_tensor(self.raw, token.owner, token.slot, token.generation, output.ptr, output.len)) {
            0 => {},
            1 => return error.InvalidShape,
            else => return error.CoreFailure,
        }
    }

    pub fn tensorLayout(self: *const DenseGraphHandle, token: TensorToken, strides: []usize) Error!TensorLayout {
        if (self.raw == null) return error.InvalidShape;
        var rank: usize = 0;
        var backing_bytes: usize = 0;
        var offset_bytes: usize = 0;
        switch (compute_core_describe_dense_graph_tensor(self.raw, token.owner, token.slot, token.generation, &rank, &backing_bytes, &offset_bytes, strides.ptr, strides.len)) {
            0 => {},
            1 => return error.InvalidShape,
            else => return error.CoreFailure,
        }
        if (rank > strides.len) return error.InvalidShape;
        return .{
            .rank = rank,
            .backing_bytes = backing_bytes,
            .offset_bytes = offset_bytes,
        };
    }

    pub fn readTensorBacking(self: *const DenseGraphHandle, token: TensorToken, output: []u8) Error!void {
        if (self.raw == null) return error.InvalidShape;
        switch (compute_core_read_dense_graph_tensor_backing(self.raw, token.owner, token.slot, token.generation, output.ptr, output.len)) {
            0 => {},
            1 => return error.InvalidShape,
            else => return error.CoreFailure,
        }
    }

    pub fn releaseTensor(self: *const DenseGraphHandle, token: TensorToken) Error!void {
        if (self.raw == null) return error.InvalidShape;
        switch (compute_core_release_dense_graph_tensor(self.raw, token.owner, token.slot, token.generation)) {
            0 => {},
            1 => return error.InvalidShape,
            else => return error.CoreFailure,
        }
    }

    pub fn runTensors(
        self: *const DenseGraphHandle,
        allocator: std.mem.Allocator,
        inputs: []const TensorToken,
        outputs: []TensorToken,
    ) (Error || std.mem.Allocator.Error)!void {
        if (self.raw == null or inputs.len != self.input_count or
            outputs.len != self.output_count) return error.InvalidShape;
        const input_words = std.math.mul(usize, inputs.len, 3) catch return error.InvalidShape;
        const output_words = std.math.mul(usize, outputs.len, 3) catch return error.InvalidShape;
        var input_words_buffer: [24]usize = undefined;
        var output_words_buffer: [24]usize = undefined;
        const packed_inputs = if (input_words <= input_words_buffer.len)
            input_words_buffer[0..input_words]
        else
            try allocator.alloc(usize, input_words);
        defer if (input_words > input_words_buffer.len) allocator.free(packed_inputs);
        const packed_outputs = if (output_words <= output_words_buffer.len)
            output_words_buffer[0..output_words]
        else
            try allocator.alloc(usize, output_words);
        defer if (output_words > output_words_buffer.len) allocator.free(packed_outputs);
        for (inputs, 0..) |token, index| {
            packed_inputs[index * 3] = token.owner;
            packed_inputs[index * 3 + 1] = token.slot;
            packed_inputs[index * 3 + 2] = token.generation;
        }
        switch (compute_core_run_dense_graph_tensors(self.raw, packed_inputs.ptr, inputs.len, packed_outputs.ptr, outputs.len)) {
            0 => {},
            1 => return error.InvalidShape,
            3 => return error.InvalidCommand,
            else => return error.CoreFailure,
        }
        for (outputs, 0..) |*token, index| {
            token.* = .{ .owner = packed_outputs[index * 3], .slot = packed_outputs[index * 3 + 1], .generation = packed_outputs[index * 3 + 2] };
        }
    }

    pub fn describeTensors(
        self: *const DenseGraphHandle,
        tokens: []const TensorToken,
        words: []usize,
    ) Error!void {
        if (self.raw == null or words.len != std.math.mul(usize, tokens.len, tensor_layout_words_per_output) catch return error.InvalidShape)
            return error.InvalidShape;
        var packed_tokens: [24]usize = undefined;
        const packed_tokens_slice = if (tokens.len * 3 <= packed_tokens.len)
            packed_tokens[0 .. tokens.len * 3]
        else
            return error.InvalidShape;
        for (tokens, 0..) |token, index| {
            packed_tokens_slice[index * 3] = token.owner;
            packed_tokens_slice[index * 3 + 1] = token.slot;
            packed_tokens_slice[index * 3 + 2] = token.generation;
        }
        switch (compute_core_describe_dense_graph_tensors(
            self.raw,
            packed_tokens_slice.ptr,
            tokens.len,
            words.ptr,
            words.len,
        )) {
            0 => {},
            1 => return error.InvalidShape,
            else => return error.CoreFailure,
        }
    }

    pub fn readTensorsAndRelease(
        self: *const DenseGraphHandle,
        tokens: []const TensorToken,
        destinations: []const usize,
        byte_counts: []const usize,
    ) Error!void {
        if (self.raw == null or destinations.len != tokens.len or
            byte_counts.len != tokens.len)
            return error.InvalidShape;
        var packed_tokens: [24]usize = undefined;
        const packed_tokens_slice = if (tokens.len * 3 <= packed_tokens.len)
            packed_tokens[0 .. tokens.len * 3]
        else
            return error.InvalidShape;
        for (tokens, 0..) |token, index| {
            packed_tokens_slice[index * 3] = token.owner;
            packed_tokens_slice[index * 3 + 1] = token.slot;
            packed_tokens_slice[index * 3 + 2] = token.generation;
        }
        switch (compute_core_read_dense_graph_tensors_and_release(
            self.raw,
            packed_tokens_slice.ptr,
            tokens.len,
            destinations.ptr,
            byte_counts.ptr,
        )) {
            0 => {},
            1 => return error.InvalidShape,
            else => return error.CoreFailure,
        }
    }

    pub fn cacheHits(self: *const DenseGraphHandle) usize {
        return compute_core_dense_graph_handle_cache_hits(self.raw);
    }

    pub fn cacheMisses(self: *const DenseGraphHandle) usize {
        return compute_core_dense_graph_handle_cache_misses(self.raw);
    }

    pub fn cacheBytes(self: *const DenseGraphHandle) usize {
        return compute_core_dense_graph_handle_cache_bytes(self.raw);
    }
};

pub fn createDenseGraphHandle(
    allocator: std.mem.Allocator,
    inputs: []const DenseByteInput,
    steps: []const DenseStep,
    selected: []const usize,
    goal: DenseGoal,
) (Error || std.mem.Allocator.Error)!DenseGraphHandle {
    return createDenseGraphHandleWithLayouts(allocator, inputs, steps, selected, goal, false);
}

pub fn createDenseGraphHandleWithLayouts(
    allocator: std.mem.Allocator,
    inputs: []const DenseByteInput,
    steps: []const DenseStep,
    selected: []const usize,
    goal: DenseGoal,
    dense_inputs: bool,
) (Error || std.mem.Allocator.Error)!DenseGraphHandle {
    return createDenseGraphHandleWithResiduals(allocator, inputs, steps, selected, goal, dense_inputs, &.{}, &.{}, .retain);
}

pub fn createDenseGraphHandleWithResiduals(
    allocator: std.mem.Allocator,
    inputs: []const DenseByteInput,
    steps: []const DenseStep,
    selected: []const usize,
    goal: DenseGoal,
    dense_inputs: bool,
    requirements: []const DenseResidualRequirement,
    uses: []const DenseResidualUse,
    residual_policy: DenseResidualPolicy,
) (Error || std.mem.Allocator.Error)!DenseGraphHandle {
    return createDenseGraphHandleWithConstants(allocator, inputs, inputs.len, &.{}, steps, selected, goal, dense_inputs, requirements, uses, residual_policy);
}

pub fn createDenseGraphHandleWithConstantsAndSearch(
    allocator: std.mem.Allocator,
    inputs: []const DenseByteInput,
    dynamic_count: usize,
    constant_bytes: []const u8,
    steps: []const DenseStep,
    selected: []const usize,
    goal: DenseGoal,
    dense_inputs: bool,
    requirements: []const DenseResidualRequirement,
    uses: []const DenseResidualUse,
    residual_policy: DenseResidualPolicy,
    search_limits: SearchLimits,
) (Error || std.mem.Allocator.Error)!DenseGraphHandle {
    if (inputs.len == 0 or selected.len == 0 or
        dynamic_count > inputs.len or
        steps.len > std.math.maxInt(usize) - inputs.len) return error.InvalidShape;
    var dimension_count: usize = 0;
    var byte_count: usize = 0;
    var constant_byte_count: usize = 0;
    for (inputs, 0..) |input, index| {
        if (input.shape.len > 8) return error.InvalidShape;
        dimension_count = std.math.add(usize, dimension_count, input.shape.len) catch return error.InvalidShape;
        var elements: usize = 1;
        for (input.shape) |dimension| {
            if (dimension == 0) return error.InvalidShape;
            elements = std.math.mul(usize, elements, dimension) catch return error.InvalidShape;
        }
        const size = std.math.mul(usize, elements, denseDTypeWidth(input.dtype)) catch return error.InvalidShape;
        if (size != input.bytes.len) return error.InvalidShape;
        if (index < dynamic_count)
            byte_count = std.math.add(usize, byte_count, size) catch return error.InvalidShape
        else
            constant_byte_count = std.math.add(usize, constant_byte_count, size) catch return error.InvalidShape;
    }
    if (constant_byte_count != constant_bytes.len) return error.InvalidShape;
    const ranks = try allocator.alloc(usize, inputs.len);
    defer allocator.free(ranks);
    const dimensions = try allocator.alloc(usize, dimension_count);
    defer allocator.free(dimensions);
    const dtypes = try allocator.alloc(c_int, inputs.len);
    defer allocator.free(dtypes);
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
    const offsets = try allocator.alloc(usize, steps.len + 1);
    defer allocator.free(offsets);
    const metadata = try allocator.alloc(usize, metadata_count);
    defer allocator.free(metadata);
    var dimension_cursor: usize = 0;
    for (inputs, 0..) |input, index| {
        ranks[index] = input.shape.len;
        dtypes[index] = @backingInt(input.dtype);
        @memcpy(dimensions[dimension_cursor..][0..input.shape.len], input.shape);
        dimension_cursor += input.shape.len;
    }
    offsets[0] = 0;
    var metadata_cursor: usize = 0;
    for (steps, 0..) |step, index| {
        operations[index] = @backingInt(step.operation);
        lhs_ids[index] = step.lhs;
        rhs_ids[index] = step.rhs;
        third_ids[index] = step.third;
        @memcpy(metadata[metadata_cursor..][0..step.metadata.len], step.metadata);
        metadata_cursor += step.metadata.len;
        offsets[index + 1] = metadata_cursor;
    }
    const requirement_word_count = std.math.mul(usize, requirements.len, 4) catch return error.InvalidShape;
    const requirement_words = try allocator.alloc(usize, requirement_word_count);
    defer allocator.free(requirement_words);
    for (requirements, 0..) |requirement, index| {
        requirement_words[index * 4] = requirement.value;
        requirement_words[index * 4 + 1] = @intFromBool(requirement.recomputable_view);
        requirement_words[index * 4 + 2] = requirement.release_after orelse std.math.maxInt(usize);
        requirement_words[index * 4 + 3] = requirement.recompute_before orelse std.math.maxInt(usize);
    }
    const use_word_count = std.math.mul(usize, uses.len, 3) catch return error.InvalidShape;
    const use_words = try allocator.alloc(usize, use_word_count);
    defer allocator.free(use_words);
    for (uses, 0..) |use, index| {
        use_words[index * 3] = use.value;
        use_words[index * 3 + 1] = use.instruction;
        use_words[index * 3 + 2] = use.input_index;
    }
    const search_limit_words = [_]usize{
        search_limits.structural_states,
        search_limits.structural_expansions,
        search_limits.complete_evaluations,
        search_limits.physical_work,
        search_limits.ready_work,
        search_limits.max_peak_active_bytes orelse std.math.maxInt(usize),
        search_limits.max_invocation_scratch_bytes orelse std.math.maxInt(usize),
    };
    const raw = compute_core_create_dense_graph_handle_configured(
        @backingInt(goal),
        dtypes.ptr,
        byte_count,
        ranks.ptr,
        inputs.len,
        dynamic_count,
        constant_bytes.ptr,
        constant_bytes.len,
        dimensions.ptr,
        dimensions.len,
        operations.ptr,
        lhs_ids.ptr,
        rhs_ids.ptr,
        third_ids.ptr,
        offsets.ptr,
        metadata.ptr,
        metadata.len,
        steps.len,
        selected.ptr,
        selected.len,
        @intFromBool(dense_inputs),
        requirement_words.ptr,
        requirements.len,
        use_words.ptr,
        uses.len,
        @backingInt(residual_policy),
        &search_limit_words,
        search_limit_words.len,
    );
    const handle = raw orelse return error.InvalidShape;
    return .{ .raw = handle, .input_byte_count = byte_count, .input_count = dynamic_count, .output_count = selected.len };
}

pub fn createDenseGraphHandleWithConstants(
    allocator: std.mem.Allocator,
    inputs: []const DenseByteInput,
    dynamic_count: usize,
    constant_bytes: []const u8,
    steps: []const DenseStep,
    selected: []const usize,
    goal: DenseGoal,
    dense_inputs: bool,
    requirements: []const DenseResidualRequirement,
    uses: []const DenseResidualUse,
    residual_policy: DenseResidualPolicy,
) (Error || std.mem.Allocator.Error)!DenseGraphHandle {
    return createDenseGraphHandleWithConstantsAndSearch(allocator, inputs, dynamic_count, constant_bytes, steps, selected, goal, dense_inputs, requirements, uses, residual_policy, .{});
}

fn allocDenseGraphMultiBytesMode(
    resident: bool,
    allocator: std.mem.Allocator,
    inputs: []const DenseByteInput,
    steps: []const DenseStep,
    selected: []const usize,
    goal: DenseGoal,
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
    switch (compute_core_infer_dense_graph_multi_layout_typed(input_dtypes.ptr, input_ranks.ptr, inputs.len, input_dimensions.ptr, input_dimensions.len, operations.ptr, lhs_ids.ptr, rhs_ids.ptr, third_ids.ptr, metadata_offsets.ptr, metadata_values.ptr, metadata_values.len, steps.len, selected.ptr, selected.len, ranks.ptr, elements.ptr, dimension_buffer.ptr, dimension_buffer.len, dtypes.ptr)) {
        0 => {},
        1 => return error.InvalidShape,
        else => return error.CoreFailure,
    }
    var written: usize = 0;
    for (ranks) |rank| written = std.math.add(usize, written, rank) catch return error.CoreFailure;
    const dimensions = try allocator.dupe(usize, dimension_buffer[0..written]);
    errdefer allocator.free(dimensions);
    var total_bytes: usize = 0;
    for (elements, dtypes) |count, code| {
        const output_dtype: DenseDType = switch (code) {
            1 => .f32,
            2 => .f64,
            3 => .i64,
            4 => .u32,
            5 => .bf16,
            else => return error.CoreFailure,
        };
        const size = std.math.mul(usize, count, denseDTypeWidth(output_dtype)) catch return error.InvalidShape;
        total_bytes = std.math.add(usize, total_bytes, size) catch return error.InvalidShape;
    }
    const bytes = try allocator.alloc(u8, total_bytes);
    errdefer allocator.free(bytes);
    const status = if (resident) compute_core_run_dense_graph_resident_multi_layout_bytes(@backingInt(goal), input_dtypes.ptr, input_bytes.ptr, input_bytes.len, input_ranks.ptr, inputs.len, input_dimensions.ptr, input_dimensions.len, operations.ptr, lhs_ids.ptr, rhs_ids.ptr, third_ids.ptr, metadata_offsets.ptr, metadata_values.ptr, metadata_values.len, steps.len, selected.ptr, selected.len, bytes.ptr, bytes.len) else compute_core_run_dense_graph_multi_layout_bytes(@backingInt(goal), input_dtypes.ptr, input_bytes.ptr, input_bytes.len, input_ranks.ptr, inputs.len, input_dimensions.ptr, input_dimensions.len, operations.ptr, lhs_ids.ptr, rhs_ids.ptr, third_ids.ptr, metadata_offsets.ptr, metadata_values.ptr, metadata_values.len, steps.len, selected.ptr, selected.len, bytes.ptr, bytes.len);
    switch (status) {
        0 => {},
        1 => return error.InvalidShape,
        else => return error.CoreFailure,
    }
    return .{ .bytes = bytes, .ranks = ranks, .elements = elements, .dimensions = dimensions, .dtypes = dtypes };
}

pub fn allocDenseGraphMultiBytes(
    allocator: std.mem.Allocator,
    inputs: []const DenseByteInput,
    steps: []const DenseStep,
    selected: []const usize,
    goal: DenseGoal,
) (Error || std.mem.Allocator.Error)!DenseGraphBytesResult {
    return allocDenseGraphMultiBytesMode(false, allocator, inputs, steps, selected, goal);
}

pub fn inferDenseGraphMulti(
    allocator: std.mem.Allocator,
    inputs: []const DenseByteInput,
    steps: []const DenseStep,
    selected: []const usize,
) (Error || std.mem.Allocator.Error)!DenseGraphInferred {
    if (inputs.len == 0 or selected.len == 0) return error.InvalidShape;
    var dimension_count: usize = 0;
    for (inputs) |input| {
        if (input.shape.len > 8) return error.InvalidShape;
        dimension_count = std.math.add(usize, dimension_count, input.shape.len) catch
            return error.InvalidShape;
    }
    const input_ranks = try allocator.alloc(usize, inputs.len);
    defer allocator.free(input_ranks);
    const input_dimensions = try allocator.alloc(usize, dimension_count);
    defer allocator.free(input_dimensions);
    const input_dtypes = try allocator.alloc(c_int, inputs.len);
    defer allocator.free(input_dtypes);
    const operations = try allocator.alloc(c_int, steps.len);
    defer allocator.free(operations);
    const lhs_ids = try allocator.alloc(usize, steps.len);
    defer allocator.free(lhs_ids);
    const rhs_ids = try allocator.alloc(usize, steps.len);
    defer allocator.free(rhs_ids);
    const third_ids = try allocator.alloc(usize, steps.len);
    defer allocator.free(third_ids);
    var metadata_count: usize = 0;
    for (steps) |step| metadata_count = std.math.add(usize, metadata_count, step.metadata.len) catch
        return error.InvalidShape;
    const metadata_offsets = try allocator.alloc(usize, steps.len + 1);
    defer allocator.free(metadata_offsets);
    const metadata_values = try allocator.alloc(usize, metadata_count);
    defer allocator.free(metadata_values);
    var dimension_cursor: usize = 0;
    for (inputs, 0..) |input, index| {
        input_ranks[index] = input.shape.len;
        input_dtypes[index] = @backingInt(input.dtype);
        @memcpy(input_dimensions[dimension_cursor..][0..input.shape.len], input.shape);
        dimension_cursor += input.shape.len;
    }
    metadata_offsets[0] = 0;
    var metadata_cursor: usize = 0;
    for (steps, 0..) |step, index| {
        operations[index] = @backingInt(step.operation);
        lhs_ids[index] = step.lhs;
        rhs_ids[index] = step.rhs;
        third_ids[index] = step.third;
        @memcpy(metadata_values[metadata_cursor..][0..step.metadata.len], step.metadata);
        metadata_cursor += step.metadata.len;
        metadata_offsets[index + 1] = metadata_cursor;
    }
    const ranks = try allocator.alloc(usize, selected.len);
    errdefer allocator.free(ranks);
    const elements = try allocator.alloc(usize, selected.len);
    errdefer allocator.free(elements);
    const dtypes = try allocator.alloc(c_int, selected.len);
    errdefer allocator.free(dtypes);
    const max_dimensions = std.math.mul(usize, selected.len, 8) catch
        return error.InvalidShape;
    const dimension_buffer = try allocator.alloc(usize, max_dimensions);
    defer allocator.free(dimension_buffer);
    switch (compute_core_infer_dense_graph_multi_layout_typed(
        input_dtypes.ptr,
        input_ranks.ptr,
        inputs.len,
        input_dimensions.ptr,
        input_dimensions.len,
        operations.ptr,
        lhs_ids.ptr,
        rhs_ids.ptr,
        third_ids.ptr,
        metadata_offsets.ptr,
        metadata_values.ptr,
        metadata_values.len,
        steps.len,
        selected.ptr,
        selected.len,
        ranks.ptr,
        elements.ptr,
        dimension_buffer.ptr,
        dimension_buffer.len,
        dtypes.ptr,
    )) {
        0 => {},
        1 => return error.InvalidShape,
        else => return error.CoreFailure,
    }
    var written: usize = 0;
    for (ranks) |rank| written = std.math.add(usize, written, rank) catch
        return error.CoreFailure;
    const dimensions = try allocator.dupe(usize, dimension_buffer[0..written]);
    errdefer allocator.free(dimensions);
    return .{ .ranks = ranks, .elements = elements, .dimensions = dimensions, .dtypes = dtypes };
}

pub fn allocDenseGraphResidentMultiBytes(
    allocator: std.mem.Allocator,
    inputs: []const DenseByteInput,
    steps: []const DenseStep,
    selected: []const usize,
    goal: DenseGoal,
) (Error || std.mem.Allocator.Error)!DenseGraphBytesResult {
    return allocDenseGraphMultiBytesMode(true, allocator, inputs, steps, selected, goal);
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
                (@backingInt(step.operation) >= 37 and step.operation != .dot))) return error.InvalidShape;
        if (step.lhs >= available or
            (switch (step.operation) {
                .add, .sub, .mul, .div, .matmul, .dot, .eq, .lt, .gt => step.rhs >= available,
                else => false,
            })) return error.InvalidShape;
        if (dtype == .i64 and switch (step.operation) {
            .add, .sub, .mul, .div, .abs, .neg, .relu, .matmul, .dot, .sum_all, .mean_all, .sum_axis, .mean_axis, .sign, .min_all, .max_all, .min_axis, .max_axis, .argmin_all, .argmax_all, .argmin_axis, .argmax_axis, .variance_all, .std_all, .variance_axis, .std_axis, .eq, .lt, .gt => false,
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
        compute_core_infer_dense_graph_typed(@backingInt(dtype), ranks.ptr, ranks.len, dimensions.ptr, dimensions.len, operations.ptr, lhs_ids.ptr, rhs_ids.ptr, steps.len, selected.ptr, selected.len, output_ranks.ptr, output_elements.ptr, output_dimensions.ptr, output_dimensions.len, dtypes.ptr)
    else
        compute_core_infer_dense_graph(@backingInt(dtype), ranks.ptr, ranks.len, dimensions.ptr, dimensions.len, operations.ptr, lhs_ids.ptr, rhs_ids.ptr, steps.len, selected.ptr, selected.len, output_ranks.ptr, output_elements.ptr, output_dimensions.ptr, output_dimensions.len);
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

pub const Error = error{ InvalidShape, InvalidCommand, CoreFailure };

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

pub fn runDenseGraphResidentF32(
    allocator: std.mem.Allocator,
    inputs: []const DenseInput,
    steps: []const DenseStep,
    selected: []const usize,
    output: []f32,
) (Error || std.mem.Allocator.Error)!void {
    return runDenseGraphWithGoalMode(f32, allocator, inputs, steps, selected, output, .baseline, true);
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

pub fn runDenseGraphResidentI64(
    allocator: std.mem.Allocator,
    inputs: []const DenseInputI64,
    steps: []const DenseStep,
    selected: []const usize,
    output: []i64,
) (Error || std.mem.Allocator.Error)!void {
    return runDenseGraphWithGoalMode(i64, allocator, inputs, steps, selected, output, .baseline, true);
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

pub fn runDenseGraphResidentF64(
    allocator: std.mem.Allocator,
    inputs: []const DenseInputF64,
    steps: []const DenseStep,
    selected: []const usize,
    output: []f64,
) (Error || std.mem.Allocator.Error)!void {
    return runDenseGraphWithGoalMode(f64, allocator, inputs, steps, selected, output, .baseline, true);
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

pub fn runDenseGraphResidentU32(
    allocator: std.mem.Allocator,
    inputs: []const DenseInputU32,
    steps: []const DenseStep,
    selected: []const usize,
    output: []u32,
) (Error || std.mem.Allocator.Error)!void {
    return runDenseGraphWithGoalMode(u32, allocator, inputs, steps, selected, output, .baseline, true);
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

pub fn runDenseGraphResidentBf16Bits(
    allocator: std.mem.Allocator,
    inputs: []const DenseInputBf16Bits,
    steps: []const DenseStep,
    selected: []const usize,
    output_bits: []u16,
) (Error || std.mem.Allocator.Error)!void {
    return runDenseGraphWithGoalMode(u16, allocator, inputs, steps, selected, output_bits, .baseline, true);
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
    return runDenseGraphWithGoalMode(T, allocator, inputs, steps, selected, output, goal, false);
}

fn runDenseGraphWithGoalMode(
    comptime T: type,
    allocator: std.mem.Allocator,
    inputs: []const denseInput(T),
    steps: []const DenseStep,
    selected: []const usize,
    output: []T,
    goal: DenseGoal,
    resident: bool,
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
            (@backingInt(step.operation) >= 37 and step.operation != .dot))) return error.InvalidShape;
        if (T == i64 and switch (step.operation) {
            .add, .sub, .mul, .div, .abs, .neg, .relu, .matmul, .dot, .sum_all, .mean_all, .sum_axis, .mean_axis, .sign, .min_all, .max_all, .min_axis, .max_axis, .argmin_all, .argmax_all, .argmin_axis, .argmax_axis, .variance_all, .std_all, .variance_axis, .std_axis, .eq, .lt, .gt => false,
            else => true,
        }) return error.InvalidShape;
        if (T == u32 and @backingInt(step.operation) > 12 and @backingInt(step.operation) < 25) return error.InvalidShape;
        if (step.lhs >= available or
            (switch (step.operation) {
                .add, .sub, .mul, .div, .matmul, .dot, .eq, .lt, .gt => step.rhs >= available,
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
    const status = if (T == f32 and resident) compute_core_run_dense_graph_resident_f32(
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
    ) else if (T == f32) compute_core_run_dense_graph_with_goal_f32(
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
    ) else if (T == i64 and resident) compute_core_run_dense_graph_resident_i64(
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
    ) else if (T == f64 and resident) compute_core_run_dense_graph_resident_f64(
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
    ) else if (T == u32 and resident) compute_core_run_dense_graph_resident_u32(
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
    ) else if (T == u16 and resident) compute_core_run_dense_graph_resident_bf16(
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
