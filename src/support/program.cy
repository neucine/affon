import {
    compile_for_resident_with_search_limits
} from compute_core.compiler.resident;
import {
    ResidualPolicy,
    SearchLimits,
    default_search_limits
} from compute_core.compiler.executable;
import { SliceRange } from compute_core.program.operation;
import { AbstractLayout } from compute_core.backend.layout;
import {
    ComputeError,
    DType,
    Executable,
    InputRole,
    OpTag,
    OptimizationGoal,
    PreparedBackend,
    Program,
    ProgramBuilder,
    StateId,
    Tensor,
    TensorSpec,
    bf16_spec,
    builder,
    differentiate,
    f32_spec,
    f64_spec,
    i64_spec,
    u32_spec
} from compute_core.lib;

pub enum ProgramKind: i32 {
    authored = 0,
    gradient = 1,
}

pub enum ProgramNodeKind: i32 {
    argument = 0,
    parameter = 1,
    state = 2,
    constant = 3,
    operation = 4,
    gradient = 5,
}

pub enum CompileOptimizationLevel: i32 {
    disabled = 0,
    safe = 1,
}

pub enum NumericalPolicy: i32 {
    exact_only = 0,
    backend_equivalent = 1,
    approximate = 2,
}

pub enum ExplanationLevel: i32 {
    summary = 0,
    detailed = 1,
}

pub struct OperationOptions {
    pub axis: ?usize;
    pub keep_dims: bool;
    pub minimum: ?f64;
    pub maximum: ?f64;
    pub value: ?f64;
    pub epsilon: ?f64;
    pub k: ?usize;
    pub num_classes: ?usize;
    pub class_axis: usize;
    pub bits: ?usize;
    pub group_size: ?usize;
    pub axes: ?List<usize>;
    pub permutation: ?List<usize>;
    pub ranges: ?List<SliceRange>;
    pub largest: bool;
    pub sorted: bool;
}

pub struct ProgramNodeInput {
    pub id: usize;
    pub kind: ProgramNodeKind;
    pub spec: TensorSpec;
    pub op: ?OpTag;
    pub operands: List<usize>;
    pub options: OperationOptions;
    pub constant: ?f64;
    pub constant_bytes: ?List<u8>;
    pub gradient_target: ?usize;
}

pub struct ProgramInput {
    pub kind: ProgramKind;
    pub nodes: List<ProgramNodeInput>;
    pub outputs: ?List<usize>;
}

pub struct CompileOptionsInput {
    pub optimization_level: CompileOptimizationLevel;
    pub optimization_goal: OptimizationGoal;
    pub numerical_policy: NumericalPolicy;
    pub explanation_level: ExplanationLevel;
    pub dense_inputs: bool;
    pub residual_policy: ResidualPolicy;
    pub search_limits: SearchLimits;
}

pub struct OutputSpec {
    pub dtype: DType;
    pub dimensions: List<usize>;
    pub byte_count: usize;
}

pub struct CompiledGraph {
    pub executable: Executable;
    pub outputs: List<OutputSpec>;
    pub source_instruction_by_step: List<usize>;
    pub needs_seed: bool;
    pub dense_inputs: bool;
    pub optimization_goal: string;
    pub optimization_level: string;
    pub residual_policy: ResidualPolicy;
    pub observed_run_count: usize;
    pub failed_run_count: usize;
}

pub struct CompileSettings {
    pub optimization_level: CompileOptimizationLevel;
    pub optimization_goal: OptimizationGoal;
    pub dense_inputs: bool;
    pub residual_policy: ResidualPolicy;
    pub search_limits: SearchLimits;
}

fn invalid(): ComputeError {
    return ComputeError.invalid_program_graph;
}

fn require_usize(value: ?usize): usize ! ComputeError {
    if value is found {
        return found;
    }
    throw invalid();
}

fn require_number(value: ?f64): f64 ! ComputeError {
    if value is found {
        return found;
    }
    throw invalid();
}

fn require_op(value: ?OpTag): OpTag ! ComputeError {
    if value is selected {
        return selected;
    }
    throw invalid();
}

pub fn dtype_width(value: DType): usize {
    return switch value {
        .f32 => 4;
        .u32 => 4;
        .f64 => 8;
        .i64 => 8;
        .bf16 => 2;
    };
}

pub fn tensor_bytes(value: &const Tensor): List<u8> {
    var result: List<u8> = list<u8>();
    switch value.data {
        .f32(values) => {
            for item in values[..] {
                append_u32(&result, item as! u32);
            }
        }
        .f64(values) => {
            for item in values[..] {
                append_u64(&result, item as! u64);
            }
        }
        .i64(values) => {
            for item in values[..] {
                append_u64(&result, item as! u64);
            }
        }
        .u32(values) => {
            for item in values[..] {
                append_u32(&result, item);
            }
        }
        .bf16(values) => {
            for item in values[..] {
                append_u16(&result, item);
            }
        }
    }
    return result;
}

fn tensor_spec(selected_dtype: DType,
    dimensions: []const usize
): TensorSpec ! ComputeError {
    if selected_dtype == DType.f32 {
        return try f32_spec(dimensions);
    }
    if selected_dtype == DType.f64 {
        return try f64_spec(dimensions);
    }
    if selected_dtype == DType.i64 {
        return try i64_spec(dimensions);
    }
    if selected_dtype == DType.u32 {
        return try u32_spec(dimensions);
    }
    return try bf16_spec(dimensions);
}

fn append_u16(output: &List<u8>, value: u16): void {
    output.push((value & 0xff) as u8);
    output.push((value >> 8) as u8);
}

fn append_u32(output: &List<u8>, value: u32): void {
    var shift: u32 = 0;
    while shift < 32 {
        output.push(((value >> shift) & 0xff) as u8);
        shift = shift + 8;
    }
}

fn append_u64(output: &List<u8>, value: u64): void {
    var shift: u32 = 0;
    while shift < 64 {
        output.push(((value >> shift) & 0xff) as u8);
        shift = shift + 8;
    }
}

fn round_shift(value: u64, shift: u32): u64 {
    if shift == 0 {
        return value;
    }
    if shift >= 64 {
        return 0;
    }
    var result = value >> shift;
    const mask = ((1 as u64) << shift) - 1;
    const remainder = value & mask;
    const halfway = (1 as u64) << (shift - 1);
    if remainder > halfway or
    (remainder == halfway and (result & 1) != 0) {
        result = result + 1;
    }
    return result;
}

fn f32_bits(value: f64): u32 {
    const bits = value as! u64;
    const sign = ((bits >> 32) & 0x80000000) as u32;
    const exponent = ((bits >> 52) & 0x7ff) as i64;
    const fraction = bits & 0x000fffffffffffff;
    if exponent == 0x7ff {
        if fraction == 0 {
            return sign | 0x7f800000;
        }
        return sign | 0x7fc00000;
    }
    if exponent == 0 {
        return sign;
    }
    const significand = fraction | 0x0010000000000000;
    var unbiased = exponent - 1023;
    if unbiased > 127 {
        return sign | 0x7f800000;
    }
    if unbiased >= -126 {
        var rounded = round_shift(significand, 29);
        if rounded == 0x01000000 {
            rounded = 0x00800000;
            unbiased = unbiased + 1;
            if unbiased > 127 {
                return sign | 0x7f800000;
            }
        }
        const selected_exponent = ((unbiased + 127) as u32) << 23;
        return sign | selected_exponent |
        (rounded as u32 & 0x007fffff);
    }
    const subnormal_shift = 0 - (unbiased + 97);
    if subnormal_shift < 0 or subnormal_shift >= 64 {
        return sign;
    }
    return sign | (round_shift(
        significand,
        subnormal_shift as u32
    ) as u32);
}

fn append_number(output: &List<u8>, selected_dtype: DType,
    value: f64
): void {
    switch selected_dtype {
        .f32 => {
            append_u32(output, f32_bits(value));
        }
        .f64 => append_u64(output, value as! u64);
        .i64 => {
            const narrowed = value as i64;
            append_u64(output, narrowed as! u64);
        }
        .u32 => {
            const narrowed = value as u32;
            append_u32(output, narrowed);
        }
        .bf16 => {
            append_u16(output, (f32_bits(value) >> 16) as u16);
        }
    }
}

fn owned_spec(node: &const ProgramNodeInput): TensorSpec {
    return TensorSpec {
        dtype: node.spec.dtype,
        dimensions: node.spec.dimensions.clone(),
        axis_names: none
    };
}

fn add_constant(program_builder: &ProgramBuilder,
    node: &const ProgramNodeInput, spec: TensorSpec
): usize ! ComputeError {
    var bytes: List<u8> = list<u8>();
    const count = try node.spec.element_count();
    if node.constant_bytes is values {
        if values.len() != count * dtype_width(node.spec.dtype) {
            throw invalid();
        }
        for value in values[..] {
            bytes.push(value);
        }
    } else if node.constant is scalar {
        var index: usize = 0;
        while index < count {
            append_number(&bytes, node.spec.dtype, scalar);
            index = index + 1;
        }
    } else {
        throw invalid();
    }
    if bytes.len() != count * dtype_width(node.spec.dtype) {
        throw invalid();
    }
    return try program_builder.add_bytes_constant(
        spec,
        bytes.view()
    );
}

fn operand(node: &const ProgramNodeInput, index: usize): usize ! ComputeError {
    if index >= node.operands.len() {
        throw invalid();
    }
    return node.operands[index];
}

fn mapped_id(mapped: []const usize, source: usize): usize ! ComputeError {
    if source >= mapped.len() or mapped[source] == (0 as usize) -% 1 {
        throw invalid();
    }
    return mapped[source];
}

fn add_operation(builder: &ProgramBuilder, node: &const ProgramNodeInput,
    mapped: []const usize
): usize ! ComputeError {
    const tag = try require_op(node.op);
    const options = &node.options;
    if tag == OpTag.add or tag == OpTag.sub or tag == OpTag.mul or
    tag == OpTag.div or tag == OpTag.eq or tag == OpTag.lt or
    tag == OpTag.gt or tag == OpTag.matmul or tag == OpTag.dot {
        return try builder.add(
            tag,
            try mapped_id(mapped, try operand(node, 0)),
            try mapped_id(mapped, try operand(node, 1))
        );
    }
    if tag == OpTag.abs or tag == OpTag.neg or tag == OpTag.sign or
    tag == OpTag.relu or tag == OpTag.exp or tag == OpTag.log or
    tag == OpTag.sqrt or tag == OpTag.sigmoid or tag == OpTag.silu or
    tag == OpTag.tanh or tag == OpTag.erf or tag == OpTag.gelu or
    tag == OpTag.sin or tag == OpTag.cos {
        return try builder.add_unary(
            tag,
            try mapped_id(mapped, try operand(node, 0))
        );
    }
    if tag == OpTag.sum_all or tag == OpTag.mean_all or
    tag == OpTag.min_all or tag == OpTag.max_all or
    tag == OpTag.variance_all or tag == OpTag.std_all or
    tag == OpTag.argmin_all or tag == OpTag.argmax_all {
        return try builder.add_reduce_all(
            tag,
            try mapped_id(mapped, try operand(node, 0)),
            options.keep_dims
        );
    }
    if tag == OpTag.sum_axis or tag == OpTag.mean_axis or
    tag == OpTag.min_axis or tag == OpTag.max_axis or
    tag == OpTag.variance_axis or tag == OpTag.std_axis or
    tag == OpTag.argmin_axis or tag == OpTag.argmax_axis {
        return try builder.add_reduce_axis(
            tag,
            try mapped_id(mapped, try operand(node, 0)),
            try require_usize(options.axis),
            options.keep_dims
        );
    }
    if tag == OpTag.softmax or tag == OpTag.log_softmax {
        return try builder.add_softmax(
            tag,
            try mapped_id(mapped, try operand(node, 0)),
            try require_usize(options.axis)
        );
    }
    if tag == OpTag.reshape {
        return try builder.add_reshape(
            try mapped_id(mapped, try operand(node, 0)),
            node.spec.dimensions[..]
        );
    }
    if tag == OpTag.reduce_to_shape {
        return try builder.add_reduce_to_shape(
            try mapped_id(mapped, try operand(node, 0)),
            node.spec.dimensions[..]
        );
    }
    if tag == OpTag.permute {
        if options.axes is axes {
            return try builder.add_permute(
                try mapped_id(mapped, try operand(node, 0)),
                axes[..]
            );
        }
        throw invalid();
    }
    if tag == OpTag.transpose {
        if options.permutation is axes {
            return try builder.add_transpose_permutation(
                try mapped_id(mapped, try operand(node, 0)),
                axes[..]
            );
        }
        return try builder.add_transpose(
            try mapped_id(mapped, try operand(node, 0))
        );
    }
    if tag == OpTag.squeeze {
        return try builder.add_squeeze(
            try mapped_id(mapped, try operand(node, 0)),
            options.axis
        );
    }
    if tag == OpTag.unsqueeze {
        return try builder.add_unsqueeze(
            try mapped_id(mapped, try operand(node, 0)),
            try require_usize(options.axis)
        );
    }
    if tag == OpTag.contiguous {
        return try builder.add_contiguous(
            try mapped_id(mapped, try operand(node, 0))
        );
    }
    if tag == OpTag.slice {
        if options.ranges is ranges {
            return try builder.add_slice(
                try mapped_id(mapped, try operand(node, 0)),
                ranges[..]
            );
        }
        throw invalid();
    }
    if tag == OpTag.cast {
        return try builder.add_cast(
            try mapped_id(mapped, try operand(node, 0)),
            node.spec.dtype
        );
    }
    if tag == OpTag.clamp {
        return try builder.add_clamp(
            try mapped_id(mapped, try operand(node, 0)),
            try require_number(options.minimum),
            try require_number(options.maximum)
        );
    }
    if tag == OpTag.where {
        return try builder.add_where(
            try mapped_id(mapped, try operand(node, 0)),
            try mapped_id(mapped, try operand(node, 1)),
            try mapped_id(mapped, try operand(node, 2))
        );
    }
    if tag == OpTag.masked_fill {
        return try builder.add_masked_fill(
            try mapped_id(mapped, try operand(node, 0)),
            try mapped_id(mapped, try operand(node, 1)),
            try require_number(options.value)
        );
    }
    if tag == OpTag.cat or tag == OpTag.stack {
        var values: List<usize> = list<usize>();
        for source in node.operands[..] {
            values.push(try mapped_id(mapped, source));
        }
        const axis = try require_usize(options.axis);
        if tag == OpTag.cat {
            return try builder.add_cat(values.view(), axis);
        }
        return try builder.add_stack(values.view(), axis);
    }
    if tag == OpTag.layer_norm or tag == OpTag.rms_norm {
        return try builder.add_normalization(
            tag,
            try mapped_id(mapped, try operand(node, 0)),
            try require_usize(options.axis),
            try require_number(options.epsilon)
        );
    }
    if tag == OpTag.embedding {
        return try builder.add_embedding(
            try mapped_id(mapped, try operand(node, 0)),
            try mapped_id(mapped, try operand(node, 1))
        );
    }
    if tag == OpTag.index_select or tag == OpTag.gather {
        const lhs = try mapped_id(mapped, try operand(node, 0));
        const rhs = try mapped_id(mapped, try operand(node, 1));
        const axis = try require_usize(options.axis);
        if tag == OpTag.index_select {
            return try builder.add_index_select(lhs, rhs, axis);
        }
        return try builder.add_gather(lhs, rhs, axis);
    }
    if tag == OpTag.scatter_add {
        return try builder.add_scatter_add(
            try mapped_id(mapped, try operand(node, 0)),
            try mapped_id(mapped, try operand(node, 1)),
            try mapped_id(mapped, try operand(node, 2)),
            try require_usize(options.axis)
        );
    }
    if tag == OpTag.top_k_values or tag == OpTag.top_k_indices {
        const pair = try builder.add_top_k_with_sort(
            try mapped_id(mapped, try operand(node, 0)),
            try require_usize(options.axis),
            try require_usize(options.k),
            options.largest,
            options.sorted
        );
        if tag == OpTag.top_k_indices {
            return pair.indices;
        }
        return pair.values;
    }
    if tag == OpTag.one_hot {
        return try builder.add_one_hot(
            try mapped_id(mapped, try operand(node, 0)),
            try require_usize(options.num_classes)
        );
    }
    if tag == OpTag.cross_entropy or tag == OpTag.cross_entropy_indexed or
    tag == OpTag.log_softmax_nll {
        return try builder.add_cross_entropy(
            tag,
            try mapped_id(mapped, try operand(node, 0)),
            try mapped_id(mapped, try operand(node, 1)),
            options.class_axis
        );
    }
    if tag == OpTag.quantized_matmul {
        return try builder.add_quantized_matmul(
            try mapped_id(mapped, try operand(node, 0)),
            try mapped_id(mapped, try operand(node, 1)),
            try mapped_id(mapped, try operand(node, 2)),
            try mapped_id(mapped, try operand(node, 3)),
            try require_usize(options.bits),
            try require_usize(options.group_size)
        );
    }
    throw invalid();
}

fn build_program(options: &const ProgramInput,
    source_instruction_by_step: &List<usize>, mapped: &List<usize>,
    selected_outputs: ?[]const usize
): Program ! ComputeError {
    if mapped.len() != options.nodes.len() {
        throw invalid();
    }
    var program_builder = builder();
    var state_sequence: u64 = 0;
    for node in options.nodes[..] {
        const id = node.id;
        if id >= mapped.len() {
            throw invalid();
        }
        switch node.kind {
            .gradient => {}
            .argument => {
                mapped[id] = program_builder.add_input(owned_spec(&node));
            }
            .parameter => {
                mapped[id] = try program_builder.add_state_input(
                    owned_spec(&node),
                    InputRole.parameter,
                    StateId.from_int(state_sequence)
                );
                state_sequence = state_sequence + 1;
            }
            .state => {
                mapped[id] = try program_builder.add_state_input(
                    owned_spec(&node),
                    InputRole.model_state,
                    StateId.from_int(state_sequence)
                );
                state_sequence = state_sequence + 1;
            }
            .constant => {
                mapped[id] = try add_constant(
                    &program_builder,
                    &node,
                    owned_spec(&node)
                );
            }
            .operation => {
                const instruction = program_builder.instruction_count();
                mapped[id] = try add_operation(
                    &program_builder,
                    &node,
                    mapped.view()
                );
                while source_instruction_by_step.len() <
                program_builder.instruction_count() {
                    source_instruction_by_step.push(instruction);
                }
            }
        }
    }
    var selected: List<usize> = list<usize>();
    if selected_outputs is outputs {
        for output in outputs[..] {
            selected.push(try mapped_id(mapped.view(), output));
        }
    } else if options.outputs is outputs {
        for output in outputs[..] {
            selected.push(try mapped_id(mapped.view(), output));
        }
    } else {
        throw invalid();
    }
    return try program_builder.finish(selected.view());
}

fn gradient_count(options: &const ProgramInput): usize {
    var count: usize = 0;
    for node in options.nodes[..] {
        if node.kind == ProgramNodeKind.gradient {
            count = count + 1;
        }
    }
    return count;
}

fn gradient_loss_id(options: &const ProgramInput): usize ! ComputeError {
    for node in options.nodes[..] {
        if node.kind == ProgramNodeKind.gradient {
            return try operand(&node, 0);
        }
    }
    throw invalid();
}

fn gradient_target_id(node: &const ProgramNodeInput): usize ! ComputeError {
    return try require_usize(node.gradient_target);
}

fn compile_settings(options: &const CompileOptionsInput): CompileSettings {
    return CompileSettings {
        optimization_level: options.optimization_level,
        optimization_goal: options.optimization_goal,
        dense_inputs: options.dense_inputs,
        residual_policy: options.residual_policy,
        search_limits: SearchLimits {
            structural_states: options.search_limits.structural_states,
            structural_expansions:
            options.search_limits.structural_expansions,
            complete_evaluations:
            options.search_limits.complete_evaluations,
            physical_work: options.search_limits.physical_work,
            ready_work: options.search_limits.ready_work,
            max_peak_active_bytes:
            options.search_limits.max_peak_active_bytes,
            max_invocation_scratch_bytes:
            options.search_limits.max_invocation_scratch_bytes
        }
    };
}

fn goal_name(value: OptimizationGoal): string {
    return switch value {
        .baseline => "none";
        .memory => "memory";
        .balanced => "balanced";
        .latency => "latency";
        .throughput => "throughput";
    };
}

fn level_name(value: CompileOptimizationLevel): string {
    return switch value {
        .disabled => "none";
        .safe => "safe";
    };
}

pub fn compile_program(options: &const ProgramInput,
    compile_options: &const CompileOptionsInput,
    backend: &const PreparedBackend
): CompiledGraph ! ComputeError {
    var source_instruction_by_step: List<usize> = list<usize>();
    var mapped: List<usize> = list<usize>();
    var node_index: usize = 0;
    while node_index < options.nodes.len() {
        mapped.push((0 as usize) -% 1);
        node_index = node_index + 1;
    }
    var needs_seed = false;
    var loss_id: usize = 0;
    var loss_outputs: [1]usize = [0];
    var selected_outputs: ?[]const usize = none;
    if options.kind == ProgramKind.gradient {
        loss_id = try gradient_loss_id(options);
        loss_outputs[0] = loss_id;
        const frozen_loss_outputs: []const usize = loss_outputs[..];
        selected_outputs = frozen_loss_outputs;
    }
    var program = try build_program(
        options,
        &source_instruction_by_step,
        &mapped,
        selected_outputs
    );
    if options.kind == ProgramKind.gradient {
        const derivative_count = gradient_count(options);
        if derivative_count == 0 {
            throw invalid();
        }
        var targets: List<usize> = list<usize>();
        for node in options.nodes[..] {
            if node.kind == ProgramNodeKind.gradient {
                const source = try gradient_target_id(&node);
                targets.push(try mapped_id(mapped.view(), source));
            }
        }
        var derivative = try differentiate(
            &program,
            try mapped_id(mapped.view(), loss_id),
            targets[..]
        );
        program = derivative.into_program();
        needs_seed = true;
    }
    while source_instruction_by_step.len() < program.instructions.len() {
        source_instruction_by_step.push(source_instruction_by_step.len());
    }
    const settings = compile_settings(compile_options);
    var layouts: List<AbstractLayout> = list<AbstractLayout>();
    if settings.dense_inputs {
        var input_index: usize = 0;
        while input_index < program.inputs.len() {
            layouts.push(AbstractLayout.dense);
            input_index = input_index + 1;
        }
    }
    var selected_layouts: ?[]const AbstractLayout = none;
    const frozen_layouts = layouts;
    if settings.dense_inputs {
        selected_layouts = frozen_layouts[..];
    }
    var selected_goal = settings.optimization_goal;
    if settings.optimization_level == CompileOptimizationLevel.disabled {
        selected_goal = OptimizationGoal.baseline;
    }
    const executable = try compile_for_resident_with_search_limits(
        program,
        selected_goal,
        backend,
        selected_layouts,
        settings.residual_policy,
        settings.search_limits
    );
    if settings.search_limits.max_peak_active_bytes is maximum {
        if executable.schedule.peak_active_bytes > maximum {
            throw ComputeError.unsupported_resource;
        }
    }
    var outputs: List<OutputSpec> = list<OutputSpec>();
    for binding in executable.graph.output_bindings[..] {
        const value = &executable.graph.values[binding];
        const count = try value.spec.element_count();
        outputs.push(OutputSpec {
            dtype: value.spec.dtype,
            dimensions: value.spec.dimensions.clone(),
            byte_count: count * dtype_width(value.spec.dtype),
        });
    }
    return CompiledGraph {
        executable: executable,
        outputs: outputs,
        source_instruction_by_step: source_instruction_by_step,
        needs_seed: needs_seed,
        dense_inputs: settings.dense_inputs,
        optimization_goal: goal_name(settings.optimization_goal),
        optimization_level: level_name(settings.optimization_level),
        residual_policy: settings.residual_policy,
        observed_run_count: 0,
        failed_run_count: 0,
    };
}
