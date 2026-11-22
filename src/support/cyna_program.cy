import { parse } from std.json;
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
import { CpuContext } from compute_core.backend.cpu.storage;
import {
    ComputeError,
    DType,
    Executable,
    InputRole,
    OpTag,
    OptimizationGoal,
    Program,
    ProgramBuilder,
    SessionConfig,
    StateId,
    Tensor,
    TensorSpec,
    bf16_spec,
    builder,
    close_cpu_context,
    cpu_backend,
    cpu_context,
    differentiate,
    f32_spec,
    f64_spec,
    i64_spec,
    u32_spec
} from compute_core.lib;
import .support.json_value as json;

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
    pub optimization_level: string;
    pub optimization_goal: string;
    pub dense_inputs: bool;
    pub residual_policy: ResidualPolicy;
    pub search_limits: SearchLimits;
}

fn invalid(): ComputeError {
    return ComputeError.invalid_program_graph;
}

fn require_string(value: ?string): string ! ComputeError {
    if value is text {
        return text;
    }
    throw invalid();
}

fn require_usize(value: ?usize): usize ! ComputeError {
    if value as? usize is found {
        return found;
    }
    throw invalid();
}

fn require_number(value: ?f64): f64 ! ComputeError {
    if value as? f64 is found {
        return found;
    }
    throw invalid();
}

fn dtype(text: string): ?DType {
    if text == "f32" {
        return DType.f32;
    }
    if text == "f64" {
        return DType.f64;
    }
    if text == "i64" {
        return DType.i64;
    }
    if text == "u32" {
        return DType.u32;
    }
    if text == "bf16" {
        return DType.bf16;
    }
    return none;
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

fn dimensions(value: &const json.Value): List<usize> !ComputeError {
    var result: List<usize> = list<usize>();
    if value is .array(items) {
        for item in items.items[..] {
            if item is .number(number) {
                const dimension = try require_usize(json.parse_usize(number.spelling));
                result.push(dimension);
            } else {
                throw invalid();
            }
        }
        return result;
    }
    throw invalid();
}

fn tensor_spec(value: &const json.Value): TensorSpec ! ComputeError {
    if value is .object(object) {
        const dtype_text = try require_string(json.string_field(&object, "dtype"));
        if dtype(dtype_text) is selected_dtype {
            const shape_index = try require_usize(json.field_index(&object, "shape"));
            const shape = try dimensions(&object.members[shape_index].value);
            if selected_dtype == DType.f32 {
                return try f32_spec(shape[..]);
            }
            if selected_dtype == DType.f64 {
                return try f64_spec(shape[..]);
            }
            if selected_dtype == DType.i64 {
                return try i64_spec(shape[..]);
            }
            if selected_dtype == DType.u32 {
                return try u32_spec(shape[..]);
            }
            return try bf16_spec(shape[..]);
        }
        throw invalid();
    }
    throw invalid();
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

fn append_constant(output: &List<u8>, selected_dtype: DType,
    value: &const json.Value
): void ! ComputeError {
    switch value {
        .number(number) => {
            append_number(output, selected_dtype,
                try require_number(json.parse_number(number.spelling))
            );
        }
        .array(items) => {
            for item in items.items[..] {
                try append_constant(output, selected_dtype, &item);
            }
        }
        _ => {
            throw invalid();
        }
    }
}

fn add_constant(program_builder: &ProgramBuilder,
    spec: TensorSpec, value: &const json.Value
): usize ! ComputeError {
    var bytes: List<u8> = list<u8>();
    const count = try spec.element_count();
    var repeated = false;
    if value is .number(number) {
        if count != 1 {
            const selected = try require_number(
                json.parse_number(number.spelling)
            );
            var index: usize = 0;
            while index < count {
                append_number(&bytes, spec.dtype, selected);
                index = index + 1;
            }
            repeated = true;
        }
    }
    if not repeated {
        try append_constant(&bytes, spec.dtype, value);
    }
    if bytes.len() != count * dtype_width(spec.dtype) {
        throw invalid();
    }
    return try program_builder.add_bytes_constant(spec, bytes.view());
}

fn operand(node: &const json.Object, index: usize): usize ! ComputeError {
    const operands_index = try require_usize(json.field_index(node, "operands"));
    if node.members[operands_index].value is .array(items) {
        if index >= items.items.len() {
            throw invalid();
        }
        if items.items[index] is .number(number) {
            return try require_usize(json.parse_usize(number.spelling));
        }
    }
    throw invalid();
}

fn operand_ids(node: &const json.Object): List<usize> !ComputeError {
    var result: List<usize> = list<usize>();
    const operands_index = try require_usize(json.field_index(node, "operands"));
    if node.members[operands_index].value is .array(items) {
        for item in items.items[..] {
            if item is .number(number) {
                result.push(try require_usize(json.parse_usize(number.spelling)));
            } else {
                throw invalid();
            }
        }
        return result;
    }
    throw invalid();
}

fn mapped_id(mapped: []const usize, source: usize): usize ! ComputeError {
    if source >= mapped.len() or mapped[source] == (0 as usize) -% 1 {
        throw invalid();
    }
    return mapped[source];
}

fn option_usize(node: &const json.Object, key: string,
    fallback: ?usize
): ?usize {
    const options_index = json.field_index(node, "options");
    if options_index is index {
        if node.members[index].value is .object(options) {
            if json.usize_field(&options, key) is value {
                return value;
            }
        }
    }
    return fallback;
}

fn option_bool(node: &const json.Object, key: string, fallback: bool): bool {
    const options_index = json.field_index(node, "options");
    if options_index is index {
        if node.members[index].value is .object(options) {
            return json.bool_field(&options, key, fallback);
        }
    }
    return fallback;
}

fn option_number(node: &const json.Object, key: string,
    fallback: ?f64
): ?f64 {
    const options_index = json.field_index(node, "options");
    if options_index is index {
        if node.members[index].value is .object(options) {
            if json.number_field(&options, key) is value {
                return value;
            }
        }
    }
    return fallback;
}

fn option_usizes(node: &const json.Object, key: string): ?List<usize> {
    if json.field_index(node, "options") is options_index {
        if node.members[options_index].value is .object(options) {
            if json.field_index(&options, key) is field_index {
                if options.members[field_index].value is .array(items) {
                    var result: List<usize> = list<usize>();
                    for item in items.items[..] {
                        if item is .number(number) {
                            const maybe_value = json.parse_usize(number.spelling);
                            if maybe_value is value {
                                result.push(value);
                            } else {
                                return none;
                            }
                        } else {
                            return none;
                        }
                    }
                    return result;
                }
            }
        }
    }
    return none;
}

fn option_string(node: &const json.Object, key: string): ?string {
    if json.field_index(node, "options") is options_index {
        if node.members[options_index].value is .object(options) {
            return json.string_field(&options, key);
        }
    }
    return none;
}

fn option_slice_ranges(node: &const json.Object): ?List<SliceRange> {
    if json.field_index(node, "options") is options_index {
        if node.members[options_index].value is .object(options) {
            if json.field_index(&options, "ranges") is ranges_index {
                if options.members[ranges_index].value is .array(items) {
                    var result: List<SliceRange> = list<SliceRange>();
                    for item in items.items[..] {
                        if item is .object(range) {
                            const start = json.usize_field(&range, "start");
                            const stop = json.usize_field(&range, "stop");
                            if start is selected_start {
                                if stop is selected_stop {
                                    var step: isize = 1;
                                    if json.usize_field(
                                        &range,
                                        "step"
                                    ) is selected_step {
                                        step = selected_step as isize;
                                    }
                                    result.push(SliceRange {
                                        start: selected_start,
                                        stop: selected_stop,
                                        step: step
                                    });
                                } else {
                                    return none;
                                }
                            } else {
                                return none;
                            }
                        } else {
                            return none;
                        }
                    }
                    return result;
                }
            }
        }
    }
    return none;
}

fn binary_tag(operation: string): ?OpTag {
    if operation == "add" {
        return OpTag.add;
    }
    if operation == "sub" {
        return OpTag.sub;
    }
    if operation == "mul" {
        return OpTag.mul;
    }
    if operation == "div" {
        return OpTag.div;
    }
    if operation == "eq" {
        return OpTag.eq;
    }
    if operation == "lt" {
        return OpTag.lt;
    }
    if operation == "gt" {
        return OpTag.gt;
    }
    if operation == "matmul" {
        return OpTag.matmul;
    }
    if operation == "dot" {
        return OpTag.dot;
    }
    return none;
}

fn unary_tag(operation: string): ?OpTag {
    if operation == "abs" {
        return OpTag.abs;
    }
    if operation == "neg" {
        return OpTag.neg;
    }
    if operation == "sign" {
        return OpTag.sign;
    }
    if operation == "relu" {
        return OpTag.relu;
    }
    if operation == "exp" {
        return OpTag.exp;
    }
    if operation == "log" {
        return OpTag.log;
    }
    if operation == "sqrt" {
        return OpTag.sqrt;
    }
    if operation == "sigmoid" {
        return OpTag.sigmoid;
    }
    if operation == "silu" {
        return OpTag.silu;
    }
    if operation == "tanh" {
        return OpTag.tanh;
    }
    if operation == "erf" {
        return OpTag.erf;
    }
    if operation == "gelu" {
        return OpTag.gelu;
    }
    if operation == "sin" {
        return OpTag.sin;
    }
    if operation == "cos" {
        return OpTag.cos;
    }
    return none;
}

fn all_reduction_tag(operation: string): ?OpTag {
    if operation == "sum" {
        return OpTag.sum_all;
    }
    if operation == "mean" {
        return OpTag.mean_all;
    }
    if operation == "min" {
        return OpTag.min_all;
    }
    if operation == "max" {
        return OpTag.max_all;
    }
    if operation == "variance" {
        return OpTag.variance_all;
    }
    if operation == "std" {
        return OpTag.std_all;
    }
    if operation == "argmin" {
        return OpTag.argmin_all;
    }
    if operation == "argmax" {
        return OpTag.argmax_all;
    }
    return none;
}

fn axis_reduction_tag(operation: string): ?OpTag {
    if operation == "sum" {
        return OpTag.sum_axis;
    }
    if operation == "mean" {
        return OpTag.mean_axis;
    }
    if operation == "min" {
        return OpTag.min_axis;
    }
    if operation == "max" {
        return OpTag.max_axis;
    }
    if operation == "variance" {
        return OpTag.variance_axis;
    }
    if operation == "std" {
        return OpTag.std_axis;
    }
    if operation == "argmin" {
        return OpTag.argmin_axis;
    }
    if operation == "argmax" {
        return OpTag.argmax_axis;
    }
    return none;
}

fn add_operation(builder: &ProgramBuilder, node: &const json.Object,
    mapped: []const usize
): usize ! ComputeError {
    const operation = try require_string(json.string_field(node, "op"));
    if binary_tag(operation) is tag {
        return try builder.add(tag,
            try mapped_id(mapped, try operand(node, 0)),
            try mapped_id(mapped, try operand(node, 1))
        );
    }
    if unary_tag(operation) is tag {
        return try builder.add_unary(tag,
            try mapped_id(mapped, try operand(node, 0))
        );
    }
    if all_reduction_tag(operation) is tag {
        if option_usize(node, "axis", none) is none {
            return try builder.add_reduce_all(tag,
                try mapped_id(mapped, try operand(node, 0)),
                option_bool(node, "keep_dims", false)
            );
        }
    }
    if axis_reduction_tag(operation) is tag {
        return try builder.add_reduce_axis(tag,
            try mapped_id(mapped, try operand(node, 0)),
            try require_usize(option_usize(node, "axis", none)),
            option_bool(node, "keep_dims", false)
        );
    }
    if operation == "softmax" or operation == "log_softmax" {
        var tag = OpTag.log_softmax;
        if operation == "softmax" {
            tag = OpTag.softmax;
        }
        return try builder.add_softmax(tag,
            try mapped_id(mapped, try operand(node, 0)),
            try require_usize(option_usize(node, "axis", none))
        );
    }
    if operation == "reshape" or operation == "reduce_to_shape" {
        const spec_index = try require_usize(json.field_index(node, "spec"));
        if node.members[spec_index].value is .object(spec) {
            const shape_index = try require_usize(json.field_index(&spec, "shape"));
            const target = try dimensions(&spec.members[shape_index].value);
            if operation == "reshape" {
                return try builder.add_reshape(
                    try mapped_id(mapped, try operand(node, 0)), target[..]
                );
            }
            return try builder.add_reduce_to_shape(
                try mapped_id(mapped, try operand(node, 0)), target[..]
            );
        }
        throw invalid();
    }
    if operation == "permute" {
        if option_usizes(node, "axes") is axes {
            return try builder.add_permute(
                try mapped_id(mapped, try operand(node, 0)), axes[..]
            );
        }
        throw invalid();
    }
    if operation == "transpose" {
        if option_usizes(node, "permutation") is axes {
            return try builder.add_transpose_permutation(
                try mapped_id(mapped, try operand(node, 0)), axes[..]
            );
        }
        return try builder.add_transpose(
            try mapped_id(mapped, try operand(node, 0))
        );
    }
    if operation == "squeeze" {
        return try builder.add_squeeze(
            try mapped_id(mapped, try operand(node, 0)),
            option_usize(node, "axis", none)
        );
    }
    if operation == "unsqueeze" {
        return try builder.add_unsqueeze(
            try mapped_id(mapped, try operand(node, 0)),
            try require_usize(option_usize(node, "axis", none))
        );
    }
    if operation == "contiguous" {
        return try builder.add_contiguous(
            try mapped_id(mapped, try operand(node, 0))
        );
    }
    if operation == "slice" {
        if option_slice_ranges(node) is ranges {
            return try builder.add_slice(
                try mapped_id(mapped, try operand(node, 0)),
                ranges[..]
            );
        }
        throw invalid();
    }
    if operation == "cast" {
        const spec_index = try require_usize(json.field_index(node, "spec"));
        if node.members[spec_index].value is .object(spec) {
            const target = try require_string(json.string_field(&spec, "dtype"));
            if dtype(target) is target_dtype {
                return try builder.add_cast(
                    try mapped_id(mapped, try operand(node, 0)),
                    target_dtype
                );
            }
        }
        throw invalid();
    }
    if operation == "clamp" {
        return try builder.add_clamp(
            try mapped_id(mapped, try operand(node, 0)),
            try require_number(option_number(node, "min", none)),
            try require_number(option_number(node, "max", none))
        );
    }
    if operation == "where" {
        return try builder.add_where(
            try mapped_id(mapped, try operand(node, 0)),
            try mapped_id(mapped, try operand(node, 1)),
            try mapped_id(mapped, try operand(node, 2))
        );
    }
    if operation == "masked_fill" {
        return try builder.add_masked_fill(
            try mapped_id(mapped, try operand(node, 0)),
            try mapped_id(mapped, try operand(node, 1)),
            try require_number(option_number(node, "value", none))
        );
    }
    if operation == "cat" or operation == "stack" {
        const sources = try operand_ids(node);
        var values: List<usize> = list<usize>();
        for source in sources[..] {
            values.push(try mapped_id(mapped, source));
        }
        const axis = try require_usize(option_usize(node, "axis", none));
        if operation == "cat" {
            return try builder.add_cat(values.view(), axis);
        }
        return try builder.add_stack(values.view(), axis);
    }
    if operation == "layer_norm" or operation == "rms_norm" {
        var tag = OpTag.rms_norm;
        if operation == "layer_norm" {
            tag = OpTag.layer_norm;
        }
        return try builder.add_normalization(
            tag,
            try mapped_id(mapped, try operand(node, 0)),
            try require_usize(option_usize(node, "axis", none)),
            try require_number(option_number(node, "epsilon", none))
        );
    }
    if operation == "embedding" {
        return try builder.add_embedding(
            try mapped_id(mapped, try operand(node, 0)),
            try mapped_id(mapped, try operand(node, 1))
        );
    }
    if operation == "index_select" or operation == "gather" {
        const lhs = try mapped_id(mapped, try operand(node, 0));
        const rhs = try mapped_id(mapped, try operand(node, 1));
        const axis = try require_usize(option_usize(node, "axis", none));
        if operation == "index_select" {
            return try builder.add_index_select(lhs, rhs, axis);
        }
        return try builder.add_gather(lhs, rhs, axis);
    }
    if operation == "scatter_add" {
        return try builder.add_scatter_add(
            try mapped_id(mapped, try operand(node, 0)),
            try mapped_id(mapped, try operand(node, 1)),
            try mapped_id(mapped, try operand(node, 2)),
            try require_usize(option_usize(node, "axis", none))
        );
    }
    if operation == "topk_values" or operation == "topk_indices" {
        const pair = try builder.add_top_k_with_sort(
            try mapped_id(mapped, try operand(node, 0)),
            try require_usize(option_usize(node, "axis", none)),
            try require_usize(option_usize(node, "k", none)),
            option_bool(node, "largest", true),
            option_bool(node, "sorted", true)
        );
        if operation == "topk_indices" {
            return pair.indices;
        }
        return pair.values;
    }
    if operation == "one_hot" {
        return try builder.add_one_hot(
            try mapped_id(mapped, try operand(node, 0)),
            try require_usize(option_usize(node, "num_classes", none))
        );
    }
    if operation == "cross_entropy" or operation == "dense_cross_entropy" or
    operation == "log_softmax_nll" {
        var tag = OpTag.cross_entropy_indexed;
        if operation == "dense_cross_entropy" {
            tag = OpTag.cross_entropy;
        } else if operation == "log_softmax_nll" {
            tag = OpTag.log_softmax_nll;
        }
        const requested_axis = option_usize(node, "class_axis", 1);
        var axis: usize = 1;
        if requested_axis is selected_axis {
            axis = selected_axis;
        }
        return try builder.add_cross_entropy(tag,
            try mapped_id(mapped, try operand(node, 0)),
            try mapped_id(mapped, try operand(node, 1)),
            axis
        );
    }
    if operation == "quantized_matmul" {
        return try builder.add_quantized_matmul(
            try mapped_id(mapped, try operand(node, 0)),
            try mapped_id(mapped, try operand(node, 1)),
            try mapped_id(mapped, try operand(node, 2)),
            try mapped_id(mapped, try operand(node, 3)),
            try require_usize(option_usize(node, "bits", none)),
            try require_usize(option_usize(node, "group_size", none))
        );
    }
    throw invalid();
}

fn build_program(document: &const json.Value,
    source_instruction_by_step: &List<usize>, mapped: &List<usize>,
    selected_outputs: ?[]const usize
): Program ! ComputeError {
    if document is .object(program) {
        const nodes_index = try require_usize(json.field_index(&program, "nodes"));
        if program.members[nodes_index].value is .array(nodes) {
            if mapped.len() != nodes.items.len() {
                throw invalid();
            }
            var program_builder = builder();
            var state_sequence: u64 = 0;
            for node in nodes.items[..] {
                if node is .object(value) {
                    const id = try require_usize(json.usize_field(&value, "id"));
                    if id >= mapped.len() {
                        throw invalid();
                    }
                    const kind = try require_string(json.string_field(&value, "kind"));
                    if kind == "gradient" {
                        continue;
                    }
                    const spec_index = try require_usize(json.field_index(&value, "spec"));
                    const spec = try tensor_spec(&value.members[spec_index].value);
                    if json.string_field(&value, "role") is role {
                        if role == "argument" {
                            mapped[id] = program_builder.add_input(spec);
                            continue;
                        }
                        if role == "parameter" or role == "state" {
                            var selected_role = InputRole.model_state;
                            if role == "parameter" {
                                selected_role = InputRole.parameter;
                            }
                            mapped[id] = try program_builder.add_state_input(
                                spec, selected_role,
                                StateId.from_int(state_sequence)
                            );
                            state_sequence = state_sequence + 1;
                            continue;
                        }
                        if role == "constant" {
                            const value_index = try require_usize(json.field_index(&value, "value"));
                            mapped[id] = try add_constant(&program_builder, spec,
                                &value.members[value_index].value
                            );
                            continue;
                        }
                        throw invalid();
                    }
                    const instruction = program_builder.instruction_count();
                    mapped[id] = try add_operation(
                        &program_builder,
                        &value,
                        mapped.view()
                    );
                    while source_instruction_by_step.len() <
                        program_builder.instruction_count() {
                        source_instruction_by_step.push(instruction);
                    }
                } else {
                    throw invalid();
                }
            }
            var selected: List<usize> = list<usize>();
            if selected_outputs is outputs {
                for output in outputs[..] {
                    selected.push(try mapped_id(mapped.view(), output));
                }
            } else {
                const outputs_index = try require_usize(
                    json.field_index(&program, "outputs")
                );
                if program.members[outputs_index].value is .array(outputs) {
                    for output in outputs.items[..] {
                        if output is .number(number) {
                            const source = try require_usize(
                                json.parse_usize(number.spelling)
                            );
                            selected.push(try mapped_id(
                                mapped.view(),
                                source
                            ));
                        } else {
                            throw invalid();
                        }
                    }
                } else {
                    throw invalid();
                }
            }
            return try program_builder.finish(selected.view());
        }
    }
    throw invalid();
}

fn node_count(document: &const json.Value): usize !ComputeError {
    if document is .object(program) {
        const nodes_index = try require_usize(
            json.field_index(&program, "nodes")
        );
        if program.members[nodes_index].value is .array(nodes) {
            return nodes.items.len();
        }
    }
    throw invalid();
}

fn gradient_count(document: &const json.Value): usize !ComputeError {
    if document is .object(program) {
        const nodes_index = try require_usize(
            json.field_index(&program, "nodes")
        );
        if program.members[nodes_index].value is .array(nodes) {
            var count: usize = 0;
            for node in nodes.items[..] {
                if node is .object(value) {
                    if json.string_field(&value, "kind") is kind {
                        if kind == "gradient" {
                            count = count + 1;
                        }
                    }
                }
            }
            return count;
        }
    }
    throw invalid();
}

fn gradient_loss_id(document: &const json.Value): usize !ComputeError {
    if document is .object(program) {
        const nodes_index = try require_usize(
            json.field_index(&program, "nodes")
        );
        if program.members[nodes_index].value is .array(nodes) {
            for node in nodes.items[..] {
                if node is .object(value) {
                    if json.string_field(&value, "kind") is kind {
                        if kind == "gradient" {
                            return try operand(&value, 0);
                        }
                    }
                }
            }
        }
    }
    throw invalid();
}

fn gradient_target_id(document: &const json.Value,
    ordinal: usize
): usize !ComputeError {
    if document is .object(program) {
        const nodes_index = try require_usize(
            json.field_index(&program, "nodes")
        );
        if program.members[nodes_index].value is .array(nodes) {
            var current: usize = 0;
            var provenance: ?string = none;
            for node in nodes.items[..] {
                if node is .object(value) {
                    if json.string_field(&value, "kind") is kind {
                        if kind == "gradient" {
                            if current == ordinal {
                                provenance = option_string(
                                    &value,
                                    "with_respect_to"
                                );
                                break;
                            }
                            current = current + 1;
                        }
                    }
                }
            }
            if provenance is selected {
                for node in nodes.items[..] {
                    if node is .object(value) {
                        if json.string_field(
                            &value,
                            "provenance"
                        ) is candidate {
                            if candidate == selected {
                                return try require_usize(
                                    json.usize_field(&value, "id")
                                );
                            }
                        }
                    }
                }
            }
        }
    }
    throw invalid();
}

fn goal(value: string): OptimizationGoal {
    if value == "memory" {
        return OptimizationGoal.memory;
    }
    if value == "latency" {
        return OptimizationGoal.latency;
    }
    if value == "throughput" {
        return OptimizationGoal.throughput;
    }
    return OptimizationGoal.balanced;
}

fn program_kind(document: &const json.Value): string !ComputeError {
    if document is .object(program) {
        return try require_string(json.string_field(&program, "kind"));
    }
    throw invalid();
}

fn compile_settings(options_json: ?string): CompileSettings {
    var result = CompileSettings {
        optimization_level: "safe",
        optimization_goal: "balanced",
        dense_inputs: false,
        residual_policy: ResidualPolicy.retain,
        search_limits: default_search_limits()
    };
    if options_json is text {
        const document = parse(text) catch (error_code) {
            const ignored = error_code;
            return result;
        };
        if document is .object(options) {
            if json.string_field(&options, "optimizationLevel") is level {
                result.optimization_level = level;
            }
            if json.string_field(&options, "optimizationGoal") is selected_goal {
                result.optimization_goal = selected_goal;
            }
            result.dense_inputs = json.bool_field(
                &options,
                "denseInputs",
                false
            );
            if json.string_field(&options, "residualPolicy") is policy {
                if policy == "automatic" {
                    result.residual_policy = ResidualPolicy.automatic;
                } else if policy == "recompute" {
                    result.residual_policy = ResidualPolicy.recompute;
                }
            }
            if json.field_index(&options, "searchLimits") is limits_index {
                if options.members[limits_index].value is .object(limits) {
                    if json.usize_field(
                        &limits,
                        "structuralStates"
                    ) is value {
                        result.search_limits.structural_states = value;
                    }
                    if json.usize_field(
                        &limits,
                        "structuralExpansions"
                    ) is value {
                        result.search_limits.structural_expansions = value;
                    }
                    if json.usize_field(
                        &limits,
                        "completeEvaluations"
                    ) is value {
                        result.search_limits.complete_evaluations = value;
                    }
                    if json.usize_field(
                        &limits,
                        "physicalWork"
                    ) is value {
                        result.search_limits.physical_work = value;
                    }
                    if json.usize_field(
                        &limits,
                        "readyWork"
                    ) is value {
                        result.search_limits.ready_work = value;
                    }
                    if json.usize_field(
                        &limits,
                        "maxPeakActiveBytes"
                    ) is value {
                        result.search_limits.max_peak_active_bytes = value;
                    }
                    if json.usize_field(
                        &limits,
                        "maxInvocationScratchBytes"
                    ) is value {
                        result.search_limits.max_invocation_scratch_bytes =
                            value;
                    }
                }
            }
        }
    }
    return result;
}

pub fn compile_program(document: &const json.Value,
    options_json: ?string
): CompiledGraph ! ComputeError {
    var source_instruction_by_step: List<usize> = list<usize>();
    var mapped: List<usize> = list<usize>();
    const node_total = try node_count(document);
    var node_index: usize = 0;
    while node_index < node_total {
        mapped.push((0 as usize) -% 1);
        node_index = node_index + 1;
    }
    var needs_seed = false;
    var loss_id: usize = 0;
    var loss_outputs: [1]usize = [0];
    var selected_outputs: ?[]const usize = none;
    const kind = try program_kind(document);
    if kind == "gradient" {
        loss_id = try gradient_loss_id(document);
        loss_outputs[0] = loss_id;
        const frozen_loss_outputs: []const usize = loss_outputs[..];
        selected_outputs = frozen_loss_outputs;
    }
    var program = try build_program(
        document,
        &source_instruction_by_step,
        &mapped,
        selected_outputs
    );
    if kind == "gradient" {
        const derivative_count = try gradient_count(document);
        if derivative_count == 0 {
            throw invalid();
        }
        var targets: List<usize> = list<usize>();
        var derivative_index: usize = 0;
        while derivative_index < derivative_count {
            const source = try gradient_target_id(
                document,
                derivative_index
            );
            targets.push(try mapped_id(mapped.view(), source));
            derivative_index = derivative_index + 1;
        }
        var derivative = try differentiate(
            &program,
            try mapped_id(mapped.view(), loss_id),
            targets[..]
        );
        program = derivative.into_program();
        needs_seed = true;
    } else if kind != "authored" {
        throw invalid();
    }
    while source_instruction_by_step.len() < program.instructions.len() {
        source_instruction_by_step.push(source_instruction_by_step.len());
    }
    const settings = compile_settings(options_json);
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
    var selected_goal = goal(settings.optimization_goal);
    if settings.optimization_level == "none" {
        selected_goal = OptimizationGoal.baseline;
    }
    var contexts: [1]CpuContext = [cpu_context()];
    defer close_cpu_context(contexts[..].ptr);
    const backend = cpu_backend(contexts[..].ptr);
    const executable = try compile_for_resident_with_search_limits(
        program,
        selected_goal,
        &backend,
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
        optimization_goal: settings.optimization_goal,
        optimization_level: settings.optimization_level,
        residual_policy: settings.residual_policy,
        observed_run_count: 0,
        failed_run_count: 0,
    };
}
