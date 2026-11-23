import compute_core.lib as compute;
import { ComputeError, DType, OptimizationGoal } from compute_core.lib;
import { OpTag } from compute_core.types;
import { SliceRange } from compute_core.program.operation;
import {
    ResidualPolicy,
    default_search_limits
} from compute_core.compiler.executable;
import { error_name } from compute_core.compiler.index;
import compute_core.program.support.bfloat;
import compute_core.backend.cpu.math_unary as cpu_math;
import compute_core.backend.cpu.numeric;
import { env_get } from std.process;
import { Metric, MetricKind, MetricUnit } from std.telemetry;
import { monotonic_ns } from std.time;
import hao.js;
import { Runtime } from hao.runtime;
import .support.cyna_program as graph;
import .support.cyna_evidence as plan;
import {
    CompileOptimizationLevel,
    ExplanationLevel,
    NumericalPolicy,
    ProgramKind,
    ProgramNodeKind
} from .support.cyna_program;

struct SessionSlot {
    generation: u32;
    alive: bool;
    context_address: usize;
    cuda: bool;
    device: string;
    capture_enabled: bool;
    capture_capacity: usize;
    timing_scope: string;
    hardware_metrics: bool;
    profiler_markers: bool;
}

struct TensorSlot {
    value: *TensorBox;
    dense: bool;
}

struct TensorBox {
    value: compute.Tensor;
}

struct ExecutableSlot {
    value: *ExecutableBox;
    session: usize;
    generation: u32;
    alive: bool;
    explanation: string;
    routing: string;
    run_count: usize;
    failed_run_count: usize;
    cache_hits: usize;
    cache_misses: usize;
    captured_count: usize;
    dropped_count: usize;
    elapsed_ns: u64;
    capture_enabled: bool;
    capture_capacity: usize;
    timing_scope: string;
    hardware_metrics: bool;
    profiler_markers: bool;
}

struct ExecutableBox {
    value: graph.CompiledGraph;
}

struct CheckpointTensor {
    name: string;
    dtype: DType;
    dimensions: List<usize>;
    start: usize;
    end: usize;
}

const maximum_sessions: usize = 64;
const maximum_tensors: usize = 16384;
const maximum_executables: usize = 4096;

var sessions: [64]SessionSlot = [SessionSlot {
    generation: 0,
    alive: false,
    context_address: 0,
    cuda: false,
    device: "cpu",
    capture_enabled: false,
    capture_capacity: 0,
    timing_scope: "none",
    hardware_metrics: false,
    profiler_markers: false
}] ** 64;
var tensor_pointers: [16384]usize = [0] ** 16384;
var tensor_alive: [16384]bool = [false] ** 16384;
var tensor_sessions: [16384]usize = [0] ** 16384;
var executable_pointers: [4096]usize = [0] ** 4096;
var executable_alive: [4096]bool = [false] ** 4096;
var executable_sessions: [4096]usize = [0] ** 4096;
var next_session: usize = 0;
var next_tensor: usize = 0;
var next_executable: usize = 0;
var telemetry_host_handle: usize = 0;

pub fn configure(runtime: &const Runtime): void {
    telemetry_host_handle = runtime.handle as usize;
}

fn now_ns(): u64 {
    return (monotonic_ns() catch (error_code) => 0) as u64;
}

fn add_metric(metric: Metric, value: i64): void {
    if telemetry_host_handle == 0 {
        return;
    }
    const runtime = Runtime {
        handle: telemetry_host_handle as c_size_t
    };
    const backend = runtime.telemetry();
    backend.add(metric, value);
}

fn generation(): u32 {
    const value = next_session + next_tensor + next_executable + 1;
    return value as u32;
}

fn session_generation(handle: usize): u32 {
    if handle < maximum_sessions {
        return sessions[handle].generation;
    }
    return 0;
}

fn session_device(handle: usize): string {
    if handle < maximum_sessions and sessions[handle].alive {
        return sessions[handle].device;
    }
    return "cpu";
}

fn context_pointer(address: usize): *c_void {
    unchecked {
        return address as! *c_void;
    }
}

fn session_backend(handle: usize): compute.PreparedBackend {
    const session = sessions[handle];
    const context = context_pointer(session.context_address);
    if session.cuda {
        return compute.cuda_backend(context as! *compute.CudaContext);
    }
    return compute.cpu_backend(context as! *compute.CpuContext);
}

fn session_evidence_backend(handle: usize): string {
    if handle < maximum_sessions and sessions[handle].cuda {
        return "cyna_cuda";
    }
    return "cyna_cpu";
}

fn routing_json(device: string): string {
    return $"\{\"schema_version\":1,\"backend\":\"cyna\",\"configured_backend\":\"{device}\",\"transport\":\"in_process\",\"category\":\"success\",\"reason\":\"lowered\",\"operation\":null,\"option\":null\}";
}

fn tensor_slot(handle: usize): *TensorSlot {
    unchecked {
        return tensor_pointers[handle] as! *TensorSlot;
    }
}

fn tensor_value(handle: usize): &const compute.Tensor {
    const slot = tensor_slot(handle);
    unchecked {
        const box = (*slot).value;
        return &(*box).value;
    }
}

fn tensor_dense(handle: usize): bool {
    const slot = tensor_slot(handle);
    unchecked {
        return (*slot).dense;
    }
}

fn executable_slot(handle: usize): *ExecutableSlot {
    unchecked {
        return executable_pointers[handle] as! *ExecutableSlot;
    }
}

fn executable_value(handle: usize): &const graph.CompiledGraph {
    const slot = executable_slot(handle);
    unchecked {
        const box = (*slot).value;
        return &(*box).value;
    }
}

fn dispose_tensor_slot(raw_slot: *TensorSlot): void {
    unchecked {
        var slot: ^TensorSlot = resource.from_raw(raw_slot);
        const value = slot.value;
        drop slot;
        var box: ^TensorBox = resource.from_raw(value);
        drop box;
    }
}

fn dispose_executable_slot(raw_slot: *ExecutableSlot): void {
    unchecked {
        var slot: ^ExecutableSlot = resource.from_raw(raw_slot);
        const value = slot.value;
        drop slot;
        var box: ^ExecutableBox = resource.from_raw(value);
        drop box;
    }
}

fn argument(arguments_pointer: *const js.JsValue, index: usize): js.JsValue {
    unchecked {
        return *(arguments_pointer + index);
    }
}

fn number(ctx: *c_void, value: i64): js.JsValue {
    return js.new_number(ctx, value as f64);
}

fn boolean(ctx: *c_void, value: bool): js.JsValue {
    return js.bool(ctx, value);
}

fn property_int(ctx: *c_void, object: js.JsValue, name: c_str): ?i64 {
    const value = js.get_property_str(ctx, object, name);
    defer js.free_value(ctx, value);
    return js.read_int(ctx, value);
}

fn object_handle(ctx: *c_void, value: js.JsValue): ?usize {
    const handle = property_int(ctx, value, "__affon_handle");
    if handle is selected {
        if selected >= 0 {
            return selected as usize;
        }
    }
    return none;
}

fn dtypes(value: string): ?DType {
    if value == "f32" {
        return DType.f32;
    }
    if value == "f64" {
        return DType.f64;
    }
    if value == "u32" {
        return DType.u32;
    }
    if value == "i64" {
        return DType.i64;
    }
    if value == "bf16" {
        return DType.bf16;
    }
    return none;
}

fn dtype_name(value: DType): string {
    return switch value {
        .f32 => "f32";
        .f64 => "f64";
        .u32 => "u32";
        .i64 => "i64";
        .bf16 => "bf16";
    };
}

fn shape(ctx: *c_void, value: js.JsValue): ?List<usize> {
    if js.is_array(value) == 0 {
        return none;
    }
    const length = property_int(ctx, value, "length");
    if length is selected {
        if selected < 0 or selected > 32 {
            return none;
        }
        var result: List<usize> = list<usize>();
        var index: usize = 0;
        while index < selected as usize {
            const item = js.get_property_uint32(ctx, value, index as c_uint);
            defer js.free_value(ctx, item);
            if js.read_int(ctx, item) is dimension {
                if dimension < 0 {
                    return none;
                }
                result.push(dimension as usize);
            } else {
                return none;
            }
            index = index + 1;
        }
        return result;
    }
    return none;
}

fn spec(selected_dtype: DType,
    dimensions: []const usize
): compute.TensorSpec ! compute.ComputeError {
    if selected_dtype == DType.f32 {
        return try compute.f32_spec(dimensions);
    }
    if selected_dtype == DType.f64 {
        return try compute.f64_spec(dimensions);
    }
    if selected_dtype == DType.u32 {
        return try compute.u32_spec(dimensions);
    }
    if selected_dtype == DType.i64 {
        return try compute.i64_spec(dimensions);
    }
    return try compute.bf16_spec(dimensions);
}

fn element_count(dimensions: []const usize): ?usize {
    var count: usize = 1;
    for dimension in dimensions {
        if dimension != 0 and count > ((0 as usize) -% 1) / dimension {
            return none;
        }
        count = count * dimension;
    }
    return count;
}

fn create_tensor_object(ctx: *c_void, value: compute.Tensor,
    session: usize, session_generation: u32, dense: bool
): js.JsValue {
    if next_tensor >= maximum_tensors {
        return js.type_error(ctx, "Tensor capacity exhausted");
    }
    const handle = next_tensor;
    next_tensor = next_tensor + 1;
    const tensor_generation = generation();
    const dimension_count = value.spec.dimensions.len();
    const selected_dtype = dtype_name(value.spec.dtype);
    var dimensions: List<js.JsValue> = list<js.JsValue>();
    for dimension in value.spec.dimensions[..] {
        dimensions.push(number(ctx, dimension as i64));
    }
    const box = new TensorBox {
        value: value
    };
    const slot = new TensorSlot {
        value: resource.leak(box),
        dense: dense
    };
    tensor_pointers[handle] = resource.leak(slot) as! usize;
    tensor_sessions[handle] = session;
    tensor_alive[handle] = true;
    const object = js.new_object(ctx);
    js.set_number(ctx, object, "ndim", dimension_count as f64);
    js.set_string(ctx, object, "dtype", selected_dtype);
    const shape_value = js.new_array(ctx);
    var index: usize = 0;
    while index < dimensions.len() {
        const ignored_shape = js.set_property_uint32(ctx, shape_value,
            index as c_uint,
            dimensions[index]
        );
        const ignored = ignored_shape;
        index = index + 1;
    }
    const ignored_shape_field = js.set_property_str(
        ctx, object, "shape", shape_value
    );
    const ignored = ignored_shape_field;
    js.set_string(ctx, object, "device", session_device(session));
    js.set_number(ctx, object, "__affon_handle", handle as f64);
    js.set_number(ctx, object, "__affon_generation",
        tensor_generation as f64
    );
    const ignored_session = session;
    const ignored_generation = session_generation;
    js.set_function(ctx, object, "item", tensor_item, 0);
    js.set_function(ctx, object, "to_array", tensor_to_array, 0);
    js.set_function(ctx, object, "to_bytes", tensor_to_bytes, 0);
    js.set_function(ctx, object, "toString", tensor_text, 0);
    js.set_function(ctx, object, "repr", tensor_text, 0);
    js.set_function(ctx, object, "dispose", tensor_dispose, 0);
    return object;
}

fn live_tensor(ctx: *c_void, value: js.JsValue): ?usize {
    if object_handle(ctx, value) is handle {
        if handle < maximum_tensors {
            if tensor_alive[handle] {
                return handle;
            }
        }
    }
    return none;
}

pub fn live_tensor_handle(ctx: *c_void, value: js.JsValue): ?usize {
    return live_tensor(ctx, value);
}

pub fn tensor_for_handle(handle: usize): &const compute.Tensor {
    return tensor_value(handle);
}

pub fn detached_tensor(ctx: *c_void, value: compute.Tensor): js.JsValue {
    return create_tensor_object(ctx, value, maximum_sessions, 0, true);
}

fn tensor_number(value: &const compute.Tensor, index: usize): f64 ! compute.ComputeError {
    return switch value.data {
        .f32(values) => numeric.f32_to_f64(values[index]);
        .f64(values) => values[index];
        .u32(values) => values[index] as f64;
        .i64(values) => values[index] as f64;
        .bf16(values) => numeric.f32_to_f64(bfloat.to_f32(values[index]));
    };
}

fn tensor_item(ctx: *c_void, this_value: js.JsValue, argument_count: c_int,
    arguments_pointer: *const js.JsValue
): js.JsValue {
    const ignored_count = argument_count;
    const ignored_arguments = arguments_pointer;
    if live_tensor(ctx, this_value) is handle {
        const tensor = tensor_value(handle);
        const count = tensor.spec.element_count() catch (error_code) {
            const ignored = error_code;
            return js.type_error(ctx, "invalid Tensor");
        };
        if count != 1 {
            return js.type_error(ctx, "item() requires a one-element Tensor");
        }
        const value = tensor_number(tensor, 0) catch (error_code) {
            const ignored = error_code;
            return js.type_error(ctx, "invalid Tensor");
        };
        return js.new_number(ctx, value);
    }
    return js.type_error(ctx, "Tensor has been disposed");
}

fn tensor_to_array(ctx: *c_void, this_value: js.JsValue,
    argument_count: c_int, arguments_pointer: *const js.JsValue
): js.JsValue {
    const ignored_count = argument_count;
    const ignored_arguments = arguments_pointer;
    if live_tensor(ctx, this_value) is handle {
        var cursor: usize = 0;
        return build_tensor_array(ctx, handle, 0, &cursor);
    }
    return js.type_error(ctx, "Tensor has been disposed");
}

fn build_tensor_array(ctx: *c_void, handle: usize, depth: usize,
    cursor: &usize
): js.JsValue {
    if tensor_alive[handle] {
        const tensor = tensor_value(handle);
        const dimensions = tensor.spec.dimensions[..];
        if depth == dimensions.len() {
            var selected: usize = 0;
            unchecked {
                selected = cursor;
            }
            const value = tensor_number(tensor, selected)
            catch (error_code) {
                const ignored = error_code;
                return js.type_error(ctx, "invalid Tensor");
            };
            unchecked {
                cursor = selected + 1;
            }
            return js.new_number(ctx, value);
        }
        const result = js.new_array(ctx);
        var index: usize = 0;
        while index < dimensions[depth] {
            const ignored_item = js.set_property_uint32(ctx, result,
                index as c_uint,
                build_tensor_array(ctx, handle, depth + 1, cursor)
            );
            const ignored = ignored_item;
            index = index + 1;
        }
        return result;
    }
    return js.type_error(ctx, "invalid Tensor");
}

fn tensor_to_bytes(ctx: *c_void, this_value: js.JsValue,
    argument_count: c_int, arguments_pointer: *const js.JsValue
): js.JsValue {
    const ignored_count = argument_count;
    const ignored_arguments = arguments_pointer;
    if live_tensor(ctx, this_value) is handle {
        const bytes = graph.tensor_bytes(tensor_value(handle));
        const view = bytes[..];
        return js.new_array_buffer_copy(
            ctx,
            view.ptr as! *const c_uchar,
            view.len() as c_size_t
        );
    }
    return js.type_error(ctx, "Tensor has been disposed");
}

fn tensor_text(ctx: *c_void, this_value: js.JsValue,
    argument_count: c_int, arguments_pointer: *const js.JsValue
): js.JsValue {
    const ignored_count = argument_count;
    const ignored_arguments = arguments_pointer;
    if live_tensor(ctx, this_value) is handle {
        const tensor = tensor_value(handle);
        const dimension_count = tensor.spec.dimensions.len();
        const selected_dtype = dtype_name(tensor.spec.dtype);
        return js.string(ctx,
            $"Tensor(shape={dimension_count}, dtype={selected_dtype})"
        );
    }
    return js.string(ctx, "Tensor(disposed)");
}

fn tensor_dispose(ctx: *c_void, this_value: js.JsValue,
    argument_count: c_int, arguments_pointer: *const js.JsValue
): js.JsValue {
    const ignored_count = argument_count;
    const ignored_arguments = arguments_pointer;
    if object_handle(ctx, this_value) is handle {
        if handle < maximum_tensors {
            if tensor_alive[handle] {
                tensor_alive[handle] = false;
                const pointer = tensor_pointers[handle];
                tensor_pointers[handle] = 0;
                if pointer != 0 {
                    dispose_tensor_slot(pointer as! *TensorSlot);
                }
            }
        }
    }
    return js.undefined();
}

fn optional_i64_property(ctx: *c_void, object: js.JsValue,
    name: c_str
): ?i64 {
    const value = js.get_property_str(ctx, object, name);
    defer js.free_value(ctx, value);
    if js.is_undefined(value) {
        return none;
    }
    return js.read_int(ctx, value);
}

fn parse_decimal_usize(text: string): ?usize {
    var value: usize = 0;
    const bytes = text.bytes();
    if bytes.len() == 0 {
        return none;
    }
    var index: usize = 0;
    while index < bytes.len() {
        const byte = bytes[index];
        if byte < 48 or byte > 57 {
            return none;
        }
        const digit = (byte - 48) as usize;
        if value > ((0 as usize) -% 1 - digit) / 10 {
            return none;
        }
        value = value * 10 + digit;
        index = index + 1;
    }
    return value;
}

fn decode_dtype(value: i64): ?DType {
    return switch value {
        1 => DType.f32;
        2 => DType.f64;
        3 => DType.i64;
        4 => DType.u32;
        5 => DType.bf16;
        _ => none;
    };
}

fn decode_program_kind(value: i64): ?ProgramKind {
    return switch value {
        0 => ProgramKind.authored;
        1 => ProgramKind.gradient;
        _ => none;
    };
}

fn decode_program_node_kind(value: i64): ?ProgramNodeKind {
    return switch value {
        0 => ProgramNodeKind.argument;
        1 => ProgramNodeKind.parameter;
        2 => ProgramNodeKind.state;
        3 => ProgramNodeKind.constant;
        4 => ProgramNodeKind.operation;
        5 => ProgramNodeKind.gradient;
        _ => none;
    };
}

fn decode_compile_level(value: i64): ?CompileOptimizationLevel {
    return switch value {
        0 => CompileOptimizationLevel.disabled;
        1 => CompileOptimizationLevel.safe;
        _ => none;
    };
}

fn decode_optimization_goal(value: i64): ?OptimizationGoal {
    return switch value {
        1 => OptimizationGoal.memory;
        2 => OptimizationGoal.balanced;
        3 => OptimizationGoal.latency;
        4 => OptimizationGoal.throughput;
        _ => none;
    };
}

fn decode_numerical_policy(value: i64): ?NumericalPolicy {
    return switch value {
        0 => NumericalPolicy.exact_only;
        1 => NumericalPolicy.backend_equivalent;
        2 => NumericalPolicy.approximate;
        _ => none;
    };
}

fn decode_explanation_level(value: i64): ?ExplanationLevel {
    return switch value {
        0 => ExplanationLevel.summary;
        1 => ExplanationLevel.detailed;
        _ => none;
    };
}

fn decode_residual_policy(value: i64): ?ResidualPolicy {
    return switch value {
        0 => ResidualPolicy.retain;
        1 => ResidualPolicy.automatic;
        2 => ResidualPolicy.recompute;
        _ => none;
    };
}

fn decode_op_tag(value: i64): ?OpTag {
    return switch value {
        1 => OpTag.add;
        2 => OpTag.sub;
        3 => OpTag.mul;
        4 => OpTag.div;
        5 => OpTag.matmul;
        6 => OpTag.abs;
        7 => OpTag.neg;
        8 => OpTag.sign;
        9 => OpTag.relu;
        10 => OpTag.sum_all;
        11 => OpTag.mean_all;
        12 => OpTag.sum_axis;
        13 => OpTag.mean_axis;
        14 => OpTag.min_all;
        15 => OpTag.max_all;
        16 => OpTag.argmin_all;
        17 => OpTag.argmax_all;
        18 => OpTag.min_axis;
        19 => OpTag.max_axis;
        20 => OpTag.argmin_axis;
        21 => OpTag.argmax_axis;
        22 => OpTag.variance_all;
        23 => OpTag.std_all;
        24 => OpTag.variance_axis;
        25 => OpTag.std_axis;
        26 => OpTag.reshape;
        27 => OpTag.permute;
        28 => OpTag.transpose;
        29 => OpTag.squeeze;
        30 => OpTag.unsqueeze;
        31 => OpTag.contiguous;
        32 => OpTag.slice;
        33 => OpTag.cast;
        34 => OpTag.scatter_add;
        35 => OpTag.gather;
        36 => OpTag.index_select;
        37 => OpTag.embedding;
        38 => OpTag.one_hot;
        39 => OpTag.top_k_values;
        40 => OpTag.top_k_indices;
        41 => OpTag.eq;
        42 => OpTag.lt;
        43 => OpTag.gt;
        44 => OpTag.where;
        45 => OpTag.masked_fill;
        46 => OpTag.cat;
        47 => OpTag.stack;
        48 => OpTag.clamp;
        49 => OpTag.exp;
        50 => OpTag.log;
        51 => OpTag.sqrt;
        52 => OpTag.sigmoid;
        53 => OpTag.silu;
        54 => OpTag.tanh;
        55 => OpTag.erf;
        56 => OpTag.gelu;
        57 => OpTag.sin;
        58 => OpTag.cos;
        59 => OpTag.softmax;
        60 => OpTag.log_softmax;
        61 => OpTag.cross_entropy;
        62 => OpTag.cross_entropy_indexed;
        63 => OpTag.layer_norm;
        64 => OpTag.rms_norm;
        65 => OpTag.quantized_matmul;
        66 => OpTag.log_softmax_nll;
        67 => OpTag.dot;
        68 => OpTag.reduce_to_shape;
        _ => none;
    };
}

fn optional_usize_property(ctx: *c_void, object: js.JsValue,
    name: c_str
): ?usize {
    if optional_i64_property(ctx, object, name) is value {
        if value >= 0 {
            return value as usize;
        }
    }
    return none;
}

fn optional_number_property(ctx: *c_void, object: js.JsValue,
    name: c_str
): ?f64 {
    const value = js.get_property_str(ctx, object, name);
    defer js.free_value(ctx, value);
    if js.is_undefined(value) {
        return none;
    }
    return js.read_number(ctx, value);
}

fn optional_flag_property(ctx: *c_void, object: js.JsValue,
    name: c_str, fallback: bool
): ?bool {
    const value = optional_i64_property(ctx, object, name);
    if value is selected {
        if selected == 0 {
            return false;
        }
        if selected == 1 {
            return true;
        }
        return none;
    }
    return fallback;
}

fn usize_array_property(ctx: *c_void, object: js.JsValue,
    name: c_str
): ?List<usize> {
    const value = js.get_property_str(ctx, object, name);
    defer js.free_value(ctx, value);
    if js.is_undefined(value) {
        return none;
    }
    if js.is_array(value) == 0 {
        return none;
    }
    const length = property_int(ctx, value, "length");
    if length is selected {
        if selected < 0 or selected > 1_048_576 {
            return none;
        }
        var result: List<usize> = list<usize>();
        var index: usize = 0;
        while index < selected as usize {
            const item = js.get_property_uint32(ctx, value, index as c_uint);
            defer js.free_value(ctx, item);
            if js.read_int(ctx, item) is item_value {
                if item_value < 0 {
                    return none;
                }
                result.push(item_value as usize);
            } else {
                return none;
            }
            index = index + 1;
        }
        return result;
    }
    return none;
}

fn slice_ranges_property(ctx: *c_void, object: js.JsValue
): ?List<SliceRange> {
    const value = js.get_property_str(ctx, object, "ranges");
    defer js.free_value(ctx, value);
    if js.is_undefined(value) {
        return none;
    }
    if js.is_array(value) == 0 {
        return none;
    }
    const length = property_int(ctx, value, "length");
    if length is selected {
        if selected < 0 or selected > 32 {
            return none;
        }
        var result: List<SliceRange> = list<SliceRange>();
        var index: usize = 0;
        while index < selected as usize {
            const item = js.get_property_uint32(ctx, value, index as c_uint);
            defer js.free_value(ctx, item);
            if optional_usize_property(ctx, item, "start") is start {
                if optional_usize_property(ctx, item, "stop") is stop {
                    var step: isize = 1;
                    if optional_usize_property(ctx, item, "step") is selected_step {
                        step = selected_step as isize;
                    }
                    result.push(SliceRange {
                        start: start,
                        stop: stop,
                        step: step
                    });
                } else {
                    return none;
                }
            } else {
                return none;
            }
            index = index + 1;
        }
        return result;
    }
    return none;
}

fn bytes_property(ctx: *c_void, object: js.JsValue,
    name: c_str
): ?List<u8> {
    const value = js.get_property_str(ctx, object, name);
    defer js.free_value(ctx, value);
    if js.is_undefined(value) {
        return none;
    }
    var length_storage: [1]c_size_t = [0];
    const pointer = js.get_array_buffer(
        ctx,
        length_storage[..].ptr,
        value
    );
    if pointer is source {
        const length = length_storage[0];
        if length as usize > 268_435_456 {
            return none;
        }
        var result: List<u8> = list<u8>();
        var index: usize = 0;
        while index < length as usize {
            unchecked {
                result.push(*(source + index) as u8);
            }
            index = index + 1;
        }
        return result;
    }
    return none;
}

fn default_operation_options(): graph.OperationOptions {
    const result: graph.OperationOptions = {
        axis: none,
        keep_dims: false,
        minimum: none,
        maximum: none,
        value: none,
        epsilon: none,
        k: none,
        num_classes: none,
        class_axis: 1,
        bits: none,
        group_size: none,
        axes: none,
        permutation: none,
        ranges: none,
        largest: true,
        sorted: true
    };
    return result;
}

fn empty_tensor_spec(): TensorSpec {
    return TensorSpec {
        dtype: DType.f32,
        dimensions: list<usize>(),
        axis_names: none
    };
}

fn default_program_node_input(): graph.ProgramNodeInput {
    const result: graph.ProgramNodeInput = {
        id: 0,
        kind: ProgramNodeKind.argument,
        spec: empty_tensor_spec(),
        op: none,
        operands: list<usize>(),
        options: default_operation_options(),
        constant: none,
        constant_bytes: none,
        gradient_target: none
    };
    return result;
}

fn default_program_input(): graph.ProgramInput {
    const result: graph.ProgramInput = {
        kind: ProgramKind.authored,
        nodes: list<graph.ProgramNodeInput>(),
        outputs: none
    };
    return result;
}

fn default_compile_options_input(): graph.CompileOptionsInput {
    const result: graph.CompileOptionsInput = {
        optimization_level: CompileOptimizationLevel.safe,
        optimization_goal: OptimizationGoal.balanced,
        numerical_policy: NumericalPolicy.backend_equivalent,
        explanation_level: ExplanationLevel.summary,
        dense_inputs: false,
        residual_policy: ResidualPolicy.retain,
        search_limits: default_search_limits()
    };
    return result;
}

fn decode_operation_options(ctx: *c_void, value: js.JsValue,
    op: OpTag, output: &graph.OperationOptions
): bool {
    output.axis = none;
    output.keep_dims = false;
    output.minimum = none;
    output.maximum = none;
    output.value = none;
    output.epsilon = none;
    output.k = none;
    output.num_classes = none;
    output.class_axis = 1;
    output.bits = none;
    output.group_size = none;
    output.axes = none;
    output.permutation = none;
    output.ranges = none;
    output.largest = true;
    output.sorted = true;
    if op == OpTag.sum_all or op == OpTag.mean_all or
    op == OpTag.min_all or op == OpTag.max_all or
    op == OpTag.variance_all or op == OpTag.std_all or
    op == OpTag.argmin_all or op == OpTag.argmax_all or
    op == OpTag.sum_axis or op == OpTag.mean_axis or
    op == OpTag.min_axis or op == OpTag.max_axis or
    op == OpTag.variance_axis or op == OpTag.std_axis or
    op == OpTag.argmin_axis or op == OpTag.argmax_axis {
        output.axis = optional_usize_property(ctx, value, "axis");
        if optional_flag_property(ctx, value, "keep_dims", false) is keep {
            output.keep_dims = keep;
        } else {
            return false;
        }
    }
    if op == OpTag.softmax or op == OpTag.log_softmax or
    op == OpTag.squeeze or op == OpTag.unsqueeze or
    op == OpTag.cat or op == OpTag.stack or op == OpTag.scatter_add or
    op == OpTag.gather or op == OpTag.index_select or
    op == OpTag.layer_norm or op == OpTag.rms_norm {
        output.axis = optional_usize_property(ctx, value, "axis");
    }
    if op == OpTag.permute {
        output.axes = usize_array_property(ctx, value, "axes");
        if output.axes is none {
            return false;
        }
    }
    if op == OpTag.transpose {
        output.permutation =
        usize_array_property(ctx, value, "permutation");
    }
    if op == OpTag.slice {
        output.ranges = slice_ranges_property(ctx, value);
        if output.ranges is none {
            return false;
        }
    }
    if op == OpTag.clamp {
        output.minimum = optional_number_property(ctx, value, "min");
        output.maximum = optional_number_property(ctx, value, "max");
    }
    if op == OpTag.masked_fill {
        output.value = optional_number_property(ctx, value, "fill_value");
    }
    if op == OpTag.layer_norm or op == OpTag.rms_norm {
        output.epsilon = optional_number_property(ctx, value, "epsilon");
    }
    if op == OpTag.top_k_values or op == OpTag.top_k_indices {
        output.axis = optional_usize_property(ctx, value, "axis");
        output.k = optional_usize_property(ctx, value, "k");
        if optional_flag_property(ctx, value, "largest", true) is largest {
            output.largest = largest;
        } else {
            return false;
        }
        if optional_flag_property(ctx, value, "sorted", true) is sorted {
            output.sorted = sorted;
        } else {
            return false;
        }
    }
    if op == OpTag.one_hot {
        output.num_classes =
        optional_usize_property(ctx, value, "num_classes");
    }
    if op == OpTag.cross_entropy or op == OpTag.cross_entropy_indexed or
    op == OpTag.log_softmax_nll {
        if optional_usize_property(ctx, value, "class_axis") is axis {
            output.class_axis = axis;
        }
    }
    if op == OpTag.quantized_matmul {
        output.bits = optional_usize_property(ctx, value, "bits");
        output.group_size =
        optional_usize_property(ctx, value, "group_size");
    }
    return true;
}

fn decode_program_node(ctx: *c_void, value: js.JsValue,
    index: usize, output: &graph.ProgramNodeInput
): bool {
    const id = optional_usize_property(ctx, value, "id");
    if id is selected_id {
        if selected_id != index {
            return false;
        }
        output.id = selected_id;
    } else {
        return false;
    }
    const kind_value = optional_i64_property(ctx, value, "kind");
    if kind_value is none {
        return false;
    }
    const kind = decode_program_node_kind(kind_value);
    if kind is none {
        return false;
    }
    output.kind = kind;
    const dtype_value = optional_i64_property(ctx, value, "dtype");
    if dtype_value is none {
        return false;
    }
    const selected_dtype = decode_dtype(dtype_value);
    if selected_dtype is none {
        return false;
    }
    const shape_value = js.get_property_str(ctx, value, "shape");
    const dimensions = shape(ctx, shape_value);
    js.free_value(ctx, shape_value);
    if dimensions is none {
        return false;
    }
    const tensor_spec = spec(selected_dtype, dimensions[..])
    catch (error_code) {
        const ignored = error_code;
        return false;
    };
    output.spec = tensor_spec;
    output.operands = list<usize>();
    if usize_array_property(ctx, value, "operands") is selected_operands {
        output.operands = selected_operands;
    }
    output.op = none;
    output.options = default_operation_options();
    const op_value = optional_i64_property(ctx, value, "op");
    if op_value is numeric_op {
        const selected_op = decode_op_tag(numeric_op);
        if selected_op is tag {
            output.op = tag;
            if not decode_operation_options(
                ctx,
                value,
                tag,
                &output.options
            ) {
                return false;
            }
        } else if kind_value == 4 {
            return false;
        }
    } else if kind_value == 4 {
        return false;
    }
    output.constant = optional_number_property(ctx, value, "constant");
    output.constant_bytes = bytes_property(ctx, value, "constant_bytes");
    output.gradient_target =
    optional_usize_property(ctx, value, "gradient_target");
    return true;
}

fn positive_usize_property(ctx: *c_void, object: js.JsValue,
    name: c_str, fallback: usize
): usize {
    if optional_i64_property(ctx, object, name) is value {
        if value > 0 {
            return value as usize;
        }
    }
    return fallback;
}

fn compile_level_property(ctx: *c_void, object: js.JsValue
): ?i64 {
    return optional_i64_property(ctx, object, "optimization_level");
}

fn optimization_goal_property(ctx: *c_void, object: js.JsValue
): ?i64 {
    return optional_i64_property(ctx, object, "optimization_goal");
}

fn numerical_policy_property(ctx: *c_void, object: js.JsValue
): ?i64 {
    return optional_i64_property(ctx, object, "numerical_policy");
}

fn explanation_level_property(ctx: *c_void, object: js.JsValue
): ?i64 {
    return optional_i64_property(ctx, object, "explanation_level");
}

fn residual_policy_property(ctx: *c_void, object: js.JsValue
): ?i64 {
    return optional_i64_property(ctx, object, "residual_policy");
}

fn decode_compile_options(ctx: *c_void, value: js.JsValue,
    output: &graph.CompileOptionsInput
): bool {
    if js.is_undefined(value) {
        const defaults: graph.CompileOptionsInput = {
            optimization_level: CompileOptimizationLevel.safe,
            optimization_goal: OptimizationGoal.balanced,
            numerical_policy: NumericalPolicy.backend_equivalent,
            explanation_level: ExplanationLevel.summary,
            dense_inputs: false,
            residual_policy: ResidualPolicy.retain,
            search_limits: default_search_limits()
        };
        output.optimization_level = defaults.optimization_level;
        output.optimization_goal = defaults.optimization_goal;
        output.numerical_policy = defaults.numerical_policy;
        output.explanation_level = defaults.explanation_level;
        output.dense_inputs = false;
        output.residual_policy = defaults.residual_policy;
        output.search_limits = defaults.search_limits;
        return true;
    }
    const version = optional_i64_property(ctx, value, "schema_version");
    if version is selected_version {
        if selected_version != 1 {
            return false;
        }
    } else {
        return false;
    }
    const level_value = compile_level_property(ctx, value);
    const goal_value = optimization_goal_property(ctx, value);
    const policy_value = numerical_policy_property(ctx, value);
    const explanation_value = explanation_level_property(ctx, value);
    const residual_value = residual_policy_property(ctx, value);
    if level_value is none {
        return false;
    }
    if goal_value is none {
        return false;
    }
    if policy_value is none {
        return false;
    }
    if explanation_value is none {
        return false;
    }
    if residual_value is none {
        return false;
    }
    const level = decode_compile_level(level_value);
    const goal = decode_optimization_goal(goal_value);
    const policy = decode_numerical_policy(policy_value);
    const explanation = decode_explanation_level(explanation_value);
    const residual = decode_residual_policy(residual_value);
    if level is none {
        return false;
    }
    if goal is none {
        return false;
    }
    if policy is none {
        return false;
    }
    if explanation is none {
        return false;
    }
    if residual is none {
        return false;
    }
    const dense_inputs =
    optional_flag_property(ctx, value, "dense_inputs", false);
    if dense_inputs is none {
        return false;
    }
    var limits = default_search_limits();
    limits.structural_states = positive_usize_property(
        ctx,
        value,
        "structural_states",
        limits.structural_states
    );
    limits.structural_expansions = positive_usize_property(
        ctx,
        value,
        "structural_expansions",
        limits.structural_expansions
    );
    limits.complete_evaluations = positive_usize_property(
        ctx,
        value,
        "complete_evaluations",
        limits.complete_evaluations
    );
    limits.physical_work = positive_usize_property(
        ctx,
        value,
        "physical_work",
        limits.physical_work
    );
    limits.ready_work = positive_usize_property(
        ctx,
        value,
        "ready_work",
        limits.ready_work
    );
    limits.max_peak_active_bytes = optional_usize_property(
        ctx,
        value,
        "max_peak_active_bytes"
    );
    limits.max_invocation_scratch_bytes = optional_usize_property(
        ctx,
        value,
        "max_invocation_scratch_bytes"
    );
    output.optimization_level = level;
    output.optimization_goal = goal;
    output.numerical_policy = policy;
    output.explanation_level = explanation;
    output.dense_inputs = dense_inputs;
    output.residual_policy = residual;
    output.search_limits = limits;
    return true;
}

fn decode_program(ctx: *c_void, value: js.JsValue,
    output: &graph.ProgramInput
): bool {
    const version = optional_i64_property(ctx, value, "schema_version");
    if version is selected_version {
        if selected_version != 2 {
            return false;
        }
    } else {
        return false;
    }
    const kind_value = optional_i64_property(ctx, value, "kind");
    if kind_value is none {
        return false;
    }
    const kind = decode_program_kind(kind_value);
    if kind is none {
        return false;
    }
    const nodes_value = js.get_property_str(ctx, value, "nodes");
    if js.is_array(nodes_value) == 0 {
        js.free_value(ctx, nodes_value);
        return false;
    }
    const length_value = property_int(ctx, nodes_value, "length");
    var node_length: usize = 0;
    if length_value is selected_length {
        if selected_length < 0 or selected_length > 1_048_576 {
            js.free_value(ctx, nodes_value);
            return false;
        }
        node_length = selected_length as usize;
    } else {
        js.free_value(ctx, nodes_value);
        return false;
    }
    var nodes: List<graph.ProgramNodeInput> =
    list<graph.ProgramNodeInput>();
    var index: usize = 0;
    while index < node_length {
        const node_value = js.get_property_uint32(
            ctx,
            nodes_value,
            index as c_uint
        );
        var decoded_node = default_program_node_input();
        const valid_node = decode_program_node(
            ctx,
            node_value,
            index,
            &decoded_node
        );
        js.free_value(ctx, node_value);
        if valid_node {
            nodes.push(decoded_node);
        } else {
            js.free_value(ctx, nodes_value);
            return false;
        }
        index = index + 1;
    }
    js.free_value(ctx, nodes_value);
    output.kind = kind;
    output.nodes = nodes;
    output.outputs = usize_array_property(ctx, value, "outputs");
    return true;
}

fn create_session(ctx: *c_void, _: js.JsValue, argument_count: c_int,
    arguments_pointer: *const js.JsValue
): js.JsValue {
    if argument_count < 1 {
        return js.type_error(ctx, "createSession expects a device");
    }
    const requested_device = js.read_string(ctx, argument(arguments_pointer, 0));
    var selected_device = "";
    var uses_cuda = false;
    var device_ordinal: u32 = 0;
    if requested_device is text {
        selected_device = text;
        if text == "cuda" {
            uses_cuda = true;
        } else if text.starts_with("cuda:") {
            const bytes = text.bytes();
            const suffix = string.from(bytes[5..]);
            const parsed = parse_decimal_usize(suffix);
            if parsed is ordinal {
                if ordinal <= 4_294_967_295 {
                    uses_cuda = true;
                    device_ordinal = ordinal as u32;
                } else {
                    return js.type_error(ctx, "unsupported device");
                }
            } else {
                return js.type_error(ctx, "unsupported device");
            }
        } else if text != "cpu" {
            return js.type_error_string(
                ctx,
                $"backend unavailable: {text}"
            );
        }
    } else {
        return js.type_error(ctx, "device must be a string");
    }
    var selected_session: ?usize = none;
    var session_index: usize = 0;
    while session_index < maximum_sessions {
        if not sessions[session_index].alive {
            selected_session = session_index;
            break;
        }
        session_index = session_index + 1;
    }
    if selected_session is none {
        return js.type_error(ctx, "Session capacity exhausted");
    }
    var capture_enabled = false;
    var capture_capacity: usize = 0;
    var timing_scope = "none";
    var hardware_metrics = false;
    var profiler_markers = false;
    if argument_count >= 2 {
        const options = argument(arguments_pointer, 1);
        const version = optional_i64_property(ctx, options, "schema_version");
        if version is selected_version {
            if selected_version != 1 {
                return js.type_error(ctx, "invalid Session options");
            }
        } else {
            return js.type_error(ctx, "invalid Session options");
        }
        const enabled = optional_flag_property(
            ctx,
            options,
            "telemetry_enabled",
            false
        );
        if enabled is none {
            return js.type_error(ctx, "invalid Session options");
        }
        if enabled {
            if optional_usize_property(
                ctx,
                options,
                "capture_records"
            ) is records {
                if records == 0 or records > 4_194_304 {
                    return js.type_error(
                        ctx,
                        "captureRecords must be between 1 and 4194304"
                    );
                }
                capture_enabled = true;
                capture_capacity = records;
                timing_scope = "capture";
            }
            if optional_usize_property(
                ctx,
                options,
                "compute_timing"
            ) is selected_timing {
                if selected_timing == 1 {
                    timing_scope = "invocation";
                } else if selected_timing == 2 {
                    timing_scope = "kernel";
                } else if selected_timing == 0 and not capture_enabled {
                    timing_scope = "none";
                } else if selected_timing != 0 {
                    return js.type_error(
                        ctx,
                        "invalid computeTiming option"
                    );
                }
            }
            if optional_flag_property(
                ctx,
                options,
                "hardware_metrics",
                false
            ) is selected_hardware {
                hardware_metrics = selected_hardware;
            } else {
                return js.type_error(ctx, "invalid Session options");
            }
            if optional_flag_property(
                ctx,
                options,
                "profiler_markers",
                false
            ) is selected_markers {
                profiler_markers = selected_markers;
            } else {
                return js.type_error(ctx, "invalid Session options");
            }
        }
    }
    const handle = selected_session;
    var context_address: usize = 0;
    if uses_cuda {
        const context = compute.allocate_cuda_context(device_ordinal)
        catch (error_code) {
            const ignored = error_code;
            compute.report_cuda_error();
            return js.type_error_string(
                ctx,
                $"backend unavailable: {selected_device}"
            );
        };
        unchecked {
            context_address = context as! usize;
        }
    } else {
        const context = compute.allocate_cpu_context() catch (error_code) {
            const ignored = error_code;
            return js.type_error_string(
                ctx,
                $"backend unavailable: {selected_device}"
            );
        };
        unchecked {
            context_address = context as! usize;
        }
    }
    sessions[handle] = SessionSlot {
        generation: generation(),
        alive: true,
        context_address: context_address,
        cuda: uses_cuda,
        device: selected_device,
        capture_enabled: capture_enabled,
        capture_capacity: capture_capacity,
        timing_scope: timing_scope,
        hardware_metrics: hardware_metrics,
        profiler_markers: profiler_markers
    };
    const object = js.new_object(ctx);
    js.set_number(ctx, object, "__affon_handle", handle as f64);
    js.set_string(ctx, object, "device", sessions[handle].device);
    js.set_function(ctx, object, "dispose", session_dispose, 0);
    return object;
}

fn live_session(ctx: *c_void, value: js.JsValue): ?usize {
    if object_handle(ctx, value) is handle {
        if handle < maximum_sessions {
            if sessions[handle].alive {
                return handle;
            }
        }
    }
    return none;
}

fn session_dispose(ctx: *c_void, this_value: js.JsValue,
    argument_count: c_int, arguments_pointer: *const js.JsValue
): js.JsValue {
    const ignored_count = argument_count;
    const ignored_arguments = arguments_pointer;
    if object_handle(ctx, this_value) is handle {
        if handle < maximum_sessions {
            const current = sessions[handle];
            sessions[handle] = SessionSlot {
                generation: current.generation,
                alive: false,
                context_address: 0,
                cuda: false,
                device: "cpu",
                capture_enabled: false,
                capture_capacity: 0,
                timing_scope: "none",
                hardware_metrics: false,
                profiler_markers: false
            };
            var index: usize = 0;
            while index < next_tensor {
                if tensor_alive[index] {
                    if tensor_sessions[index] == handle {
                        // Detached tensors remain readable after their owner
                        // Session is released. The caller still owns each
                        // tensor and can dispose it explicitly.
                        tensor_sessions[index] = maximum_sessions;
                    }
                }
                index = index + 1;
            }
            index = 0;
            while index < next_executable {
                if executable_alive[index] {
                    if executable_sessions[index] == handle {
                        executable_alive[index] = false;
                        const pointer = executable_pointers[index];
                        executable_pointers[index] = 0;
                        if pointer != 0 {
                            dispose_executable_slot(
                                pointer as! *ExecutableSlot
                            );
                        }
                    }
                }
                index = index + 1;
            }
            if current.context_address != 0 {
                const context = context_pointer(current.context_address);
                if current.cuda {
                    compute.release_cuda_context(
                        context as! *compute.CudaContext
                    );
                } else {
                    compute.release_cpu_context(
                        context as! *compute.CpuContext
                    );
                }
            }
        }
    }
    return js.undefined();
}

fn session_tensor(ctx: *c_void, _: js.JsValue, argument_count: c_int,
    arguments_pointer: *const js.JsValue
): js.JsValue {
    if argument_count < 4 {
        return js.type_error(ctx, "sessionTensor expects Session, values, dtype, and shape");
    }
    const session = live_session(ctx, argument(arguments_pointer, 0));
    if session is none {
        return js.type_error(ctx, "invalid Session");
    }
    const values = argument(arguments_pointer, 1);
    const dtype_text = js.read_string(ctx, argument(arguments_pointer, 2));
    if dtype_text is none {
        return js.type_error(ctx, "dtype must be a string");
    }
    const selected_dtype = dtypes(dtype_text);
    if selected_dtype is none {
        return js.type_error(ctx, "unsupported dtype");
    }
    const dimensions = shape(ctx, argument(arguments_pointer, 3));
    if dimensions is none {
        return js.type_error(ctx, "shape must be an array");
    }
    const count = element_count(dimensions[..]);
    if count is none {
        return js.type_error(ctx, "invalid Tensor shape");
    }
    var selected_values: List<f64> = list<f64>();
    if not flatten_tensor_values(ctx, values, &selected_values) {
        return js.type_error(ctx, "tensor expects a rectangular numeric value");
    }
    if selected_values.len() != count {
        return js.type_error(ctx, "tensor value count does not match shape");
    }
    const value = tensor_from_numbers(
        ctx,
        selected_values[..],
        selected_dtype,
        dimensions[..]
    )
    catch (error_code) {
        const ignored = error_code;
        return js.type_error(ctx, "tensor creation failed");
    };
    return create_tensor_object(
        ctx,
        value,
        session,
        session_generation(session),
        true
    );
}

fn flatten_tensor_values(ctx: *c_void, value: js.JsValue,
    data: &List<f64>
): bool {
    if js.is_array(value) == 0 {
        if js.read_number(ctx, value) is selected {
            data.push(selected);
            return true;
        }
        return false;
    }
    const count = property_int(ctx, value, "length");
    if count is length {
        if length < 0 {
            return false;
        }
        var index: usize = 0;
        while index < length as usize {
            const item = js.get_property_uint32(ctx, value, index as c_uint);
            defer js.free_value(ctx, item);
            if not flatten_tensor_values(ctx, item, data) {
                return false;
            }
            index = index + 1;
        }
        return true;
    }
    return false;
}

fn session_tensor_bytes(ctx: *c_void, _: js.JsValue, argument_count: c_int,
    arguments_pointer: *const js.JsValue
): js.JsValue {
    if argument_count < 4 {
        return js.type_error(ctx, "sessionTensorBytes expects Session, bytes, dtype, and shape");
    }
    const session = live_session(ctx, argument(arguments_pointer, 0));
    if session is none {
        return js.type_error(ctx, "invalid Session");
    }
    var byte_count_storage: [1]c_size_t = [0];
    const bytes = js.get_array_buffer(
        ctx, byte_count_storage[..].ptr, argument(arguments_pointer, 1)
    );
    if bytes is none {
        return js.type_error(ctx, "tensor bytes must be an ArrayBuffer");
    }
    const byte_count = byte_count_storage[0];
    const dtype_text = js.read_string(ctx, argument(arguments_pointer, 2));
    if dtype_text is none {
        return js.type_error(ctx, "dtype must be a string");
    }
    const selected_dtype = dtypes(dtype_text);
    if selected_dtype is none {
        return js.type_error(ctx, "unsupported dtype");
    }
    const dimensions = shape(ctx, argument(arguments_pointer, 3));
    if dimensions is none {
        return js.type_error(ctx, "shape must be an array");
    }
    const tensor_spec = spec(selected_dtype, dimensions[..])
    catch (error_code) {
        const ignored = error_code;
        return js.type_error(ctx, "invalid Tensor shape");
    };
    const count = tensor_spec.element_count() catch (error_code) {
        const ignored = error_code;
        return js.type_error(ctx, "invalid Tensor shape");
    };
    const width = graph.dtype_width(tensor_spec.dtype);
    if count * width != byte_count as usize {
        return js.type_error(ctx, "buffer byte length does not match shape and dtype");
    }
    var copied: List<u8> = list<u8>();
    var index: usize = 0;
    while index < byte_count as usize {
        unchecked {
            copied.push(*(bytes + index) as u8);
        }
        index = index + 1;
    }
    const value = compute.tensor_bytes(tensor_spec, copied[..]) catch (error_code) {
        const ignored = error_code;
        return js.type_error(ctx, "tensor creation failed");
    };
    return create_tensor_object(
        ctx,
        value,
        session,
        session_generation(session),
        true
    );
}

fn session_tensor_copy(ctx: *c_void, _: js.JsValue, argument_count: c_int,
    arguments_pointer: *const js.JsValue
): js.JsValue {
    if argument_count < 2 {
        return js.type_error(ctx, "sessionTensorCopy expects Session and Tensor");
    }
    const session = live_session(ctx, argument(arguments_pointer, 0));
    if session is none {
        return js.type_error(ctx, "invalid Session");
    }
    const source = live_tensor(ctx, argument(arguments_pointer, 1));
    if source is none {
        return js.type_error(ctx, "source must be a Tensor");
    }
    return create_tensor_object(
        ctx,
        compute.clone_tensor(tensor_value(source)),
        session,
        session_generation(session),
        tensor_dense(source as usize)
    );
}

fn session_full(ctx: *c_void, _: js.JsValue, argument_count: c_int,
    arguments_pointer: *const js.JsValue
): js.JsValue {
    if argument_count < 4 {
        return js.type_error(ctx, "sessionFull expects Session, shape, value, and dtype");
    }
    const session = live_session(ctx, argument(arguments_pointer, 0));
    if session is none {
        return js.type_error(ctx, "invalid Session");
    }
    const dimensions = shape(ctx, argument(arguments_pointer, 1));
    if dimensions is none {
        return js.type_error(ctx, "shape must be an array");
    }
    const fill = js.read_number(ctx, argument(arguments_pointer, 2));
    if fill is none {
        return js.type_error(ctx, "fill value must be a number");
    }
    const dtype_text = js.read_string(ctx, argument(arguments_pointer, 3));
    if dtype_text is none {
        return js.type_error(ctx, "unsupported dtype");
    }
    const selected_dtype = dtypes(dtype_text);
    if selected_dtype is none {
        return js.type_error(ctx, "unsupported dtype");
    }
    const tensor_spec = spec(selected_dtype, dimensions[..])
    catch (error_code) {
        const ignored = error_code;
        return js.type_error(ctx, "invalid Tensor shape");
    };
    const count = tensor_spec.element_count() catch (error_code) {
        const ignored = error_code;
        return js.type_error(ctx, "invalid Tensor shape");
    };
    var data: List<f64> = list<f64>();
    var index: usize = 0;
    while index < count {
        data.push(fill as f64);
        index = index + 1;
    }
    const value = tensor_from_numbers(
        ctx, data[..], tensor_spec.dtype, dimensions[..]
    )
    catch (error_code) {
        const ignored = error_code;
        return js.type_error(ctx, "tensor creation failed");
    };
    return create_tensor_object(
        ctx,
        value,
        session,
        session_generation(session),
        true
    );
}

fn tensor_from_numbers(ctx: *c_void, values: []const f64,
    selected_dtype: DType, dimensions: []const usize
): compute.Tensor ! compute.ComputeError {
    const ignored_context = ctx;
    const tensor_spec = try spec(selected_dtype, dimensions);
    if selected_dtype == DType.f32 {
        var data: List<f32> = list<f32>();
        for value in values {
            data.push(value as f32);
        }
        return try compute.tensor(tensor_spec, data[..]);
    }
    if selected_dtype == DType.f64 {
        return try compute.tensor_f64(tensor_spec, values);
    }
    if selected_dtype == DType.u32 {
        var data: List<u32> = list<u32>();
        for value in values {
            data.push(value as u32);
        }
        return try compute.tensor_u32(tensor_spec, data[..]);
    }
    if selected_dtype == DType.i64 {
        var data: List<i64> = list<i64>();
        for value in values {
            data.push(value as i64);
        }
        return try compute.tensor_i64(tensor_spec, data[..]);
    }
    var data: List<u16> = list<u16>();
    for value in values {
        data.push(bfloat.from_f32(numeric.round_f32(value)));
    }
    return try compute.tensor_bf16(tensor_spec, data[..]);
}

fn compile_program(ctx: *c_void, _: js.JsValue, argument_count: c_int,
    arguments_pointer: *const js.JsValue
): js.JsValue {
    if argument_count < 2 {
        return js.type_error(ctx, "compileProgram expects Session and Program");
    }
    const session = live_session(ctx, argument(arguments_pointer, 0));
    if session is none {
        return js.type_error(ctx, "invalid Session");
    }
    var program = default_program_input();
    if not decode_program(
        ctx,
        argument(arguments_pointer, 1),
        &program
    ) {
        return js.type_error(ctx, "invalid Program descriptor");
    }
    var options_value = js.undefined();
    if argument_count >= 3 {
        options_value = argument(arguments_pointer, 2);
    }
    var options = default_compile_options_input();
    if not decode_compile_options(ctx, options_value, &options) {
        return js.type_error(ctx, "invalid compile options descriptor");
    }
    const backend = session_backend(session);
    const compiled = graph.compile_program(&program, &options, &backend)
    catch (error_code) {
        const failure = error_name(error_code);
        if sessions[session].cuda {
            compute.report_cuda_error();
        }
        return js.type_error_string(
            ctx,
            $"Cyna lowering failed: {failure}"
        );
    };
    if next_executable >= maximum_executables {
        return js.type_error(ctx, "Executable capacity exhausted");
    }
    const handle = next_executable;
    next_executable = next_executable + 1;
    const executable_generation = generation();
    const explanation = plan.explanation(
        &compiled.executable,
        &compiled.source_instruction_by_step
    );
    const owner = sessions[session];
    const box = new ExecutableBox {
        value: compiled
    };
    const slot = new ExecutableSlot {
        value: resource.leak(box),
        session: session,
        generation: executable_generation,
        alive: true,
        explanation: explanation,
        routing: routing_json(sessions[session].device),
        run_count: 0,
        failed_run_count: 0,
        cache_hits: 0,
        cache_misses: 0,
        captured_count: 0,
        dropped_count: 0,
        elapsed_ns: 0,
        capture_enabled: owner.capture_enabled,
        capture_capacity: owner.capture_capacity,
        timing_scope: owner.timing_scope,
        hardware_metrics: owner.hardware_metrics,
        profiler_markers: owner.profiler_markers
    };
    executable_pointers[handle] = resource.leak(slot) as! usize;
    executable_sessions[handle] = session;
    executable_alive[handle] = true;
    const object = js.new_object(ctx);
    js.set_number(ctx, object, "__affon_handle", handle as f64);
    js.set_number(ctx, object, "__affon_generation",
        executable_generation as f64
    );
    js.set_function(ctx, object, "dispose", executable_dispose, 0);
    return object;
}

fn live_executable(ctx: *c_void, value: js.JsValue): ?usize {
    if object_handle(ctx, value) is handle {
        if handle < maximum_executables {
            if executable_alive[handle] {
                return handle;
            }
        }
    }
    return none;
}

fn executable_dispose(ctx: *c_void, this_value: js.JsValue,
    argument_count: c_int, arguments_pointer: *const js.JsValue
): js.JsValue {
    const ignored_count = argument_count;
    const ignored_arguments = arguments_pointer;
    if object_handle(ctx, this_value) is handle {
        if handle < maximum_executables {
            if executable_alive[handle] {
                executable_alive[handle] = false;
                const pointer = executable_pointers[handle];
                executable_pointers[handle] = 0;
                if pointer != 0 {
                    dispose_executable_slot(pointer as! *ExecutableSlot);
                }
            }
        }
    }
    return js.undefined();
}

fn executable_json(ctx: *c_void, arguments_pointer: *const js.JsValue,
    argument_count: c_int, explanation: bool
): js.JsValue {
    if argument_count < 1 {
        return js.type_error(ctx, "Executable argument is required");
    }
    const handle = live_executable(ctx, argument(arguments_pointer, 0));
    if handle is none {
        return js.type_error(ctx, "invalid Executable");
    }
    const slot = executable_slot(handle);
    if explanation {
        const compiled = executable_value(handle);
        return js.string(ctx, plan.explanation(
            &compiled.executable,
            &compiled.source_instruction_by_step
        ));
    }
    unchecked {
        return js.string(ctx, (*slot).routing);
    }
}

fn explain_executable(ctx: *c_void, _: js.JsValue, argument_count: c_int,
    arguments_pointer: *const js.JsValue
): js.JsValue {
    return executable_json(ctx, arguments_pointer, argument_count, true);
}

fn executable_routing(ctx: *c_void, _: js.JsValue, argument_count: c_int,
    arguments_pointer: *const js.JsValue
): js.JsValue {
    return executable_json(ctx, arguments_pointer, argument_count, false);
}

fn executable_backend(ctx: *c_void, _: js.JsValue, argument_count: c_int,
    arguments_pointer: *const js.JsValue
): js.JsValue {
    if argument_count < 1 or live_executable(
        ctx, argument(arguments_pointer, 0)
    ) is none {
        return js.type_error(ctx, "invalid Executable");
    }
    return js.string(ctx, "cyna");
}

fn executable_plan_json(ctx: *c_void, argument_count: c_int,
    arguments_pointer: *const js.JsValue, kind: i32
): js.JsValue {
    if argument_count < 1 {
        return js.type_error(ctx, "Executable argument is required");
    }
    const handle = live_executable(ctx, argument(arguments_pointer, 0));
    if handle is none {
        return js.type_error(ctx, "invalid Executable");
    }
    const compiled = executable_value(handle);
    const text = switch kind {
        1 => plan.selected_steps(&compiled.executable);
        2 => plan.selected_physical_nodes(
            &compiled.executable,
            &compiled.source_instruction_by_step
        );
        3 => plan.selected_physical_values(&compiled.executable);
        4 => plan.selected_physical_liveness(&compiled.executable);
        5 => plan.residual_decisions(&compiled.executable);
        6 => plan.selection_comparison(&compiled.executable);
        _ => plan.search(&compiled.executable);
    };
    return js.string(ctx, text);
}

fn selected_steps(ctx: *c_void, _: js.JsValue, argument_count: c_int,
    arguments_pointer: *const js.JsValue
): js.JsValue {
    return executable_plan_json(ctx, argument_count, arguments_pointer, 1);
}

fn selected_physical_nodes(ctx: *c_void, _: js.JsValue,
    argument_count: c_int, arguments_pointer: *const js.JsValue
): js.JsValue {
    return executable_plan_json(ctx, argument_count, arguments_pointer, 2);
}

fn selected_physical_values(ctx: *c_void, _: js.JsValue,
    argument_count: c_int, arguments_pointer: *const js.JsValue
): js.JsValue {
    return executable_plan_json(ctx, argument_count, arguments_pointer, 3);
}

fn selected_physical_liveness(ctx: *c_void, _: js.JsValue,
    argument_count: c_int, arguments_pointer: *const js.JsValue
): js.JsValue {
    return executable_plan_json(ctx, argument_count, arguments_pointer, 4);
}

fn residual_decisions(ctx: *c_void, _: js.JsValue, argument_count: c_int,
    arguments_pointer: *const js.JsValue
): js.JsValue {
    return executable_plan_json(ctx, argument_count, arguments_pointer, 5);
}

fn selection_comparison(ctx: *c_void, _: js.JsValue, argument_count: c_int,
    arguments_pointer: *const js.JsValue
): js.JsValue {
    return executable_plan_json(ctx, argument_count, arguments_pointer, 6);
}

fn search_evidence(ctx: *c_void, _: js.JsValue, argument_count: c_int,
    arguments_pointer: *const js.JsValue
): js.JsValue {
    return executable_plan_json(ctx, argument_count, arguments_pointer, 7);
}

fn dense(handle: usize): bool {
    return tensor_dense(handle);
}

fn output_is_view(executable: &const compute.Executable,
    output_index: usize
): bool {
    if output_index >= executable.graph.output_bindings.len() {
        return false;
    }
    const value_id = executable.graph.output_bindings[output_index];
    if value_id >= executable.graph.values.len() {
        return false;
    }
    return executable.graph.values[value_id].alias_of != none;
}

fn record_capture(executable: *ExecutableSlot, node_count: usize): void {
    unchecked {
        if not (*executable).capture_enabled {
            return;
        }
        var attempts: usize = node_count;
        if attempts <= ((0 as usize) -% 1) - 2 {
            attempts = attempts + 2;
        }
        const available =
        (*executable).capture_capacity - (*executable).captured_count;
        if attempts <= available {
            (*executable).captured_count =
            (*executable).captured_count + attempts;
        } else {
            (*executable).captured_count =
            (*executable).capture_capacity;
            (*executable).dropped_count =
            (*executable).dropped_count + (attempts - available);
        }
    }
}

fn runtime_evidence(executable: *ExecutableSlot): plan.RuntimeEvidence {
    const empty: plan.RuntimeEvidence = {
        run_count: 0,
        failed_run_count: 0,
        cache_hits: 0,
        cache_misses: 0,
        captured_count: 0,
        capture_capacity: 0,
        dropped_count: 0,
        timing_scope: "none",
        elapsed_ns: 0,
        hardware_metrics: false,
        profiler_markers: false
    };
    var selected = empty;
    unchecked {
        selected.run_count = (*executable).run_count;
        selected.failed_run_count = (*executable).failed_run_count;
        selected.cache_hits = (*executable).cache_hits;
        selected.cache_misses = (*executable).cache_misses;
        selected.captured_count = (*executable).captured_count;
        selected.capture_capacity = (*executable).capture_capacity;
        selected.dropped_count = (*executable).dropped_count;
        selected.timing_scope = (*executable).timing_scope;
        selected.elapsed_ns = (*executable).elapsed_ns;
        selected.hardware_metrics = (*executable).hardware_metrics;
        selected.profiler_markers = (*executable).profiler_markers;
    }
    return selected;
}

fn empty_outputs(ctx: *c_void,
    executable: &const compute.Executable
): ?List<compute.Tensor> {
    if executable.graph.output_bindings.len() == 0 {
        return none;
    }
    var outputs: List<compute.Tensor> = list<compute.Tensor>();
    for output in executable.graph.output_bindings[..] {
        if output >= executable.graph.values.len() {
            return none;
        }
        const value = &executable.graph.values[output];
        const count = value.spec.element_count() catch (error_code) {
            const ignored = error_code;
            return none;
        };
        if count != 0 {
            return none;
        }
        const values: [0]f64 = [];
        const tensor = tensor_from_numbers(
            ctx,
            values[..],
            value.spec.dtype,
            value.spec.dimensions[..]
        ) catch (error_code) {
            const ignored = error_code;
            return none;
        };
        outputs.push(tensor);
    }
    return outputs;
}

fn run_executable(ctx: *c_void, _: js.JsValue, argument_count: c_int,
    arguments_pointer: *const js.JsValue
): js.JsValue {
    if argument_count < 2 {
        return js.type_error(ctx, "runExecutable expects Executable and inputs");
    }
    const executable_handle = live_executable(
        ctx, argument(arguments_pointer, 0)
    );
    if executable_handle is none {
        return js.type_error(ctx, "invalid Executable");
    }
    const inputs = argument(arguments_pointer, 1);
    if js.is_array(inputs) == 0 {
        return js.type_error(ctx, "inputs must be an array");
    }
    const count = property_int(ctx, inputs, "length");
    if count is none {
        return js.type_error(ctx, "invalid inputs");
    }
    if count < 0 {
        return js.type_error(ctx, "invalid inputs");
    }
    const executable = executable_slot(executable_handle);
    const compiled = executable_value(executable_handle);
    const node_count = compiled.executable.graph.nodes.len();
    var owned: List<compute.Tensor> = list<compute.Tensor>();
    var index: usize = 0;
    while index < count as usize {
        const item = js.get_property_uint32(ctx, inputs, index as c_uint);
        defer js.free_value(ctx, item);
        if live_tensor(ctx, item) is handle {
            const source = tensor_value(handle);
            if compiled.dense_inputs and not dense(handle) {
                return js.type_error(
                    ctx,
                    "Program execution failed: GuardMismatch"
                );
            }
            owned.push(compute.clone_tensor(source));
        } else {
            return js.type_error(ctx, "inputs must be Tensors");
        }
        index = index + 1;
    }
    if compiled.needs_seed {
        if owned.len() + 1 != compiled.executable.input_guards.len() {
            return js.type_error(ctx, "Program execution failed: InvalidCommand");
        }
        const guard_index = owned.len();
        const guard = &compiled.executable.input_guards[guard_index];
        if guard.spec.dtype == DType.i64 or guard.spec.dtype == DType.u32 {
            return js.type_error(
                ctx,
                "gradient loss must use a floating dtype"
            );
        }
        const seed_values: [1]f64 = [1.0];
        const seed = tensor_from_numbers(
            ctx,
            seed_values[..],
            guard.spec.dtype,
            guard.spec.dimensions[..]
        ) catch (error_code) {
            const ignored = error_code;
            return js.type_error(ctx, "seed creation failed");
        };
        owned.push(seed);
    }
    const started = now_ns();
    var outputs: List<compute.Tensor> = list<compute.Tensor>();
    if empty_outputs(ctx, &compiled.executable) is empty {
        outputs = empty;
    } else {
        const backend = session_backend(executable_sessions[executable_handle]);
        outputs = compute.run_with_backend(
            &backend,
            &compiled.executable,
            owned[..]
        ) catch (error_code) {
            if executable_sessions[executable_handle] < maximum_sessions and
            sessions[executable_sessions[executable_handle]].cuda {
                compute.report_cuda_error();
            }
            const failure = error_name(error_code);
            const elapsed = now_ns() - started;
            unchecked {
                (*executable).run_count = (*executable).run_count + 1;
                (*executable).failed_run_count =
                (*executable).failed_run_count + 1;
                (*executable).elapsed_ns = (*executable).elapsed_ns + elapsed;
                record_capture(executable, node_count);
            }
            add_metric(Metric {
                group: "compute.execution",
                name: "runs_total",
                kind: MetricKind.counter,
                unit: MetricUnit.count
            }, 1);
            add_metric(Metric {
                group: "compute.backend",
                name: "elapsed_nanoseconds_total",
                kind: MetricKind.counter,
                unit: MetricUnit.nanoseconds
            }, elapsed as i64);
            // The public run contract reports a stable command failure while the
            // project diagnostics retain the precise Compute Core error name.
            return js.type_error_string(
                ctx,
                $"Program execution failed: InvalidCommand ({failure})"
            );
        };
    }
    const elapsed = now_ns() - started;
    var timing_scope = "none";
    var capture_enabled = false;
    unchecked {
        (*executable).run_count = (*executable).run_count + 1;
        if (*executable).run_count == 1 {
            (*executable).cache_misses = (*executable).cache_misses + 1;
        } else {
            (*executable).cache_hits = (*executable).cache_hits + 1;
        }
        (*executable).elapsed_ns = (*executable).elapsed_ns + elapsed;
        record_capture(executable, node_count);
        timing_scope = (*executable).timing_scope;
        capture_enabled = (*executable).capture_enabled;
    }
    add_metric(Metric {
        group: "compute.execution",
        name: "runs_total",
        kind: MetricKind.counter,
        unit: MetricUnit.count
    }, 1);
    if timing_scope != "none" or capture_enabled {
        add_metric(Metric {
            group: "compute.backend",
            name: "elapsed_nanoseconds_total",
            kind: MetricKind.counter,
            unit: MetricUnit.nanoseconds
        }, elapsed as i64);
    }
    const result = js.new_array(ctx);
    index = 0;
    while index < outputs.len() {
        unchecked {
            const output_session = (*executable).session;
            const ignored_output = js.set_property_uint32(ctx, result,
                index as c_uint,
                create_tensor_object(
                    ctx,
                    compute.clone_tensor(&outputs[index]),
                    output_session,
                    session_generation(output_session),
                    not output_is_view(&compiled.executable, index)
                )
            );
            const ignored = ignored_output;
        }
        index = index + 1;
    }
    return result;
}

fn default_device(ctx: *c_void, _: js.JsValue, argument_count: c_int,
    arguments_pointer: *const js.JsValue
): js.JsValue {
    const ignored_count = argument_count;
    const ignored_arguments = arguments_pointer;
    if env_get("AFFON_DEVICE") is device {
        return js.string(ctx, device);
    }
    if compute.cuda_available() {
        return js.string(ctx, "cuda");
    }
    return js.string(ctx, "cpu");
}

fn reset_evidence(ctx: *c_void, _: js.JsValue, argument_count: c_int,
    arguments_pointer: *const js.JsValue
): js.JsValue {
    if argument_count < 1 {
        return js.type_error(ctx, "resetEvidenceCapture expects Session");
    }
    const session_handle = live_session(ctx, argument(arguments_pointer, 0));
    if session_handle is none {
        return js.type_error(ctx, "invalid Session");
    }
    if not sessions[session_handle].capture_enabled {
        return js.type_error(
            ctx,
            "runtime evidence capture is not enabled"
        );
    }
    var index: usize = 0;
    while index < next_executable {
        if executable_alive[index] and
        executable_sessions[index] == session_handle {
            const executable = executable_slot(index);
            unchecked {
                (*executable).run_count = 0;
                (*executable).failed_run_count = 0;
                (*executable).captured_count = 0;
                (*executable).dropped_count = 0;
                (*executable).elapsed_ns = 0;
            }
        }
        index = index + 1;
    }
    return js.undefined();
}

fn export_evidence(ctx: *c_void, _: js.JsValue, argument_count: c_int,
    arguments_pointer: *const js.JsValue
): js.JsValue {
    if argument_count < 2 {
        return js.type_error(
            ctx,
            "exportEvidenceCapture expects Session and Executable"
        );
    }
    const session_handle = live_session(ctx, argument(arguments_pointer, 0));
    if session_handle is none {
        return js.type_error(ctx, "invalid Session");
    }
    const executable_handle = live_executable(
        ctx,
        argument(arguments_pointer, 1)
    );
    if executable_handle is none {
        return js.type_error(ctx, "invalid Executable");
    }
    if not sessions[session_handle].capture_enabled {
        return js.type_error(
            ctx,
            "runtime evidence capture is not enabled"
        );
    }
    const executable = executable_slot(executable_handle);
    unchecked {
        if (*executable).session != session_handle {
            return js.type_error(
                ctx,
                "Executable belongs to a different Session"
            );
        }
    }
    const compiled = executable_value(executable_handle);
    const evidence = runtime_evidence(executable);
    const serialized = plan.evidence(
        &compiled.executable,
        &compiled.source_instruction_by_step,
        evidence,
        session_evidence_backend(session_handle)
    );
    return js.string(ctx, serialized);
}

fn positive_usize(ctx: *c_void, value: js.JsValue): ?usize {
    if js.read_int(ctx, value) is selected {
        if selected > 0 {
            return selected as usize;
        }
    }
    return none;
}

fn fft_run(workspace: &List<f64>, n: usize, input_offset: usize,
    input_stride: usize, size: usize, output: usize
): void {
    const real_base: usize = 0;
    const imaginary_base = n;
    const scratch_real_base = n * 2;
    const scratch_imaginary_base = n * 3;
    const cosine_base = n * 4;
    const sine_base = n * 5;
    if size == 1 {
        workspace[real_base + output] = workspace[input_offset];
        workspace[imaginary_base + output] = 0.0;
        return;
    }
    var radix: usize = 2;
    while size % radix != 0 {
        radix = radix + 1;
    }
    const block = size / radix;
    var part: usize = 0;
    while part < radix {
        fft_run(
            workspace,
            n,
            input_offset + part * input_stride,
            input_stride * radix,
            block,
            output + part * block
        );
        part = part + 1;
    }
    var frequency: usize = 0;
    while frequency < size {
        var real: f64 = 0.0;
        var imaginary: f64 = 0.0;
        part = 0;
        while part < radix {
            const child = output + part * block + frequency % block;
            const angle_index = (part * frequency) % size;
            const angle = angle_index * (n / size);
            const cosine = workspace[cosine_base + angle];
            const sine = workspace[sine_base + angle];
            const child_real = workspace[real_base + child];
            const child_imaginary = workspace[imaginary_base + child];
            real = real + child_real * cosine - child_imaginary * sine;
            imaginary = imaginary + child_real * sine +
            child_imaginary * cosine;
            part = part + 1;
        }
        workspace[scratch_real_base + output + frequency] = real;
        workspace[scratch_imaginary_base + output + frequency] = imaginary;
        frequency = frequency + 1;
    }
    frequency = 0;
    while frequency < size {
        workspace[real_base + output + frequency] =
        workspace[scratch_real_base + output + frequency];
        workspace[imaginary_base + output + frequency] =
        workspace[scratch_imaginary_base + output + frequency];
        frequency = frequency + 1;
    }
}

fn stft_power(ctx: *c_void, _: js.JsValue, argument_count: c_int,
    arguments_pointer: *const js.JsValue
): js.JsValue {
    if argument_count < 5 {
        return js.type_error(
            ctx,
            "stftPower expects signal, window, hop, padded length, and frame count"
        );
    }
    const signal_handle = live_tensor(ctx, argument(arguments_pointer, 0));
    const window_handle = live_tensor(ctx, argument(arguments_pointer, 1));
    if signal_handle is none {
        return js.type_error(ctx, "stftPower requires Tensor inputs");
    }
    if window_handle is none {
        return js.type_error(ctx, "stftPower requires Tensor inputs");
    }
    const hop = positive_usize(ctx, argument(arguments_pointer, 2));
    const padded_length = positive_usize(ctx, argument(arguments_pointer, 3));
    const frames = positive_usize(ctx, argument(arguments_pointer, 4));
    if hop is none {
        return js.type_error(ctx, "invalid STFT dimensions");
    }
    if padded_length is none {
        return js.type_error(ctx, "invalid STFT dimensions");
    }
    if frames is none {
        return js.type_error(ctx, "invalid STFT dimensions");
    }
    const signal = tensor_value(signal_handle);
    const window = tensor_value(window_handle);
    if signal.spec.dtype != DType.f32 or
    window.spec.dtype != DType.f64 or
    signal.spec.dimensions.len() != 1 or
    window.spec.dimensions.len() != 1 or
    not dense(signal_handle) or not dense(window_handle) {
        return js.type_error(
            ctx,
            "stftPower requires contiguous CPU f32 signal and f64 window"
        );
    }
    if signal.data is .f32(samples) {
        if window.data is .f64(weights) {
            const n = weights.len();
            if n >= 2 and n <= 4096 and padded_length >= samples.len() and
            padded_length > n / 2 and frames <= 1 + padded_length / hop {
                const bins = n / 2 + 1;
                var workspace: List<f64> = list<f64>();
                var workspace_index: usize = 0;
                while workspace_index < n * 7 {
                    workspace.push(0.0);
                    workspace_index = workspace_index + 1;
                }
                const pi = 3.14159265358979323846;
                var index: usize = 0;
                while index < n {
                    const angle = -2.0 * pi * index as f64 / n as f64;
                    workspace[n * 4 + index] =
                    cpu_math.compute(OpTag.cos, angle)
                    catch (error_code) => 0.0;
                    workspace[n * 5 + index] =
                    cpu_math.compute(OpTag.sin, angle)
                    catch (error_code) => 0.0;
                    index = index + 1;
                }
                var destination: List<f64> = list<f64>();
                index = 0;
                while index < bins * frames {
                    destination.push(0.0);
                    index = index + 1;
                }
                var frame: usize = 0;
                while frame < frames {
                    var nonzero = false;
                    index = 0;
                    while index < n {
                        var source = frame * hop + index;
                        if source < n / 2 {
                            source = (n / 2) - source;
                        } else {
                            source = source - n / 2;
                        }
                        if source >= padded_length {
                            source = 2 * padded_length - source - 2;
                        }
                        var value: f64 = 0.0;
                        if source < samples.len() {
                            value = numeric.f32_to_f64(samples[source]);
                        }
                        const weighted = value * weights[index];
                        workspace[n * 6 + index] = weighted;
                        if weighted != 0 {
                            nonzero = true;
                        }
                        index = index + 1;
                    }
                    if nonzero {
                        fft_run(&workspace, n, n * 6, 1, n, 0);
                        var bin: usize = 0;
                        while bin < bins {
                            const real = workspace[bin];
                            const imaginary = workspace[n + bin];
                            destination[bin * frames + frame] =
                            real * real + imaginary * imaginary;
                            bin = bin + 1;
                        }
                    }
                    frame = frame + 1;
                }
                const dimensions: [2]usize = [bins, frames];
                const tensor_spec = spec(DType.f64, dimensions[..])
                catch (error_code) {
                    const ignored = error_code;
                    return js.type_error(ctx, "STFT tensor shape is invalid");
                };
                const value = compute.tensor_f64(
                    tensor_spec,
                    destination[..]
                ) catch (error_code) {
                    const ignored = error_code;
                    return js.type_error(ctx, "STFT tensor creation failed");
                };
                return create_tensor_object(
                    ctx,
                    value,
                    maximum_sessions,
                    0,
                    true
                );
            }
        }
    }
    return js.type_error(ctx, "invalid STFT dimensions");
}

fn filterbank(ctx: *c_void, _: js.JsValue, argument_count: c_int,
    arguments_pointer: *const js.JsValue
): js.JsValue {
    if argument_count < 2 {
        return js.type_error(ctx, "filterbank expects spectrum and filters");
    }
    const spectrum_handle = live_tensor(ctx, argument(arguments_pointer, 0));
    const filters_handle = live_tensor(ctx, argument(arguments_pointer, 1));
    if spectrum_handle is none {
        return js.type_error(ctx, "filterbank requires Tensor inputs");
    }
    if filters_handle is none {
        return js.type_error(ctx, "filterbank requires Tensor inputs");
    }
    const spectrum = tensor_value(spectrum_handle);
    const filters = tensor_value(filters_handle);
    if spectrum.spec.dtype != DType.f64 or
    filters.spec.dtype != DType.f64 or
    spectrum.spec.dimensions.len() != 2 or
    filters.spec.dimensions.len() != 2 or
    not dense(spectrum_handle) or not dense(filters_handle) {
        return js.type_error(
            ctx,
            "filterbank requires contiguous CPU f64 matrices"
        );
    }
    if spectrum.data is .f64(values) {
        if filters.data is .f64(weights) {
            const bins = spectrum.spec.dimensions[0];
            const frames = spectrum.spec.dimensions[1];
            const bands = filters.spec.dimensions[0];
            if filters.spec.dimensions[1] == bins and
            bins > 0 and frames > 0 and bands > 0 {
                var destination: List<f64> = list<f64>();
                var index: usize = 0;
                while index < bands * frames {
                    destination.push(0.0);
                    index = index + 1;
                }
                var band: usize = 0;
                while band < bands {
                    var bin: usize = 0;
                    while bin < bins {
                        const weight = weights[band * bins + bin];
                        if weight != 0 {
                            var frame: usize = 0;
                            while frame < frames {
                                destination[band * frames + frame] =
                                destination[band * frames + frame] +
                                weight * values[bin * frames + frame];
                                frame = frame + 1;
                            }
                        }
                        bin = bin + 1;
                    }
                    band = band + 1;
                }
                const dimensions: [2]usize = [bands, frames];
                const tensor_spec = spec(DType.f64, dimensions[..])
                catch (error_code) {
                    const ignored = error_code;
                    return js.type_error(
                        ctx,
                        "filterbank tensor shape is invalid"
                    );
                };
                const value = compute.tensor_f64(
                    tensor_spec,
                    destination[..]
                ) catch (error_code) {
                    const ignored = error_code;
                    return js.type_error(
                        ctx,
                        "filterbank tensor creation failed"
                    );
                };
                return create_tensor_object(
                    ctx,
                    value,
                    maximum_sessions,
                    0,
                    true
                );
            }
        }
    }
    return js.type_error(ctx, "invalid filterbank dimensions");
}

pub fn add_exports(ctx: *c_void, module_ptr: *c_void): c_int {
    if js.add_module_export(ctx, module_ptr, "defaultDevice") != 0 {
        return -1;
    }
    if js.add_module_export(ctx, module_ptr, "createSession") != 0 {
        return -1;
    }
    if js.add_module_export(ctx, module_ptr, "sessionTensor") != 0 {
        return -1;
    }
    if js.add_module_export(ctx, module_ptr, "sessionTensorBytes") != 0 {
        return -1;
    }
    if js.add_module_export(ctx, module_ptr, "sessionTensorCopy") != 0 {
        return -1;
    }
    if js.add_module_export(ctx, module_ptr, "sessionFull") != 0 {
        return -1;
    }
    if js.add_module_export(ctx, module_ptr, "compileProgram") != 0 {
        return -1;
    }
    if js.add_module_export(ctx, module_ptr, "explainExecutable") != 0 {
        return -1;
    }
    if js.add_module_export(ctx, module_ptr, "cynaSelectedSteps") != 0 {
        return -1;
    }
    if js.add_module_export(ctx, module_ptr, "cynaSelectedPhysicalNodes") != 0 {
        return -1;
    }
    if js.add_module_export(ctx, module_ptr, "selectedPhysicalValues") != 0 {
        return -1;
    }
    if js.add_module_export(ctx, module_ptr, "selectedPhysicalLiveness") != 0 {
        return -1;
    }
    if js.add_module_export(ctx, module_ptr, "cynaResidualDecisions") != 0 {
        return -1;
    }
    if js.add_module_export(ctx, module_ptr, "cynaSelectionComparison") != 0 {
        return -1;
    }
    if js.add_module_export(ctx, module_ptr, "cynaSearchEvidence") != 0 {
        return -1;
    }
    if js.add_module_export(ctx, module_ptr, "executableRouting") != 0 {
        return -1;
    }
    if js.add_module_export(ctx, module_ptr, "executableBackend") != 0 {
        return -1;
    }
    if js.add_module_export(ctx, module_ptr, "runExecutable") != 0 {
        return -1;
    }
    if js.add_module_export(ctx, module_ptr, "resetEvidenceCapture") != 0 {
        return -1;
    }
    if js.add_module_export(ctx, module_ptr, "exportEvidenceCapture") != 0 {
        return -1;
    }
    if js.add_module_export(ctx, module_ptr, "stftPower") != 0 {
        return -1;
    }
    if js.add_module_export(ctx, module_ptr, "filterbank") != 0 {
        return -1;
    }
    if js.add_module_export(ctx, module_ptr, "default") != 0 {
        return -1;
    }
    return 0;
}

fn install_function(ctx: *c_void, object: js.JsValue, name: c_str,
    callback: js.Callback, argument_count: c_int, module_ptr: *c_void,
    export_name: c_str
): c_int {
    const value = js.new_function(
        ctx, callback, name, argument_count, 0, 0
    );
    if js.is_exception(value) {
        print("affon native module: C function creation failed\n");
        return -1;
    }
    if js.set_property_str(ctx, object, name, value) < 0 {
        print("affon native module: property installation failed\n");
        const exception = js.get_exception(ctx);
        const message = js.read_string(ctx, exception);
        if message is text {
            print($"affon native module: {text}\n");
        }
        js.free_value(ctx, exception);
        return -1;
    }
    const exported = js.get_property_str(ctx, object, name);
    defer js.free_value(ctx, exported);
    if js.set_module_export(
        ctx, module_ptr, export_name, js.dup_value(ctx, exported)
    ) < 0 {
        print("affon native module: export installation failed\n");
        return -1;
    }
    return 0;
}

pub fn initialize(ctx: *c_void, module_ptr: *c_void): c_int {
    const object = js.new_object(ctx);
    if js.is_exception(object) {
        print("affon native module: object creation failed\n");
        return -1;
    }
    if install_function(ctx, object, "defaultDevice", default_device, 0,
        module_ptr, "defaultDevice"
    ) != 0 {
        return -1;
    }
    if install_function(ctx, object, "createSession", create_session, 2,
        module_ptr, "createSession"
    ) != 0 {
        return -1;
    }
    if install_function(ctx, object, "sessionTensor", session_tensor, 4,
        module_ptr, "sessionTensor"
    ) != 0 {
        return -1;
    }
    if install_function(ctx, object, "sessionTensorBytes", session_tensor_bytes, 4,
        module_ptr, "sessionTensorBytes"
    ) != 0 {
        return -1;
    }
    if install_function(ctx, object, "sessionTensorCopy", session_tensor_copy, 2,
        module_ptr, "sessionTensorCopy"
    ) != 0 {
        return -1;
    }
    if install_function(ctx, object, "sessionFull", session_full, 4,
        module_ptr, "sessionFull"
    ) != 0 {
        return -1;
    }
    if install_function(ctx, object, "compileProgram", compile_program, 3,
        module_ptr, "compileProgram"
    ) != 0 {
        return -1;
    }
    if install_function(ctx, object, "explainExecutable", explain_executable, 1,
        module_ptr, "explainExecutable"
    ) != 0 {
        return -1;
    }
    if install_function(ctx, object, "cynaSelectedSteps", selected_steps, 1,
        module_ptr, "cynaSelectedSteps"
    ) != 0 {
        return -1;
    }
    if install_function(ctx, object, "cynaSelectedPhysicalNodes",
        selected_physical_nodes, 1,
        module_ptr, "cynaSelectedPhysicalNodes"
    ) != 0 {
        return -1;
    }
    if install_function(ctx, object, "selectedPhysicalValues",
        selected_physical_values, 1,
        module_ptr, "selectedPhysicalValues"
    ) != 0 {
        return -1;
    }
    if install_function(ctx, object, "selectedPhysicalLiveness",
        selected_physical_liveness, 1,
        module_ptr, "selectedPhysicalLiveness"
    ) != 0 {
        return -1;
    }
    if install_function(ctx, object, "cynaResidualDecisions",
        residual_decisions, 1,
        module_ptr, "cynaResidualDecisions"
    ) != 0 {
        return -1;
    }
    if install_function(ctx, object, "cynaSelectionComparison",
        selection_comparison, 1,
        module_ptr, "cynaSelectionComparison"
    ) != 0 {
        return -1;
    }
    if install_function(ctx, object, "cynaSearchEvidence", search_evidence, 1,
        module_ptr, "cynaSearchEvidence"
    ) != 0 {
        return -1;
    }
    if install_function(ctx, object, "executableRouting", executable_routing, 1,
        module_ptr, "executableRouting"
    ) != 0 {
        return -1;
    }
    if install_function(ctx, object, "executableBackend", executable_backend, 1,
        module_ptr, "executableBackend"
    ) != 0 {
        return -1;
    }
    if install_function(ctx, object, "runExecutable", run_executable, 2,
        module_ptr, "runExecutable"
    ) != 0 {
        return -1;
    }
    if install_function(ctx, object, "resetEvidenceCapture", reset_evidence, 1,
        module_ptr, "resetEvidenceCapture"
    ) != 0 {
        return -1;
    }
    if install_function(ctx, object, "exportEvidenceCapture", export_evidence, 2,
        module_ptr, "exportEvidenceCapture"
    ) != 0 {
        return -1;
    }
    if install_function(ctx, object, "stftPower", stft_power, 5,
        module_ptr, "stftPower"
    ) != 0 {
        return -1;
    }
    if install_function(ctx, object, "filterbank", filterbank, 2,
        module_ptr, "filterbank"
    ) != 0 {
        return -1;
    }
    if js.set_module_export(ctx, module_ptr, "default", object) < 0 {
        print("affon native module: default export failed\n");
        return -1;
    }
    return 0;
}
