import compute_core.lib as compute;
import { ComputeError, DType } from compute_core.lib;
import { OpTag } from compute_core.types;
import { error_name } from compute_core.compiler.index;
import compute_core.program.support.bfloat;
import compute_core.backend.cpu.math_unary as cpu_math;
import compute_core.backend.cpu.numeric;
import std.fs;
import std.json;
import { env_get } from std.process;
import { Metric, MetricKind, MetricUnit } from std.telemetry;
import { monotonic_ns } from std.time;
import { Runtime } from hao.runtime;
import .support.cyna_program as graph;
import .support.cyna_evidence as plan;
import .support.json_value as fields;
import .js.compute.quickjs as qjs;

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

fn argument(arguments_pointer: *const qjs.JSValue, index: usize): qjs.JSValue {
    unchecked {
        return *(arguments_pointer + index);
    }
}

fn number(ctx: *c_void, value: i64): qjs.JSValue {
    return qjs.new_number_from_f64(ctx, value as f64);
}

fn boolean(ctx: *c_void, value: bool): qjs.JSValue {
    var selected: c_llong = 0;
    if value {
        selected = 1;
    }
    const result: qjs.JSValue = {
        payload: selected,
        tag: qjs.tag_bool
    };
    return result;
}

fn property_int(ctx: *c_void, object: qjs.JSValue, name: c_str): ?i64 {
    const value = qjs.get_property_str(ctx, object, name);
    defer qjs.free_value(ctx, value);
    return qjs.read_int(ctx, value);
}

fn object_handle(ctx: *c_void, value: qjs.JSValue): ?usize {
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

fn shape(ctx: *c_void, value: qjs.JSValue): ?List<usize> {
    if qjs.is_array(value) == 0 {
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
            const item = qjs.get_property_uint32(ctx, value, index as c_uint);
            defer qjs.free_value(ctx, item);
            if qjs.read_int(ctx, item) is dimension {
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
): compute.TensorSpec !compute.ComputeError {
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
): qjs.JSValue {
    if next_tensor >= maximum_tensors {
        return qjs.type_error(ctx, "Tensor capacity exhausted");
    }
    const handle = next_tensor;
    next_tensor = next_tensor + 1;
    const tensor_generation = generation();
    const dimension_count = value.spec.dimensions.len();
    const selected_dtype = dtype_name(value.spec.dtype);
    var dimensions: List<qjs.JSValue> = list<qjs.JSValue>();
    for dimension in value.spec.dimensions[..] {
        dimensions.push(number(ctx, dimension as i64));
    }
    const box = new TensorBox { value: value };
    const slot = new TensorSlot {
        value: resource.leak(box),
        dense: dense
    };
    tensor_pointers[handle] = resource.leak(slot) as! usize;
    tensor_sessions[handle] = session;
    tensor_alive[handle] = true;
    const object = qjs.new_object(ctx);
    qjs.set_number(ctx, object, "ndim", dimension_count as f64);
    qjs.set_string(ctx, object, "dtype", selected_dtype);
    const shape_value = qjs.new_array(ctx);
    var index: usize = 0;
    while index < dimensions.len() {
        const ignored_shape = qjs.set_property_uint32(ctx, shape_value,
            index as c_uint,
            dimensions[index]
        );
        const ignored = ignored_shape;
        index = index + 1;
    }
    const ignored_shape_field = qjs.set_property_str(
        ctx, object, "shape", shape_value
    );
    const ignored = ignored_shape_field;
    qjs.set_string(ctx, object, "device", session_device(session));
    qjs.set_number(ctx, object, "__affon_handle", handle as f64);
    qjs.set_number(ctx, object, "__affon_generation",
        tensor_generation as f64
    );
    const ignored_session = session;
    const ignored_generation = session_generation;
    qjs.set_function(ctx, object, "item", tensor_item, 0);
    qjs.set_function(ctx, object, "to_array", tensor_to_array, 0);
    qjs.set_function(ctx, object, "to_bytes", tensor_to_bytes, 0);
    qjs.set_function(ctx, object, "toString", tensor_text, 0);
    qjs.set_function(ctx, object, "repr", tensor_text, 0);
    qjs.set_function(ctx, object, "dispose", tensor_dispose, 0);
    return object;
}

fn live_tensor(ctx: *c_void, value: qjs.JSValue): ?usize {
    if object_handle(ctx, value) is handle {
        if handle < maximum_tensors {
            if tensor_alive[handle] {
                return handle;
            }
        }
    }
    return none;
}

fn tensor_number(value: &const compute.Tensor, index: usize): f64 !compute.ComputeError {
    return switch value.data {
        .f32(values) => numeric.f32_to_f64(values[index]);
        .f64(values) => values[index];
        .u32(values) => values[index] as f64;
        .i64(values) => values[index] as f64;
        .bf16(values) => numeric.f32_to_f64(bfloat.to_f32(values[index]));
    };
}

fn tensor_item(ctx: *c_void, this_value: qjs.JSValue, argument_count: c_int,
    arguments_pointer: *const qjs.JSValue
): qjs.JSValue {
    const ignored_count = argument_count;
    const ignored_arguments = arguments_pointer;
    if live_tensor(ctx, this_value) is handle {
        const tensor = tensor_value(handle);
        const count = tensor.spec.element_count() catch (error_code) {
            const ignored = error_code;
            return qjs.type_error(ctx, "invalid Tensor");
        };
        if count != 1 {
            return qjs.type_error(ctx, "item() requires a one-element Tensor");
        }
        const value = tensor_number(tensor, 0) catch (error_code) {
            const ignored = error_code;
            return qjs.type_error(ctx, "invalid Tensor");
        };
        return qjs.new_number_from_f64(ctx, value);
    }
    return qjs.type_error(ctx, "Tensor has been disposed");
}

fn tensor_to_array(ctx: *c_void, this_value: qjs.JSValue,
    argument_count: c_int, arguments_pointer: *const qjs.JSValue
): qjs.JSValue {
    const ignored_count = argument_count;
    const ignored_arguments = arguments_pointer;
    if live_tensor(ctx, this_value) is handle {
        var cursor: usize = 0;
        return build_tensor_array(ctx, handle, 0, &cursor);
    }
    return qjs.type_error(ctx, "Tensor has been disposed");
}

fn build_tensor_array(ctx: *c_void, handle: usize, depth: usize,
    cursor: &usize
): qjs.JSValue {
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
                return qjs.type_error(ctx, "invalid Tensor");
            };
            unchecked {
                cursor = selected + 1;
            }
            return qjs.new_number_from_f64(ctx, value);
        }
        const result = qjs.new_array(ctx);
        var index: usize = 0;
        while index < dimensions[depth] {
            const ignored_item = qjs.set_property_uint32(ctx, result,
                index as c_uint,
                build_tensor_array(ctx, handle, depth + 1, cursor)
            );
            const ignored = ignored_item;
            index = index + 1;
                }
        return result;
    }
    return qjs.type_error(ctx, "invalid Tensor");
}

fn tensor_to_bytes(ctx: *c_void, this_value: qjs.JSValue,
    argument_count: c_int, arguments_pointer: *const qjs.JSValue
): qjs.JSValue {
    const ignored_count = argument_count;
    const ignored_arguments = arguments_pointer;
    if live_tensor(ctx, this_value) is handle {
        const bytes = graph.tensor_bytes(tensor_value(handle));
        const view = bytes[..];
        return qjs.new_array_buffer_copy(
            ctx,
            view.ptr as! *const c_uchar,
            view.len() as c_size_t
        );
    }
    return qjs.type_error(ctx, "Tensor has been disposed");
}

fn tensor_text(ctx: *c_void, this_value: qjs.JSValue,
    argument_count: c_int, arguments_pointer: *const qjs.JSValue
): qjs.JSValue {
    const ignored_count = argument_count;
    const ignored_arguments = arguments_pointer;
    if live_tensor(ctx, this_value) is handle {
        const tensor = tensor_value(handle);
        const dimension_count = tensor.spec.dimensions.len();
        const selected_dtype = dtype_name(tensor.spec.dtype);
        return qjs.string(ctx,
            $"Tensor(shape={dimension_count}, dtype={selected_dtype})"
        );
    }
    return qjs.string(ctx, "Tensor(disposed)");
}

fn tensor_dispose(ctx: *c_void, this_value: qjs.JSValue,
    argument_count: c_int, arguments_pointer: *const qjs.JSValue
): qjs.JSValue {
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
    return qjs.undefined();
}

fn create_session(ctx: *c_void, _: qjs.JSValue, argument_count: c_int,
    arguments_pointer: *const qjs.JSValue
): qjs.JSValue {
    if argument_count < 1 {
        return qjs.type_error(ctx, "createSession expects a device");
    }
    const requested_device = qjs.read_string(ctx, argument(arguments_pointer, 0));
    var selected_device = "";
    var uses_cuda = false;
    var device_ordinal: u32 = 0;
    if requested_device is text {
        selected_device = text.clone();
        if text == "cuda" {
            uses_cuda = true;
        } else if text.starts_with("cuda:") {
            const bytes = text.bytes();
            const suffix = string.from(bytes[5..]);
            const parsed = fields.parse_usize(suffix);
            if parsed is ordinal {
                if ordinal <= 4_294_967_295 {
                    uses_cuda = true;
                    device_ordinal = ordinal as u32;
                } else {
                    return qjs.type_error(ctx, "unsupported device");
                }
            } else {
                return qjs.type_error(ctx, "unsupported device");
            }
        } else if text != "cpu" {
            return qjs.type_error_string(
                ctx,
                $"backend unavailable: {text}"
            );
        }
    } else {
        return qjs.type_error(ctx, "device must be a string");
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
        return qjs.type_error(ctx, "Session capacity exhausted");
    }
    var capture_enabled = false;
    var capture_capacity: usize = 0;
    var timing_scope = "none";
    var hardware_metrics = false;
    var profiler_markers = false;
    if argument_count >= 2 {
        if qjs.read_string(ctx, argument(arguments_pointer, 1)) is options_json {
            const document = json.parse(options_json) catch (error_code) {
                const ignored = error_code;
                return qjs.type_error(ctx, "invalid Session options");
            };
            if document is .object(options) {
                if fields.field_index(&options, "telemetry") is telemetry_index {
                    if options.members[telemetry_index].value is
                        .object(telemetry_options) {
                        if fields.usize_field(
                            &telemetry_options,
                            "captureRecords"
                        ) is records {
                            if records == 0 or records > 4_194_304 {
                                return qjs.type_error(
                                    ctx,
                                    "captureRecords must be between 1 and 4194304"
                                );
                            }
                            capture_enabled = true;
                            capture_capacity = records;
                            timing_scope = "capture";
                        }
                        if fields.string_field(
                            &telemetry_options,
                            "computeTiming"
                        ) is selected_timing {
                            if selected_timing == "invocation" or
                                selected_timing == "kernel" {
                                timing_scope = selected_timing;
                            } else if selected_timing == "none" and
                                not capture_enabled {
                                timing_scope = "none";
                            }
                        }
                        hardware_metrics = fields.bool_field(
                            &telemetry_options,
                            "hardwareMetrics",
                            false
                        );
                        profiler_markers = fields.bool_field(
                            &telemetry_options,
                            "profilerMarkers",
                            false
                        );
                    }
                }
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
            return qjs.type_error_string(
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
            return qjs.type_error_string(
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
    const object = qjs.new_object(ctx);
    qjs.set_number(ctx, object, "__affon_handle", handle as f64);
    qjs.set_string(ctx, object, "device", sessions[handle].device);
    qjs.set_function(ctx, object, "dispose", session_dispose, 0);
    return object;
}

fn live_session(ctx: *c_void, value: qjs.JSValue): ?usize {
    if object_handle(ctx, value) is handle {
        if handle < maximum_sessions {
            if sessions[handle].alive {
                return handle;
            }
        }
    }
    return none;
}

fn session_dispose(ctx: *c_void, this_value: qjs.JSValue,
    argument_count: c_int, arguments_pointer: *const qjs.JSValue
): qjs.JSValue {
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
    return qjs.undefined();
}

fn session_tensor(ctx: *c_void, _: qjs.JSValue, argument_count: c_int,
    arguments_pointer: *const qjs.JSValue
): qjs.JSValue {
    if argument_count < 4 {
        return qjs.type_error(ctx, "sessionTensor expects Session, values, dtype, and shape");
    }
    const session = live_session(ctx, argument(arguments_pointer, 0));
    if session is none {
        return qjs.type_error(ctx, "invalid Session");
    }
    const values = argument(arguments_pointer, 1);
    const dtype_text = qjs.read_string(ctx, argument(arguments_pointer, 2));
    if dtype_text is none {
        return qjs.type_error(ctx, "dtype must be a string");
    }
    const selected_dtype = dtypes(dtype_text);
    if selected_dtype is none {
        return qjs.type_error(ctx, "unsupported dtype");
    }
    const dimensions = shape(ctx, argument(arguments_pointer, 3));
    if dimensions is none {
        return qjs.type_error(ctx, "shape must be an array");
    }
    const count = element_count(dimensions[..]);
    if count is none {
        return qjs.type_error(ctx, "invalid Tensor shape");
    }
    var selected_values: List<f64> = list<f64>();
    if not flatten_tensor_values(ctx, values, &selected_values) {
        return qjs.type_error(ctx, "tensor expects a rectangular numeric value");
    }
    if selected_values.len() != count {
        return qjs.type_error(ctx, "tensor value count does not match shape");
    }
    const value = tensor_from_numbers(
        ctx,
        selected_values[..],
        selected_dtype,
        dimensions[..]
    )
    catch (error_code) {
        const ignored = error_code;
        return qjs.type_error(ctx, "tensor creation failed");
    };
    return create_tensor_object(
        ctx,
        value,
        session,
        session_generation(session),
        true
    );
}

fn flatten_tensor_values(ctx: *c_void, value: qjs.JSValue,
    data: &List<f64>
): bool {
    if qjs.is_array(value) == 0 {
        if qjs.read_number(ctx, value) is selected {
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
            const item = qjs.get_property_uint32(ctx, value, index as c_uint);
            defer qjs.free_value(ctx, item);
            if not flatten_tensor_values(ctx, item, data) {
                return false;
            }
            index = index + 1;
        }
        return true;
    }
    return false;
}

fn session_tensor_bytes(ctx: *c_void, _: qjs.JSValue, argument_count: c_int,
    arguments_pointer: *const qjs.JSValue
): qjs.JSValue {
    if argument_count < 4 {
        return qjs.type_error(ctx, "sessionTensorBytes expects Session, bytes, dtype, and shape");
    }
    const session = live_session(ctx, argument(arguments_pointer, 0));
    if session is none {
        return qjs.type_error(ctx, "invalid Session");
    }
    var byte_count_storage: [1]c_size_t = [0];
    const bytes = qjs.get_array_buffer(
        ctx, byte_count_storage[..].ptr, argument(arguments_pointer, 1)
    );
    if bytes is none {
        return qjs.type_error(ctx, "tensor bytes must be an ArrayBuffer");
    }
    const byte_count = byte_count_storage[0];
    const dtype_text = qjs.read_string(ctx, argument(arguments_pointer, 2));
    if dtype_text is none {
        return qjs.type_error(ctx, "dtype must be a string");
    }
    const selected_dtype = dtypes(dtype_text);
    if selected_dtype is none {
        return qjs.type_error(ctx, "unsupported dtype");
    }
    const dimensions = shape(ctx, argument(arguments_pointer, 3));
    if dimensions is none {
        return qjs.type_error(ctx, "shape must be an array");
    }
    const tensor_spec = spec(selected_dtype, dimensions[..])
    catch (error_code) {
        const ignored = error_code;
        return qjs.type_error(ctx, "invalid Tensor shape");
    };
    const count = tensor_spec.element_count() catch (error_code) {
        const ignored = error_code;
        return qjs.type_error(ctx, "invalid Tensor shape");
    };
    const width = graph.dtype_width(tensor_spec.dtype);
    if count * width != byte_count as usize {
        return qjs.type_error(ctx, "buffer byte length does not match shape and dtype");
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
        return qjs.type_error(ctx, "tensor creation failed");
    };
    return create_tensor_object(
        ctx,
        value,
        session,
        session_generation(session),
        true
    );
}

fn session_tensor_copy(ctx: *c_void, _: qjs.JSValue, argument_count: c_int,
    arguments_pointer: *const qjs.JSValue
): qjs.JSValue {
    if argument_count < 2 {
        return qjs.type_error(ctx, "sessionTensorCopy expects Session and Tensor");
    }
    const session = live_session(ctx, argument(arguments_pointer, 0));
    if session is none {
        return qjs.type_error(ctx, "invalid Session");
    }
    const source = live_tensor(ctx, argument(arguments_pointer, 1));
    if source is none {
        return qjs.type_error(ctx, "source must be a Tensor");
    }
    return create_tensor_object(
        ctx,
        compute.clone_tensor(tensor_value(source)),
        session,
        session_generation(session),
        tensor_dense(source as usize)
    );
}

fn session_full(ctx: *c_void, _: qjs.JSValue, argument_count: c_int,
    arguments_pointer: *const qjs.JSValue
): qjs.JSValue {
    if argument_count < 4 {
        return qjs.type_error(ctx, "sessionFull expects Session, shape, value, and dtype");
    }
    const session = live_session(ctx, argument(arguments_pointer, 0));
    if session is none {
        return qjs.type_error(ctx, "invalid Session");
    }
    const dimensions = shape(ctx, argument(arguments_pointer, 1));
    if dimensions is none {
        return qjs.type_error(ctx, "shape must be an array");
    }
    const fill = qjs.read_number(ctx, argument(arguments_pointer, 2));
    if fill is none {
        return qjs.type_error(ctx, "fill value must be a number");
    }
    const dtype_text = qjs.read_string(ctx, argument(arguments_pointer, 3));
    if dtype_text is none {
        return qjs.type_error(ctx, "unsupported dtype");
    }
    const selected_dtype = dtypes(dtype_text);
    if selected_dtype is none {
        return qjs.type_error(ctx, "unsupported dtype");
    }
    const tensor_spec = spec(selected_dtype, dimensions[..])
    catch (error_code) {
        const ignored = error_code;
        return qjs.type_error(ctx, "invalid Tensor shape");
    };
    const count = tensor_spec.element_count() catch (error_code) {
        const ignored = error_code;
        return qjs.type_error(ctx, "invalid Tensor shape");
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
        return qjs.type_error(ctx, "tensor creation failed");
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
): compute.Tensor !compute.ComputeError {
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

fn compile_program(ctx: *c_void, _: qjs.JSValue, argument_count: c_int,
    arguments_pointer: *const qjs.JSValue
): qjs.JSValue {
    if argument_count < 2 {
        return qjs.type_error(ctx, "compileProgram expects Session and Program JSON");
    }
    const session = live_session(ctx, argument(arguments_pointer, 0));
    if session is none {
        return qjs.type_error(ctx, "invalid Session");
    }
    const program_json = qjs.read_string(ctx, argument(arguments_pointer, 1));
    if program_json is none {
        return qjs.type_error(ctx, "Program must be JSON");
    }
    var options_json: ?string = none;
    if argument_count >= 3 {
        options_json = qjs.read_string(ctx, argument(arguments_pointer, 2));
    }
    const document = json.parse(program_json) catch (error_code) {
        const ignored = error_code;
        return qjs.type_error(ctx, "Program must be valid JSON");
    };
    const backend = session_backend(session);
    const compiled = graph.compile_program(&document, options_json, &backend)
    catch (error_code) {
        const failure = error_name(error_code);
        if sessions[session].cuda {
            compute.report_cuda_error();
        }
        return qjs.type_error_string(
            ctx,
            $"Cyna lowering failed: {failure}"
        );
    };
    if next_executable >= maximum_executables {
        return qjs.type_error(ctx, "Executable capacity exhausted");
    }
    const handle = next_executable;
    next_executable = next_executable + 1;
    const executable_generation = generation();
    const explanation = plan.explanation(
        &compiled.executable,
        &compiled.source_instruction_by_step
    );
    const owner = sessions[session];
    const box = new ExecutableBox { value: compiled };
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
    const object = qjs.new_object(ctx);
    qjs.set_number(ctx, object, "__affon_handle", handle as f64);
    qjs.set_number(ctx, object, "__affon_generation",
        executable_generation as f64
    );
    qjs.set_function(ctx, object, "dispose", executable_dispose, 0);
    return object;
}

fn live_executable(ctx: *c_void, value: qjs.JSValue): ?usize {
    if object_handle(ctx, value) is handle {
        if handle < maximum_executables {
            if executable_alive[handle] {
                return handle;
            }
        }
    }
    return none;
}

fn executable_dispose(ctx: *c_void, this_value: qjs.JSValue,
    argument_count: c_int, arguments_pointer: *const qjs.JSValue
): qjs.JSValue {
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
    return qjs.undefined();
}

fn executable_json(ctx: *c_void, arguments_pointer: *const qjs.JSValue,
    argument_count: c_int, explanation: bool
): qjs.JSValue {
    if argument_count < 1 {
        return qjs.type_error(ctx, "Executable argument is required");
    }
    const handle = live_executable(ctx, argument(arguments_pointer, 0));
    if handle is none {
        return qjs.type_error(ctx, "invalid Executable");
    }
    const slot = executable_slot(handle);
    if explanation {
        const compiled = executable_value(handle);
        return qjs.string(ctx, plan.explanation(
            &compiled.executable,
            &compiled.source_instruction_by_step
        ));
    }
    unchecked {
        return qjs.string(ctx, (*slot).routing);
    }
}

fn explain_executable(ctx: *c_void, _: qjs.JSValue, argument_count: c_int,
    arguments_pointer: *const qjs.JSValue
): qjs.JSValue {
    return executable_json(ctx, arguments_pointer, argument_count, true);
}

fn executable_routing(ctx: *c_void, _: qjs.JSValue, argument_count: c_int,
    arguments_pointer: *const qjs.JSValue
): qjs.JSValue {
    return executable_json(ctx, arguments_pointer, argument_count, false);
}

fn executable_backend(ctx: *c_void, _: qjs.JSValue, argument_count: c_int,
    arguments_pointer: *const qjs.JSValue
): qjs.JSValue {
    if argument_count < 1 or live_executable(
        ctx, argument(arguments_pointer, 0)
    ) is none {
        return qjs.type_error(ctx, "invalid Executable");
    }
    return qjs.string(ctx, "cyna");
}

fn executable_plan_json(ctx: *c_void, argument_count: c_int,
    arguments_pointer: *const qjs.JSValue, kind: i32
): qjs.JSValue {
    if argument_count < 1 {
        return qjs.type_error(ctx, "Executable argument is required");
    }
    const handle = live_executable(ctx, argument(arguments_pointer, 0));
    if handle is none {
        return qjs.type_error(ctx, "invalid Executable");
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
    return qjs.string(ctx, text);
}

fn selected_steps(ctx: *c_void, _: qjs.JSValue, argument_count: c_int,
    arguments_pointer: *const qjs.JSValue
): qjs.JSValue {
    return executable_plan_json(ctx, argument_count, arguments_pointer, 1);
}

fn selected_physical_nodes(ctx: *c_void, _: qjs.JSValue,
    argument_count: c_int, arguments_pointer: *const qjs.JSValue
): qjs.JSValue {
    return executable_plan_json(ctx, argument_count, arguments_pointer, 2);
}

fn selected_physical_values(ctx: *c_void, _: qjs.JSValue,
    argument_count: c_int, arguments_pointer: *const qjs.JSValue
): qjs.JSValue {
    return executable_plan_json(ctx, argument_count, arguments_pointer, 3);
}

fn selected_physical_liveness(ctx: *c_void, _: qjs.JSValue,
    argument_count: c_int, arguments_pointer: *const qjs.JSValue
): qjs.JSValue {
    return executable_plan_json(ctx, argument_count, arguments_pointer, 4);
}

fn residual_decisions(ctx: *c_void, _: qjs.JSValue, argument_count: c_int,
    arguments_pointer: *const qjs.JSValue
): qjs.JSValue {
    return executable_plan_json(ctx, argument_count, arguments_pointer, 5);
}

fn selection_comparison(ctx: *c_void, _: qjs.JSValue, argument_count: c_int,
    arguments_pointer: *const qjs.JSValue
): qjs.JSValue {
    return executable_plan_json(ctx, argument_count, arguments_pointer, 6);
}

fn search_evidence(ctx: *c_void, _: qjs.JSValue, argument_count: c_int,
    arguments_pointer: *const qjs.JSValue
): qjs.JSValue {
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

fn run_executable(ctx: *c_void, _: qjs.JSValue, argument_count: c_int,
    arguments_pointer: *const qjs.JSValue
): qjs.JSValue {
    if argument_count < 2 {
        return qjs.type_error(ctx, "runExecutable expects Executable and inputs");
    }
    const executable_handle = live_executable(
        ctx, argument(arguments_pointer, 0)
    );
    if executable_handle is none {
        return qjs.type_error(ctx, "invalid Executable");
    }
    const inputs = argument(arguments_pointer, 1);
    if qjs.is_array(inputs) == 0 {
        return qjs.type_error(ctx, "inputs must be an array");
    }
    const count = property_int(ctx, inputs, "length");
    if count is none {
        return qjs.type_error(ctx, "invalid inputs");
    }
    if count < 0 {
        return qjs.type_error(ctx, "invalid inputs");
    }
    const executable = executable_slot(executable_handle);
    const compiled = executable_value(executable_handle);
    const node_count = compiled.executable.graph.nodes.len();
    var owned: List<compute.Tensor> = list<compute.Tensor>();
    var index: usize = 0;
    while index < count as usize {
        const item = qjs.get_property_uint32(ctx, inputs, index as c_uint);
        defer qjs.free_value(ctx, item);
        if live_tensor(ctx, item) is handle {
            const source = tensor_value(handle);
            if compiled.dense_inputs and not dense(handle) {
                return qjs.type_error(
                    ctx,
                    "Program execution failed: GuardMismatch"
                );
            }
            owned.push(compute.clone_tensor(source));
        } else {
            return qjs.type_error(ctx, "inputs must be Tensors");
        }
        index = index + 1;
    }
    if compiled.needs_seed {
        if owned.len() + 1 != compiled.executable.input_guards.len() {
            return qjs.type_error(ctx, "Program execution failed: InvalidCommand");
        }
        const guard_index = owned.len();
        const guard = &compiled.executable.input_guards[guard_index];
        if guard.spec.dtype == DType.i64 or guard.spec.dtype == DType.u32 {
            return qjs.type_error(
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
            return qjs.type_error(ctx, "seed creation failed");
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
        return qjs.type_error_string(
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
    const result = qjs.new_array(ctx);
    index = 0;
    while index < outputs.len() {
        unchecked {
            const output_session = (*executable).session;
            const ignored_output = qjs.set_property_uint32(ctx, result,
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

fn default_device(ctx: *c_void, _: qjs.JSValue, argument_count: c_int,
    arguments_pointer: *const qjs.JSValue
): qjs.JSValue {
    const ignored_count = argument_count;
    const ignored_arguments = arguments_pointer;
    if env_get("AFFON_DEVICE") is device {
        return qjs.string(ctx, device);
    }
    if compute.cuda_available() {
        return qjs.string(ctx, "cuda");
    }
    return qjs.string(ctx, "cpu");
}

fn reset_evidence(ctx: *c_void, _: qjs.JSValue, argument_count: c_int,
    arguments_pointer: *const qjs.JSValue
): qjs.JSValue {
    if argument_count < 1 {
        return qjs.type_error(ctx, "resetEvidenceCapture expects Session");
    }
    const session_handle = live_session(ctx, argument(arguments_pointer, 0));
    if session_handle is none {
        return qjs.type_error(ctx, "invalid Session");
    }
    if not sessions[session_handle].capture_enabled {
        return qjs.type_error(
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
    return qjs.undefined();
}

fn export_evidence(ctx: *c_void, _: qjs.JSValue, argument_count: c_int,
    arguments_pointer: *const qjs.JSValue
): qjs.JSValue {
    if argument_count < 2 {
        return qjs.type_error(
            ctx,
            "exportEvidenceCapture expects Session and Executable"
        );
    }
    const session_handle = live_session(ctx, argument(arguments_pointer, 0));
    if session_handle is none {
        return qjs.type_error(ctx, "invalid Session");
    }
    const executable_handle = live_executable(
        ctx,
        argument(arguments_pointer, 1)
    );
    if executable_handle is none {
        return qjs.type_error(ctx, "invalid Executable");
    }
    if not sessions[session_handle].capture_enabled {
        return qjs.type_error(
            ctx,
            "runtime evidence capture is not enabled"
        );
    }
    const executable = executable_slot(executable_handle);
    unchecked {
        if (*executable).session != session_handle {
            return qjs.type_error(
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
    return qjs.string(ctx, serialized);
}

fn positive_usize(ctx: *c_void, value: qjs.JSValue): ?usize {
    if qjs.read_int(ctx, value) is selected {
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

fn stft_power(ctx: *c_void, _: qjs.JSValue, argument_count: c_int,
    arguments_pointer: *const qjs.JSValue
): qjs.JSValue {
    if argument_count < 5 {
        return qjs.type_error(
            ctx,
            "stftPower expects signal, window, hop, padded length, and frame count"
        );
    }
    const signal_handle = live_tensor(ctx, argument(arguments_pointer, 0));
    const window_handle = live_tensor(ctx, argument(arguments_pointer, 1));
    if signal_handle is none {
        return qjs.type_error(ctx, "stftPower requires Tensor inputs");
    }
    if window_handle is none {
        return qjs.type_error(ctx, "stftPower requires Tensor inputs");
    }
    const hop = positive_usize(ctx, argument(arguments_pointer, 2));
    const padded_length = positive_usize(ctx, argument(arguments_pointer, 3));
    const frames = positive_usize(ctx, argument(arguments_pointer, 4));
    if hop is none {
        return qjs.type_error(ctx, "invalid STFT dimensions");
    }
    if padded_length is none {
        return qjs.type_error(ctx, "invalid STFT dimensions");
    }
    if frames is none {
        return qjs.type_error(ctx, "invalid STFT dimensions");
    }
    const signal = tensor_value(signal_handle);
    const window = tensor_value(window_handle);
    if signal.spec.dtype != DType.f32 or
        window.spec.dtype != DType.f64 or
        signal.spec.dimensions.len() != 1 or
        window.spec.dimensions.len() != 1 or
        not dense(signal_handle) or not dense(window_handle) {
        return qjs.type_error(
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
                    return qjs.type_error(ctx, "STFT tensor shape is invalid");
                };
                const value = compute.tensor_f64(
                    tensor_spec,
                    destination[..]
                ) catch (error_code) {
                    const ignored = error_code;
                    return qjs.type_error(ctx, "STFT tensor creation failed");
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
    return qjs.type_error(ctx, "invalid STFT dimensions");
}

fn filterbank(ctx: *c_void, _: qjs.JSValue, argument_count: c_int,
    arguments_pointer: *const qjs.JSValue
): qjs.JSValue {
    if argument_count < 2 {
        return qjs.type_error(ctx, "filterbank expects spectrum and filters");
    }
    const spectrum_handle = live_tensor(ctx, argument(arguments_pointer, 0));
    const filters_handle = live_tensor(ctx, argument(arguments_pointer, 1));
    if spectrum_handle is none {
        return qjs.type_error(ctx, "filterbank requires Tensor inputs");
    }
    if filters_handle is none {
        return qjs.type_error(ctx, "filterbank requires Tensor inputs");
    }
    const spectrum = tensor_value(spectrum_handle);
    const filters = tensor_value(filters_handle);
    if spectrum.spec.dtype != DType.f64 or
        filters.spec.dtype != DType.f64 or
        spectrum.spec.dimensions.len() != 2 or
        filters.spec.dimensions.len() != 2 or
        not dense(spectrum_handle) or not dense(filters_handle) {
        return qjs.type_error(
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
                    return qjs.type_error(
                        ctx,
                        "filterbank tensor shape is invalid"
                    );
                };
                const value = compute.tensor_f64(
                    tensor_spec,
                    destination[..]
                ) catch (error_code) {
                    const ignored = error_code;
                    return qjs.type_error(
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
    return qjs.type_error(ctx, "invalid filterbank dimensions");
}

fn push_text(output: &List<u8>, value: string): void {
    for byte in value.bytes() {
        output.push(byte);
    }
}

fn push_decimal(output: &List<u8>, value: usize, digits: usize): void {
    if value == 0 {
        if digits > 0 {
            output.push(48);
        }
        return;
    }
    var divisor: usize = 1;
    var remaining_digits = digits;
    while remaining_digits > 1 {
        divisor = divisor * 10;
        remaining_digits = remaining_digits - 1;
    }
    while divisor > 0 {
        output.push(48 + ((value / divisor) % 10) as u8);
        divisor = divisor / 10;
    }
}

fn usize_digits(value: usize): usize {
    var digits: usize = 1;
    var remaining = value;
    while remaining >= 10 {
        remaining = remaining / 10;
        digits = digits + 1;
    }
    return digits;
}

fn push_usize(output: &List<u8>, value: usize): void {
    push_decimal(output, value, usize_digits(value));
}

fn push_json_string(output: &List<u8>, value: string): void {
    output.push(34);
    for byte in value.bytes() {
        if byte == 34 or byte == 92 {
            output.push(92);
            output.push(byte);
        } else if byte == 10 {
            push_text(output, "\\n");
        } else if byte == 13 {
            push_text(output, "\\r");
        } else if byte == 9 {
            push_text(output, "\\t");
        } else if byte >= 32 {
            output.push(byte);
        }
    }
    output.push(34);
}

fn checkpoint_dtype_name(dtype: DType): string {
    return switch dtype {
        .bf16 => "BF16";
        .f32 => "F32";
        .f64 => "F64";
        .u32 => "U32";
        .i64 => "I64";
    };
}

fn checkpoint_dtype(value: string): ?DType {
    if value == "BF16" { return DType.bf16; }
    if value == "F32" { return DType.f32; }
    if value == "F64" { return DType.f64; }
    if value == "U32" { return DType.u32; }
    if value == "I64" { return DType.i64; }
    return none;
}

fn save_native(ctx: *c_void, _: qjs.JSValue, argument_count: c_int,
    arguments_pointer: *const qjs.JSValue
): qjs.JSValue {
    if argument_count < 2 or
        qjs.is_array(argument(arguments_pointer, 0)) == 0 {
        return qjs.type_error(
            ctx,
            "checkpoint.save expects entries and a path"
        );
    }
    const path = qjs.read_string(ctx, argument(arguments_pointer, 1));
    if path is none {
        return qjs.type_error(
            ctx,
            "checkpoint.save expects a string path"
        );
    }
    const entries_value = argument(arguments_pointer, 0);
    const count = property_int(ctx, entries_value, "length");
    if count is none {
        return qjs.type_error(ctx, "checkpoint.save requires non-empty entries");
    }
    if count <= 0 {
        return qjs.type_error(ctx, "checkpoint.save requires non-empty entries");
    }
    var names: List<string> = list<string>();
    var tensors: List<usize> = list<usize>();
    var index: usize = 0;
    while index < count as usize {
        const item = qjs.get_property_uint32(
            ctx,
            entries_value,
            index as c_uint
        );
        defer qjs.free_value(ctx, item);
        const name_value = qjs.get_property_str(ctx, item, "name");
        defer qjs.free_value(ctx, name_value);
        const name = qjs.read_string(ctx, name_value);
        if name is none {
            return qjs.type_error(
                ctx,
                "checkpoint entry name must be a string"
            );
        }
        const tensor_value_js = qjs.get_property_str(ctx, item, "value");
        defer qjs.free_value(ctx, tensor_value_js);
        const tensor_handle = live_tensor(ctx, tensor_value_js);
        if tensor_handle is none {
            return qjs.type_error(
                ctx,
                "checkpoint entry value must be a Tensor"
            );
        }
        names.push(name);
        tensors.push(tensor_handle);
        index = index + 1;
    }
    var header: List<u8> = list<u8>();
    header.push(123);
    var offset: usize = 0;
    index = 0;
    while index < names.len() {
        if index != 0 {
            header.push(44);
        }
        const tensor = tensor_value(tensors[index]);
        push_json_string(&header, names[index]);
        push_text(&header, ":{\"dtype\":\"");
        push_text(&header, checkpoint_dtype_name(
            tensor.spec.dtype
        ));
        push_text(&header, "\",\"shape\":[");
        var dimension_index: usize = 0;
        while dimension_index < tensor.spec.dimensions.len() {
            if dimension_index != 0 {
                header.push(44);
            }
            push_usize(&header, tensor.spec.dimensions[dimension_index]);
            dimension_index = dimension_index + 1;
        }
        push_text(&header, "],\"data_offsets\":[");
        push_usize(&header, offset);
        const width = graph.dtype_width(tensor.spec.dtype);
        const count_value = tensor.spec.element_count() catch (error_code) {
            const ignored = error_code;
            return qjs.type_error(ctx, "checkpoint tensor shape is invalid");
        };
        offset = offset + count_value * width;
        header.push(44);
        push_usize(&header, offset);
        push_text(&header, "]}");
        index = index + 1;
    }
    header.push(125);
    var output: List<u8> = list<u8>();
    var prefix_index: usize = 0;
    while prefix_index < 8 {
        output.push(0);
        prefix_index = prefix_index + 1;
    }
    for byte in header[..] {
        output.push(byte);
    }
    index = 0;
    while index < tensors.len() {
        const tensor_data = graph.tensor_bytes(tensor_value(tensors[index]));
        for byte in tensor_data[..] {
            output.push(byte);
        }
        index = index + 1;
    }
    prefix_index = 0;
    var header_size = header.len();
    while prefix_index < 8 {
        output[prefix_index] = (header_size % 256) as u8;
        header_size = header_size / 256;
        prefix_index = prefix_index + 1;
    }
    fs.write_bytes(path, output[..]) catch (error_code) {
        const ignored = error_code;
        return qjs.type_error(
            ctx,
            "checkpoint.save failed to write file"
        );
    };
    return qjs.undefined();
}

fn little_u64(bytes: []const u8, start: usize): ?u64 {
    if start + 8 > bytes.len() {
        return none;
    }
    var value: u64 = 0;
    var index: usize = 0;
    while index < 8 {
        value = value | ((bytes[start + index] as u64) << (index * 8));
        index = index + 1;
    }
    return value;
}

fn parse_nonnegative(text: string, maximum: usize): ?usize {
    const bytes = text.bytes();
    if bytes.len() == 0 {
        return none;
    }
    var value: usize = 0;
    for byte in bytes {
        if byte < 48 or byte > 57 {
            return none;
        }
        const digit = (byte - 48) as usize;
        if value > (maximum - digit) / 10 {
            return none;
        }
        value = value * 10 + digit;
    }
    return value;
}

fn checkpoint_entries(header: &const json.Value,
    raw_size: usize
): ?List<CheckpointTensor> {
    if header is .object(document) {
        var result: List<CheckpointTensor> = list<CheckpointTensor>();
        for member in document.members[..] {
            if member.key == "__metadata__" {
                if member.value is .object(metadata) {
                    for value in metadata.members[..] {
                        if value.value is .string(text) {
                            const ignored = text;
                        } else {
                            return none;
                        }
                    }
                } else {
                    return none;
                }
                continue;
            }
            if member.value is .object(metadata) {
                const dtype_index = fields.field_index(&metadata, "dtype");
                const shape_index = fields.field_index(&metadata, "shape");
                const offsets_index = fields.field_index(
                    &metadata,
                    "data_offsets"
                );
                if dtype_index is none {
                    return none;
                }
                if shape_index is none {
                    return none;
                }
                if offsets_index is none {
                    return none;
                }
                const dtype_value = &metadata.members[dtype_index].value;
                if dtype_value is .string(dtype_text) {
                    const dtype = checkpoint_dtype(dtype_text);
                    if dtype is none {
                        return none;
                    }
                    const shape_value = &metadata.members[shape_index].value;
                    if shape_value is .array(shape) {
                        var dimensions: List<usize> = list<usize>();
                        var elements: usize = 1;
                        const maximum: usize = (0 as usize) -% 1;
                        for dimension in shape.items[..] {
                            if dimension is .number(number) {
                                const parsed = parse_nonnegative(
                                    number.spelling,
                                    maximum
                                );
                                if parsed is none {
                                    return none;
                                }
                                if parsed != 0 and
                                    elements > maximum / parsed {
                                    return none;
                                }
                                elements = elements * parsed;
                                dimensions.push(parsed);
                            } else {
                                return none;
                            }
                        }
                        const offsets_value = &metadata.members[offsets_index].value;
                        if offsets_value is .array(offsets) {
                            if offsets.items.len() != 2 {
                                return none;
                            }
                            if offsets.items[0] is .number(first) {
                                if offsets.items[1] is .number(second) {
                                    const start = parse_nonnegative(
                                        first.spelling,
                                        maximum
                                    );
                                    const end = parse_nonnegative(
                                        second.spelling,
                                        maximum
                                    );
                                    if start is none {
                                        return none;
                                    }
                                    if end is none {
                                        return none;
                                    }
                                    if end < start or end > raw_size {
                                        return none;
                                    }
                                    const width = graph.dtype_width(dtype);
                                    if elements > maximum / width or
                                        elements * width != end - start {
                                        return none;
                                    }
                                    result.push(CheckpointTensor {
                                        name: member.key.clone(),
                                        dtype: dtype,
                                        dimensions: dimensions,
                                        start: start,
                                        end: end
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
    return none;
}

fn checkpoint_result(ctx: *c_void, argument_count: c_int,
    arguments_pointer: *const qjs.JSValue, inspect: bool
): qjs.JSValue {
    if argument_count < 1 {
        return qjs.type_error(ctx, "checkpoint.load expects a path");
    }
    const path = qjs.read_string(ctx, argument(arguments_pointer, 0));
    if path is none {
        return qjs.type_error(ctx, "checkpoint.load expects a string path");
    }
    const bytes = fs.read_bytes(path) catch (error_code) {
        const ignored = error_code;
        return qjs.type_error(
            ctx,
            "checkpoint.load failed to open file"
        );
    };
    if bytes.len() < 8 {
        return qjs.type_error(ctx, "checkpoint.load file is too small");
    }
    const header_size_u64 = little_u64(bytes[..], 0);
    if header_size_u64 is none {
        return qjs.type_error(ctx, "checkpoint.load truncated header");
    }
    if header_size_u64 > 16 * 1024 * 1024 {
        return qjs.type_error(ctx, "checkpoint.load has an invalid header");
    }
    if header_size_u64 > (bytes.len() - 8) as u64 {
        return qjs.type_error(ctx, "checkpoint.load has an invalid header");
    }
    const header_size = header_size_u64 as usize;
    const header_text = string.from(bytes[8..8 + header_size]);
    const parsed = json.parse(header_text) catch (error_code) {
        const ignored = error_code;
        return qjs.type_error(ctx, "checkpoint.load has invalid metadata");
    };
    const raw_size = bytes.len() - 8 - header_size;
    const entries = checkpoint_entries(&parsed, raw_size);
    if entries is none {
        return qjs.type_error(ctx, "checkpoint.load has invalid metadata");
    }
    var requested: List<string> = list<string>();
    const selective = not inspect and argument_count > 1 and
        argument(arguments_pointer, 1).tag != qjs.tag_undefined;
    if selective {
        const names_value = argument(arguments_pointer, 1);
        if qjs.is_array(names_value) == 0 {
            return qjs.type_error(
                ctx,
                "checkpoint.load names must be an array"
            );
        }
        const name_count = property_int(ctx, names_value, "length");
        if name_count is none {
            return qjs.type_error(ctx, "Invalid checkpoint names");
        }
        if name_count < 0 {
            return qjs.type_error(ctx, "Invalid checkpoint names");
        }
        var name_index: usize = 0;
        while name_index < name_count as usize {
            const item = qjs.get_property_uint32(
                ctx,
                names_value,
                name_index as c_uint
            );
            defer qjs.free_value(ctx, item);
            const name = qjs.read_string(ctx, item);
            if name is none {
                return qjs.type_error(
                    ctx,
                    "checkpoint names must be strings"
                );
            }
            for previous in requested[..] {
                if previous == name {
                    return qjs.type_error(ctx, "Duplicate checkpoint name");
                }
            }
            requested.push(name);
            name_index = name_index + 1;
        }
        for name in requested[..] {
            var found = false;
            for entry in entries[..] {
                if entry.name == name {
                    found = true;
                }
            }
            if not found {
                return qjs.type_error_string(
                    ctx,
                    $"Missing requested checkpoint tensor: {name}"
                );
            }
        }
    }
    const result = qjs.new_object(ctx);
    if qjs.is_exception(result) {
        return result;
    }
    for entry in entries[..] {
        if inspect {
            const info = qjs.new_object(ctx);
            const shape = qjs.new_array(ctx);
            var dimension_index: usize = 0;
            while dimension_index < entry.dimensions.len() {
                const ignored = qjs.set_property_uint32(
                    ctx,
                    shape,
                    dimension_index as c_uint,
                    qjs.new_number_from_f64(
                        ctx,
                        entry.dimensions[dimension_index] as f64
                    )
                );
                dimension_index = dimension_index + 1;
            }
            const ignored_shape = qjs.set_property_str(
                ctx,
                info,
                "shape",
                shape
            );
            qjs.set_string(
                ctx,
                info,
                "dtype",
                checkpoint_dtype_name(entry.dtype)
            );
            if qjs.set_dynamic_value(ctx, result, entry.name, info) < 0 {
                return qjs.type_error(
                    ctx,
                    "checkpoint.inspect failed to create result"
                );
            }
            continue;
        }
        var wanted = not selective;
        if selective {
            for name in requested[..] {
                if name == entry.name {
                    wanted = true;
                }
            }
        }
        if not wanted {
            continue;
        }
        const tensor_spec = spec(entry.dtype, entry.dimensions[..])
        catch (error_code) {
            const ignored = error_code;
            return qjs.type_error(
                ctx,
                "checkpoint.load invalid Tensor spec"
            );
        };
        const tensor = compute.tensor_bytes(
            tensor_spec,
            bytes[8 + header_size + entry.start..8 + header_size + entry.end]
        ) catch (error_code) {
            const ignored = error_code;
            return qjs.type_error(
                ctx,
                "checkpoint.load invalid Tensor"
            );
        };
        const object = create_tensor_object(
            ctx,
            tensor,
            maximum_sessions,
            0,
            true
        );
        if qjs.set_dynamic_value(ctx, result, entry.name, object) < 0 {
            return qjs.type_error(
                ctx,
                "checkpoint.load failed to create result"
            );
        }
    }
    return result;
}

fn save_native_callback(ctx: *c_void, this_value: qjs.JSValue,
    argument_count: c_int, arguments_pointer: *const qjs.JSValue
): qjs.JSValue {
    const ignored_this = this_value;
    return save_native(ctx, this_value, argument_count, arguments_pointer);
}

fn load_native_callback(ctx: *c_void, this_value: qjs.JSValue,
    argument_count: c_int, arguments_pointer: *const qjs.JSValue
): qjs.JSValue {
    const ignored_this = this_value;
    return checkpoint_result(ctx, argument_count, arguments_pointer, false);
}

fn inspect_checkpoint_callback(ctx: *c_void, this_value: qjs.JSValue,
    argument_count: c_int, arguments_pointer: *const qjs.JSValue
): qjs.JSValue {
    const ignored_this = this_value;
    return checkpoint_result(ctx, argument_count, arguments_pointer, true);
}

pub fn add_exports(ctx: *c_void, module_ptr: *c_void): c_int {
    if qjs.add_module_export(ctx, module_ptr, "defaultDevice") != 0 { return -1; }
    if qjs.add_module_export(ctx, module_ptr, "createSession") != 0 { return -1; }
    if qjs.add_module_export(ctx, module_ptr, "sessionTensor") != 0 { return -1; }
    if qjs.add_module_export(ctx, module_ptr, "sessionTensorBytes") != 0 { return -1; }
    if qjs.add_module_export(ctx, module_ptr, "sessionTensorCopy") != 0 { return -1; }
    if qjs.add_module_export(ctx, module_ptr, "sessionFull") != 0 { return -1; }
    if qjs.add_module_export(ctx, module_ptr, "compileProgram") != 0 { return -1; }
    if qjs.add_module_export(ctx, module_ptr, "explainExecutable") != 0 { return -1; }
    if qjs.add_module_export(ctx, module_ptr, "cynaSelectedSteps") != 0 { return -1; }
    if qjs.add_module_export(ctx, module_ptr, "cynaSelectedPhysicalNodes") != 0 { return -1; }
    if qjs.add_module_export(ctx, module_ptr, "selectedPhysicalValues") != 0 { return -1; }
    if qjs.add_module_export(ctx, module_ptr, "selectedPhysicalLiveness") != 0 { return -1; }
    if qjs.add_module_export(ctx, module_ptr, "cynaResidualDecisions") != 0 { return -1; }
    if qjs.add_module_export(ctx, module_ptr, "cynaSelectionComparison") != 0 { return -1; }
    if qjs.add_module_export(ctx, module_ptr, "cynaSearchEvidence") != 0 { return -1; }
    if qjs.add_module_export(ctx, module_ptr, "executableRouting") != 0 { return -1; }
    if qjs.add_module_export(ctx, module_ptr, "executableBackend") != 0 { return -1; }
    if qjs.add_module_export(ctx, module_ptr, "runExecutable") != 0 { return -1; }
    if qjs.add_module_export(ctx, module_ptr, "resetEvidenceCapture") != 0 { return -1; }
    if qjs.add_module_export(ctx, module_ptr, "exportEvidenceCapture") != 0 { return -1; }
    if qjs.add_module_export(ctx, module_ptr, "stftPower") != 0 { return -1; }
    if qjs.add_module_export(ctx, module_ptr, "filterbank") != 0 { return -1; }
    if qjs.add_module_export(ctx, module_ptr, "saveNative") != 0 { return -1; }
    if qjs.add_module_export(ctx, module_ptr, "loadNative") != 0 { return -1; }
    if qjs.add_module_export(ctx, module_ptr, "inspectCheckpoint") != 0 { return -1; }
    if qjs.add_module_export(ctx, module_ptr, "default") != 0 { return -1; }
    return 0;
}

fn install_function(ctx: *c_void, object: qjs.JSValue, name: c_str,
    callback: qjs.Callback, argument_count: c_int, module_ptr: *c_void,
    export_name: c_str
): c_int {
    const value = qjs.new_c_function(
        ctx, callback, name, argument_count, 0, 0
    );
    if qjs.is_exception(value) {
        print("affon native module: C function creation failed\n");
        return -1;
    }
    const object_tag = object.tag;
    const function_tag = value.tag;
    if qjs.set_property_str(ctx, object, name, value) < 0 {
        print($"affon native module: property installation failed (object tag {object_tag}, function tag {function_tag})\n");
        const exception = qjs.get_exception(ctx);
        const message = qjs.read_string(ctx, exception);
        if message is text {
            print($"affon native module: {text}\n");
        }
        qjs.free_value(ctx, exception);
        return -1;
    }
    const exported = qjs.get_property_str(ctx, object, name);
    defer qjs.free_value(ctx, exported);
    if qjs.set_module_export(
        ctx, module_ptr, export_name, qjs.dup_value(ctx, exported)
    ) < 0 {
        print("affon native module: export installation failed\n");
        return -1;
    }
    return 0;
}

pub fn initialize(ctx: *c_void, module_ptr: *c_void): c_int {
    const object = qjs.new_object(ctx);
    if qjs.is_exception(object) {
        print("affon native module: object creation failed\n");
        return -1;
    }
    if install_function(ctx, object, "defaultDevice", default_device, 0,
        module_ptr, "defaultDevice"
    ) != 0 { return -1; }
    if install_function(ctx, object, "createSession", create_session, 2,
        module_ptr, "createSession"
    ) != 0 { return -1; }
    if install_function(ctx, object, "sessionTensor", session_tensor, 4,
        module_ptr, "sessionTensor"
    ) != 0 { return -1; }
    if install_function(ctx, object, "sessionTensorBytes", session_tensor_bytes, 4,
        module_ptr, "sessionTensorBytes"
    ) != 0 { return -1; }
    if install_function(ctx, object, "sessionTensorCopy", session_tensor_copy, 2,
        module_ptr, "sessionTensorCopy"
    ) != 0 { return -1; }
    if install_function(ctx, object, "sessionFull", session_full, 4,
        module_ptr, "sessionFull"
    ) != 0 { return -1; }
    if install_function(ctx, object, "compileProgram", compile_program, 3,
        module_ptr, "compileProgram"
    ) != 0 { return -1; }
    if install_function(ctx, object, "explainExecutable", explain_executable, 1,
        module_ptr, "explainExecutable"
    ) != 0 { return -1; }
    if install_function(ctx, object, "cynaSelectedSteps", selected_steps, 1,
        module_ptr, "cynaSelectedSteps"
    ) != 0 { return -1; }
    if install_function(ctx, object, "cynaSelectedPhysicalNodes",
        selected_physical_nodes, 1,
        module_ptr, "cynaSelectedPhysicalNodes"
    ) != 0 { return -1; }
    if install_function(ctx, object, "selectedPhysicalValues",
        selected_physical_values, 1,
        module_ptr, "selectedPhysicalValues"
    ) != 0 { return -1; }
    if install_function(ctx, object, "selectedPhysicalLiveness",
        selected_physical_liveness, 1,
        module_ptr, "selectedPhysicalLiveness"
    ) != 0 { return -1; }
    if install_function(ctx, object, "cynaResidualDecisions",
        residual_decisions, 1,
        module_ptr, "cynaResidualDecisions"
    ) != 0 { return -1; }
    if install_function(ctx, object, "cynaSelectionComparison",
        selection_comparison, 1,
        module_ptr, "cynaSelectionComparison"
    ) != 0 { return -1; }
    if install_function(ctx, object, "cynaSearchEvidence", search_evidence, 1,
        module_ptr, "cynaSearchEvidence"
    ) != 0 { return -1; }
    if install_function(ctx, object, "executableRouting", executable_routing, 1,
        module_ptr, "executableRouting"
    ) != 0 { return -1; }
    if install_function(ctx, object, "executableBackend", executable_backend, 1,
        module_ptr, "executableBackend"
    ) != 0 { return -1; }
    if install_function(ctx, object, "runExecutable", run_executable, 2,
        module_ptr, "runExecutable"
    ) != 0 { return -1; }
    if install_function(ctx, object, "resetEvidenceCapture", reset_evidence, 1,
        module_ptr, "resetEvidenceCapture"
    ) != 0 { return -1; }
    if install_function(ctx, object, "exportEvidenceCapture", export_evidence, 2,
        module_ptr, "exportEvidenceCapture"
    ) != 0 { return -1; }
    if install_function(ctx, object, "stftPower", stft_power, 5,
        module_ptr, "stftPower"
    ) != 0 { return -1; }
    if install_function(ctx, object, "filterbank", filterbank, 2,
        module_ptr, "filterbank"
    ) != 0 { return -1; }
    if install_function(ctx, object, "saveNative", save_native_callback, 2,
        module_ptr, "saveNative"
    ) != 0 { return -1; }
    if install_function(ctx, object, "loadNative", load_native_callback, 2,
        module_ptr, "loadNative"
    ) != 0 { return -1; }
    if install_function(ctx, object, "inspectCheckpoint",
        inspect_checkpoint_callback, 1,
        module_ptr, "inspectCheckpoint"
    ) != 0 { return -1; }
    if qjs.set_module_export(ctx, module_ptr, "default", object) < 0 {
        print("affon native module: default export failed\n");
        return -1;
    }
    return 0;
}
