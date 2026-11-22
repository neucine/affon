import std.json;
import compute_core.lib as compute;
import {
    PhysicalExecutionClass,
    PhysicalValueOrigin,
    ResidualPolicy,
    StepImplementation,
    Task
} from compute_core.compiler.executable;

fn number(value: usize): json.Value {
    const encoded: json.Number = {
        spelling: $"{value}"
    };
    const result: json.Value = .number(encoded);
    return result;
}

fn boolean(value: bool): json.Value {
    const result: json.Value = .bool(value);
    return result;
}

fn string_value(value: string): json.Value {
    const result: json.Value = .string(value);
    return result;
}

fn null_value(): json.Value {
    const result: json.Value = .null_value;
    return result;
}

fn member(key: string, value: json.Value): json.Member {
    const result: json.Member = {
        key: key,
        value: value
    };
    return result;
}

fn object(members: List<json.Member>): json.Value {
    const encoded: json.Object = { members: members };
    const result: json.Value = .object(encoded);
    return result;
}

fn array(items: List<json.Value>): json.Value {
    const encoded: json.Array = { items: items };
    const result: json.Value = .array(encoded);
    return result;
}

fn encode(value: &const json.Value): string {
    return json.encode(value);
}

pub fn operation_name(tag: compute.OpTag): string {
    return switch tag {
        .add => "add";
        .sub => "sub";
        .mul => "mul";
        .div => "div";
        .matmul => "matmul";
        .abs => "abs";
        .neg => "neg";
        .sign => "sign";
        .relu => "relu";
        .sum_all => "sum_all";
        .mean_all => "mean_all";
        .sum_axis => "sum_axis";
        .mean_axis => "mean_axis";
        .min_all => "min_all";
        .max_all => "max_all";
        .argmin_all => "argmin_all";
        .argmax_all => "argmax_all";
        .min_axis => "min_axis";
        .max_axis => "max_axis";
        .argmin_axis => "argmin_axis";
        .argmax_axis => "argmax_axis";
        .variance_all => "variance_all";
        .std_all => "std_all";
        .variance_axis => "variance_axis";
        .std_axis => "std_axis";
        .reshape => "reshape";
        .permute => "permute";
        .transpose => "transpose";
        .squeeze => "squeeze";
        .unsqueeze => "unsqueeze";
        .contiguous => "contiguous";
        .slice => "slice";
        .cast => "cast";
        .scatter_add => "scatter_add";
        .gather => "gather";
        .index_select => "index_select";
        .embedding => "embedding";
        .one_hot => "one_hot";
        .top_k_values => "top_k_values";
        .top_k_indices => "top_k_indices";
        .eq => "eq";
        .lt => "lt";
        .gt => "gt";
        .where => "where";
        .masked_fill => "masked_fill";
        .cat => "cat";
        .stack => "stack";
        .clamp => "clamp";
        .exp => "exp";
        .log => "log";
        .sqrt => "sqrt";
        .sigmoid => "sigmoid";
        .silu => "silu";
        .tanh => "tanh";
        .erf => "erf";
        .gelu => "gelu";
        .sin => "sin";
        .cos => "cos";
        .softmax => "softmax";
        .log_softmax => "log_softmax";
        .cross_entropy => "cross_entropy";
        .cross_entropy_indexed => "cross_entropy_indexed";
        .layer_norm => "layer_norm";
        .rms_norm => "rms_norm";
        .quantized_matmul => "quantized_matmul";
        .log_softmax_nll => "log_softmax_nll";
        .dot => "dot";
        .reduce_to_shape => "reduce_to_shape";
    };
}

fn physical_implementation(node: &const Task): string {
    return switch node.action {
        .instruction(selected) => string.from(selected.implementation_id[..]);
        .conversion(selected) => string.from(selected.implementation_id[..]);
    };
}

fn physical_action(node: &const Task): string {
    return switch node.action {
        .instruction(selected) => switch selected.execution_class {
            .view => "view";
            _ => "instruction";
        };
        .conversion => "conversion";
    };
}

fn physical_is_view(node: &const Task): bool {
    return switch node.action {
        .instruction(selected) => selected.execution_class ==
            PhysicalExecutionClass.view;
        .conversion => false;
    };
}

fn spec_bytes(spec: &const compute.TensorSpec): usize {
    const count = spec.element_count() catch (error_code) {
        const ignored = error_code;
        return 0;
    };
    return count * width(spec.dtype);
}

fn width(dtype: compute.DType): usize {
    return switch dtype {
        .f32 => 4;
        .u32 => 4;
        .f64 => 8;
        .i64 => 8;
        .bf16 => 2;
    };
}

fn node_output_bytes(node: &const Task,
    executable: &const compute.Executable
): usize {
    var total: usize = 0;
    for output in node.outputs[..] {
        if output < executable.graph.values.len() {
            total = total + spec_bytes(&executable.graph.values[output].spec);
        }
    }
    return total;
}

fn contains(items: &const List<usize>, value: usize): bool {
    for item in items[..] {
        if item == value {
            return true;
        }
    }
    return false;
}

fn program_origins(origins: &const List<usize>,
    source_instruction_by_step: &const List<usize>
): List<usize> {
    var result: List<usize> = list<usize>();
    for origin in origins[..] {
        if origin < source_instruction_by_step.len() {
            const instruction = source_instruction_by_step[origin];
            if not contains(&result, instruction) {
                result.push(instruction);
            }
        }
    }
    return result;
}

fn usize_array(values: &const List<usize>): json.Value {
    var items: List<json.Value> = list<json.Value>();
    for value in values[..] {
        items.push(number(value));
    }
    return array(items);
}

pub fn selected_steps(executable: &const compute.Executable): string {
    var items: List<json.Value> = list<json.Value>();
    for step in executable.semantic_steps[..] {
        var fields: List<json.Member> = list<json.Member>();
        fields.push(member("step_index", number(step.step_index)));
        fields.push(member("source_instruction", number(step.source_index)));
        fields.push(member("implementation_code", number(
            step.implementation as usize
        )));
        if step.operation is operation {
            fields.push(member("operation", string_value(operation_name(operation))));
        } else {
            fields.push(member("operation", null_value()));
        }
        items.push(object(fields));
    }
    const result = array(items);
    return encode(&result);
}

pub fn selected_physical_nodes(executable: &const compute.Executable,
    source_instruction_by_step: &const List<usize>
): string {
    var items: List<json.Value> = list<json.Value>();
    var node_index: usize = 0;
    while node_index < executable.graph.nodes.len() {
        const node = &executable.graph.nodes[node_index];
        const origins = program_origins(
            &node.origins,
            source_instruction_by_step
        );
        var fields: List<json.Member> = list<json.Member>();
        fields.push(member("id", number(node_index)));
        fields.push(member("implementation", string_value(
            physical_implementation(node)
        )));
        fields.push(member("physical_action", string_value(
            physical_action(node)
        )));
        if node.step_index is step_index {
            fields.push(member("implementation_code", number(step_index + 1)));
        } else {
            fields.push(member("implementation_code", number(0)));
        }
        fields.push(member("source_instruction_count", number(
            origins.len()
        )));
        fields.push(member("origins", usize_array(&node.origins)));
        fields.push(member("program_origins", usize_array(&origins)));
        fields.push(member("physical_output_bytes", number(
            node_output_bytes(node, executable)
        )));
        fields.push(member("physical_is_view", boolean(
            physical_is_view(node)
        )));
        var operation: ?compute.OpTag = node.instruction.tag;
        if node.source_tag is source_tag {
            operation = source_tag;
        }
        if operation is selected_operation {
            fields.push(member("operation", string_value(
                operation_name(selected_operation)
            )));
            fields.push(member("source_operation_code", null_value()));
        } else {
            fields.push(member("operation", null_value()));
            fields.push(member("source_operation_code", null_value()));
        }
        items.push(object(fields));
        node_index = node_index + 1;
    }
    const result = array(items);
    return encode(&result);
}

fn origin_name(origin: PhysicalValueOrigin): string {
    return switch origin {
        .input => "input";
        .constant => "constant";
        .node_output => "node_output";
    };
}

pub fn selected_physical_values(executable: &const compute.Executable): string {
    var items: List<json.Value> = list<json.Value>();
    var value_id: usize = 0;
    while value_id < executable.graph.values.len() {
        const value = &executable.graph.values[value_id];
        var fields: List<json.Member> = list<json.Member>();
        fields.push(member("id", number(value.id)));
        fields.push(member("origin", string_value(origin_name(value.origin))));
        fields.push(member("byte_count", number(spec_bytes(&value.spec))));
        fields.push(member("is_view", boolean(value.alias_of != none)));
        items.push(object(fields));
        value_id = value_id + 1;
    }
    const result = array(items);
    return encode(&result);
}

pub fn selected_physical_liveness(
    executable: &const compute.Executable
): string {
    var output_bytes: usize = 0;
    for output in executable.graph.output_bindings[..] {
        output_bytes = output_bytes + spec_bytes(
            &executable.graph.values[output].spec
        );
    }
    const peak_active = executable.schedule.peak_active_bytes;
    const peak_live = peak_active + output_bytes;
    var fields: List<json.Member> = list<json.Member>();
    fields.push(member("peak_active_bytes", number(peak_active)));
    fields.push(member("peak_live_bytes", number(peak_live)));
    fields.push(member("persistent_bytes", number(
        executable.schedule.persistent_bytes
    )));
    fields.push(member("external_input_bytes", number(
        executable.schedule.external_input_bytes
    )));
    const result = object(fields);
    return encode(&result);
}

pub fn residual_decisions(executable: &const compute.Executable): string {
    var items: List<json.Value> = list<json.Value>();
    for need in executable.graph.residual_needs[..] {
        var uses: List<json.Value> = list<json.Value>();
        for use in need.uses[..] {
            var use_fields: List<json.Member> = list<json.Member>();
            use_fields.push(member("instruction", number(
                use.instruction
            )));
            use_fields.push(member("input_index", number(
                use.input_index as usize
            )));
            uses.push(object(use_fields));
        }
        var fields: List<json.Member> = list<json.Member>();
        fields.push(member("value", number(need.value)));
        fields.push(member("uses", array(uses)));
        var retained = false;
        if need.value < executable.graph.values.len() {
            retained =
                executable.graph.values[need.value].origin !=
                PhysicalValueOrigin.input;
        }
        fields.push(member("retained", boolean(retained)));
        var policy = "retain";
        if executable.residual_policy == ResidualPolicy.recompute {
            policy = "recompute";
        } else if executable.residual_policy == ResidualPolicy.automatic {
            policy = "automatic";
        }
        fields.push(member("recompute_policy", string_value(policy)));
        items.push(object(fields));
    }
    const result = array(items);
    return encode(&result);
}

pub struct RuntimeEvidence {
    pub run_count: usize;
    pub failed_run_count: usize;
    pub cache_hits: usize;
    pub cache_misses: usize;
    pub captured_count: usize;
    pub capture_capacity: usize;
    pub dropped_count: usize;
    pub timing_scope: string;
    pub elapsed_ns: u64;
    pub hardware_metrics: bool;
    pub profiler_markers: bool;
}

fn physical_implementation_for(node: &const Task): string {
    return physical_implementation(node);
}

pub fn evidence(executable: &const compute.Executable,
    source_instruction_by_step: &const List<usize>, stats: RuntimeEvidence,
    backend_name: string
): string {
    var entries: List<json.Value> = list<json.Value>();
    var cache_bytes: usize = 0;
    var node_index: usize = 0;
    while node_index < executable.graph.nodes.len() {
        const node = &executable.graph.nodes[node_index];
        const origins = program_origins(
            &node.origins,
            source_instruction_by_step
        );
        const is_view = physical_is_view(node);
        const output_bytes = node_output_bytes(node, executable);
        if not is_view {
            cache_bytes = cache_bytes + output_bytes;
        }
        var fields: List<json.Member> = list<json.Member>();
        fields.push(member("step_index", number(node_index)));
        var graph_kind = "prepared_physical";
        if is_view {
            graph_kind = "resident_view";
        }
        fields.push(member("graph_kind", string_value(graph_kind)));
        fields.push(member("operation", string_value(
            operation_name(node.instruction.tag)
        )));
        fields.push(member("source_operation_code", null_value()));
        fields.push(member("source_instruction_count", number(
            origins.len()
        )));
        fields.push(member("origins", usize_array(&node.origins)));
        fields.push(member("program_origins", usize_array(&origins)));
        fields.push(member("physical_implementation", string_value(
            physical_implementation_for(node)
        )));
        fields.push(member("physical_action", string_value(
            physical_action(node)
        )));
        fields.push(member("physical_is_view", boolean(is_view)));
        fields.push(member("physical_output_bytes", number(output_bytes)));
        if stats.run_count == 0 {
            fields.push(member("observed_schedule_position", null_value()));
        } else {
            fields.push(member("observed_schedule_position", number(
                node_index
            )));
        }
        fields.push(member("metadata_available", boolean(true)));
        fields.push(member("invocation_count", number(stats.run_count)));
        if stats.failed_run_count > stats.run_count {
            fields.push(member("succeeded_count", number(0)));
        } else {
            fields.push(member("succeeded_count", number(
                stats.run_count - stats.failed_run_count
            )));
        }
        fields.push(member("failed_count", number(
            stats.failed_run_count
        )));
        fields.push(member("materialized_view_count", number(0)));
        fields.push(member("materialized_view_bytes", number(0)));
        fields.push(member("timing_scope", string_value(
            stats.timing_scope
        )));
        fields.push(member("timing_source", string_value(
            "host_synchronous"
        )));
        if stats.timing_scope == "kernel" {
            fields.push(member("invocation_timing_sample_count", number(
                0
            )));
            fields.push(member("kernel_timing_sample_count", number(
                stats.run_count
            )));
        } else if stats.timing_scope == "none" {
            fields.push(member("invocation_timing_sample_count", number(
                0
            )));
            fields.push(member("kernel_timing_sample_count", number(0)));
        } else {
            fields.push(member("invocation_timing_sample_count", number(
                stats.run_count
            )));
            fields.push(member("kernel_timing_sample_count", number(0)));
        }
        fields.push(member("timing_unavailable_count", number(0)));
        fields.push(member("total_ns", number(stats.elapsed_ns as usize)));
        if stats.run_count == 0 {
            fields.push(member("minimum_ns", null_value()));
            fields.push(member("median_ns", null_value()));
            fields.push(member("p95_ns", null_value()));
            fields.push(member("maximum_ns", null_value()));
        } else {
            fields.push(member("minimum_ns", number(
                stats.elapsed_ns as usize
            )));
            fields.push(member("median_ns", number(
                stats.elapsed_ns as usize
            )));
            fields.push(member("p95_ns", number(
                stats.elapsed_ns as usize
            )));
            fields.push(member("maximum_ns", number(
                stats.elapsed_ns as usize
            )));
        }
        if stats.run_count == 0 {
            fields.push(member("kernel_count", null_value()));
        } else if stats.dropped_count != 0 {
            fields.push(member("kernel_count", null_value()));
        } else if is_view {
            fields.push(member("kernel_count", number(0)));
        } else {
            fields.push(member("kernel_count", number(stats.run_count)));
        }
        fields.push(member("library_call_count", null_value()));
        fields.push(member("custom_dispatch_count", null_value()));
        entries.push(object(fields));
        node_index = node_index + 1;
    }

    var hardware: List<json.Value> = list<json.Value>();
    const metric_names: [3]string = ["bandwidth", "occupancy", "cache"];
    for metric in metric_names {
        var metric_fields: List<json.Member> = list<json.Member>();
        metric_fields.push(member("metric", string_value(metric)));
        metric_fields.push(member("availability", string_value(
            "unsupported"
        )));
        hardware.push(object(metric_fields));
    }

    var fields: List<json.Member> = list<json.Member>();
    fields.push(member("schema_version", number(10)));
    fields.push(member("suite", string_value("compute_observability")));
    fields.push(member("program_id", number(0)));
    fields.push(member("program_id_text", string_value("0")));
    fields.push(member("executable_id", number(0)));
    fields.push(member("executable_id_text", string_value("0")));
    fields.push(member("backend", string_value(backend_name)));
    fields.push(member("timing_scope", string_value(stats.timing_scope)));
    var views_only = executable.graph.nodes.len() != 0;
    for node in executable.graph.nodes[..] {
        if not physical_is_view(&node) {
            views_only = false;
        }
    }
    var execution_path = "prepared";
    var fallback_reason = "no_fallback";
    if views_only {
        execution_path = "resident_view";
        fallback_reason = "view_path_selected";
    }
    fields.push(member("execution_path", string_value(execution_path)));
    fields.push(member("fallback_reason", string_value(fallback_reason)));
    fields.push(member("prepared_cache_hits", number(stats.cache_hits)));
    fields.push(member("prepared_cache_misses", number(
        stats.cache_misses
    )));
    fields.push(member("prepared_cache_bytes", number(cache_bytes)));
    fields.push(member("search", search_value(executable)));
    fields.push(member("preparation", null_value()));
    fields.push(member("resources", null_value()));
    fields.push(member("entries", array(entries)));
    fields.push(member("observed_run_count", number(stats.run_count)));
    fields.push(member("observed_failed_run_count", number(
        stats.failed_run_count
    )));
    fields.push(member("dropped_event_count", number(stats.dropped_count)));
    fields.push(member("run_count_complete", boolean(
        stats.dropped_count == 0
    )));
    if stats.profiler_markers {
        var profiler_fields: List<json.Member> = list<json.Member>();
        profiler_fields.push(member("provider", string_value("nvtx")));
        profiler_fields.push(member("available", boolean(false)));
        fields.push(member(
            "profiler_marker_availability",
            object(profiler_fields)
        ));
    } else {
        fields.push(member("profiler_marker_availability", null_value()));
    }
    fields.push(member("hardware_metric_availability", array(hardware)));
    var kernel_count_evidence = "exact";
    if stats.dropped_count != 0 {
        kernel_count_evidence = "unavailable";
    }
    fields.push(member("kernel_count_evidence", string_value(
        kernel_count_evidence
    )));
    fields.push(member("phase_memory_available", boolean(false)));
    fields.push(member("phase_memory", array(list<json.Value>())));
    fields.push(member("cost_validations", array(list<json.Value>())));
    const result = object(fields);
    return encode(&result);
}

pub fn selection_comparison(
    executable: &const compute.Executable
): string {
    const evidence = &executable.evidence;
    var fields: List<json.Member> = list<json.Member>();
    fields.push(member("evaluated_states", number(evidence.evaluated_states)));
    fields.push(member("budget", number(evidence.budget)));
    fields.push(member("exclusion_search_truncated", boolean(
        evidence.exclusion_truncated
    )));
    if evidence.baseline_peak_bytes is baseline {
        fields.push(member("baseline_peak_bytes", number(baseline)));
    } else {
        fields.push(member("baseline_peak_bytes", null_value()));
    }
    if evidence.candidate_peak_bytes is candidate {
        fields.push(member("candidate_peak_bytes", number(candidate)));
    } else {
        fields.push(member("candidate_peak_bytes", null_value()));
    }
    fields.push(member("candidate_published", boolean(
        evidence.candidate_published
    )));
    var constant_bytes: usize = 0;
    for value in executable.graph.values[..] {
        if value.origin == PhysicalValueOrigin.constant {
            constant_bytes = constant_bytes + spec_bytes(&value.spec);
        }
    }
    fields.push(member("constant_input_bytes", number(constant_bytes)));
    const result = object(fields);
    return encode(&result);
}

fn search_value(executable: &const compute.Executable): json.Value {
    const evidence = &executable.evidence;
    var limits: List<json.Member> = list<json.Member>();
    limits.push(member("structural_states", number(
        evidence.limits.structural_states
    )));
    limits.push(member("structural_expansions", number(
        evidence.limits.structural_expansions
    )));
    limits.push(member("complete_evaluations", number(
        evidence.limits.complete_evaluations
    )));
    limits.push(member("physical_work", number(
        evidence.limits.physical_work
    )));
    limits.push(member("ready_work", number(
        evidence.limits.ready_work
    )));
    if evidence.limits.max_peak_active_bytes is maximum {
        limits.push(member("max_peak_active_bytes", number(maximum)));
    } else {
        limits.push(member("max_peak_active_bytes", null_value()));
    }
    if evidence.limits.max_invocation_scratch_bytes is maximum {
        limits.push(member("max_invocation_scratch_bytes", number(maximum)));
    } else {
        limits.push(member("max_invocation_scratch_bytes", null_value()));
    }
    var fields: List<json.Member> = list<json.Member>();
    fields.push(member("limits", object(limits)));
    fields.push(member("structural_states", number(
        evidence.limits.structural_states
    )));
    fields.push(member("structural_expansions", number(
        evidence.limits.structural_expansions
    )));
    fields.push(member("complete_evaluations", number(
        evidence.limits.complete_evaluations
    )));
    fields.push(member("physical_work", number(
        evidence.limits.physical_work
    )));
    fields.push(member("ready_work", number(
        evidence.limits.ready_work
    )));
    if evidence.limits.max_peak_active_bytes is maximum {
        fields.push(member("max_peak_active_bytes", number(maximum)));
    } else {
        fields.push(member("max_peak_active_bytes", null_value()));
    }
    if evidence.limits.max_invocation_scratch_bytes is maximum {
        fields.push(member(
            "max_invocation_scratch_bytes",
            number(maximum)
        ));
    } else {
        fields.push(member("max_invocation_scratch_bytes", null_value()));
    }
    fields.push(member("evaluated_states", number(
        evidence.search_evaluated_states
    )));
    fields.push(member("structural_expansions", number(
        evidence.search_structural_expansions
    )));
    fields.push(member("complete_evaluations", number(
        evidence.search_complete_evaluations
    )));
    fields.push(member("truncated", boolean(evidence.search_truncated)));
    fields.push(member("physical_work_checked", number(
        evidence.physical_work_checked
    )));
    fields.push(member("physical_work_truncated", boolean(
        evidence.physical_work_truncated
    )));
    return object(fields);
}

pub fn search(executable: &const compute.Executable): string {
    const result = search_value(executable);
    return encode(&result);
}

pub fn explanation(executable: &const compute.Executable,
    _source_instruction_by_step: &const List<usize>
): string {
    var fields: List<json.Member> = list<json.Member>();
    fields.push(member("schema_version", number(2)));
    fields.push(member("kind", string_value("program")));
    fields.push(member("execution_backend", string_value("cyna")));
    fields.push(member("executable_plan_backend", string_value("cyna")));
    fields.push(member("actual_plan_report", string_value(
        "Affon and Compute Core selected this exact Cyna executable plan"
    )));
    fields.push(member("selected_physical_values", array(
        list<json.Value>()
    )));
    fields.push(member("selected_physical_liveness", object(
        list<json.Member>()
    )));
    fields.push(member("cyna_selected_semantic_steps", array(
        list<json.Value>()
    )));
    fields.push(member("cyna_selected_physical_nodes", array(
        list<json.Value>()
    )));
    fields.push(member("cyna_residual_decisions", array(
        list<json.Value>()
    )));
    fields.push(member("cyna_selection_comparison", object(
        list<json.Member>()
    )));
    fields.push(member("cyna_search", search_value(executable)));
    var search_limits: List<json.Member> = list<json.Member>();
    search_limits.push(member("structural_states", number(
        executable.evidence.limits.structural_states
    )));
    search_limits.push(member("structural_expansions", number(
        executable.evidence.limits.structural_expansions
    )));
    search_limits.push(member("complete_evaluations", number(
        executable.evidence.limits.complete_evaluations
    )));
    search_limits.push(member("physical_work", number(
        executable.evidence.limits.physical_work
    )));
    search_limits.push(member("ready_work", number(
        executable.evidence.limits.ready_work
    )));
    var selection: List<json.Member> = list<json.Member>();
    selection.push(member("search_limits", object(search_limits)));
    selection.push(member("selected_peak_active_bytes", number(
        executable.schedule.peak_active_bytes
    )));
    var executable_fields: List<json.Member> = list<json.Member>();
    executable_fields.push(member("selection", object(selection)));
    executable_fields.push(member("optimizations", array(
        list<json.Value>()
    )));
    fields.push(member("executable", object(executable_fields)));
    const result = object(fields);
    return encode(&result);
}
