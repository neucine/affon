const std = @import("std");
const compute = @import("compute");
const core = @import("compute_core_cyna.zig");

const Entry = struct {
    step_index: usize,
    graph_kind: []const u8,
    operation: ?[]const u8,
    source_operation_code: ?usize,
    source_instruction_count: ?usize,
    physical_implementation: ?[]const u8,
    physical_action: ?[]const u8 = null,
    physical_is_view: ?bool = null,
    physical_output_bytes: ?usize = null,
    observed_schedule_position: ?usize = null,
    metadata_available: bool,
    invocation_count: usize = 0,
    succeeded_count: usize = 0,
    failed_count: usize = 0,
    materialized_view_count: usize = 0,
    materialized_view_bytes: ?usize = null,
    timing_scope: []const u8 = "none",
    timing_source: []const u8 = "host_synchronous",
    invocation_timing_sample_count: usize = 0,
    kernel_timing_sample_count: usize = 0,
    timing_unavailable_count: usize = 0,
    total_ns: u64 = 0,
    minimum_ns: ?u64 = null,
    median_ns: ?u64 = null,
    p95_ns: ?u64 = null,
    maximum_ns: ?u64 = null,
    kernel_count: ?usize = null,
    library_call_count: ?usize = null,
    custom_dispatch_count: ?usize = null,
};

const MutableEntry = struct {
    entry: Entry,
    implementation_owned: ?[]u8 = null,
    samples: std.ArrayList(u64) = .empty,
    schedule_position_conflict: bool = false,

    fn deinit(self: *MutableEntry, allocator: std.mem.Allocator) void {
        if (self.implementation_owned) |text| allocator.free(text);
        self.samples.deinit(allocator);
    }
};

const ProfilerMarkerAvailability = struct {
    provider: []const u8,
    available: bool,
};

const HardwareMetricAvailability = struct {
    metric: []const u8,
    availability: []const u8,
};

const SearchEvidence = struct {
    structural_states: usize,
    structural_expansions: usize,
    complete_evaluations: usize,
    physical_work: usize,
    ready_work: usize,
    max_peak_active_bytes: ?usize,
    max_invocation_scratch_bytes: ?usize,
    evaluated_states: usize,
    applied_structural_expansions: usize,
    applied_complete_evaluations: usize,
    truncated: bool,
    ready_truncated: bool,
};

fn graphName(kind: compute.ExternalGraphKind) []const u8 {
    return @tagName(kind);
}

fn findEntry(entries: []MutableEntry, kind: compute.ExternalGraphKind, node_id: usize) ?usize {
    for (entries, 0..) |value, index| {
        if (value.entry.step_index == node_id and
            std.mem.eql(u8, value.entry.graph_kind, graphName(kind))) return index;
    }
    return null;
}

fn makeEntry(
    allocator: std.mem.Allocator,
    handle: *const core.DenseGraphHandle,
    kind: compute.ExternalGraphKind,
    node_id: usize,
) !MutableEntry {
    var entry: Entry = .{
        .step_index = node_id,
        .graph_kind = graphName(kind),
        .operation = null,
        .source_operation_code = null,
        .source_instruction_count = null,
        .physical_implementation = null,
        .metadata_available = false,
    };
    var owned: ?[]u8 = null;
    if (kind == .prepared_physical and node_id < handle.physicalNodeCount()) {
        var implementation_buffer: [256]u8 = undefined;
        const node = try handle.describePhysicalNode(node_id, &implementation_buffer);
        owned = try allocator.dupe(u8, node.implementation);
        entry.operation = core.operationName(node.source_operation_code);
        entry.source_operation_code = node.source_operation_code;
        entry.source_instruction_count = node.source_instruction_count;
        entry.physical_implementation = owned;
        entry.physical_action = @tagName(node.action);
        entry.physical_is_view = node.is_view;
        entry.physical_output_bytes = node.output_bytes;
        entry.materialized_view_bytes = 0;
        entry.metadata_available = true;
    } else if (kind != .prepared_physical and node_id < handle.semanticStepCount()) {
        const step = try handle.describeSemanticStep(node_id);
        entry.operation = core.operationName(step.source_operation_code);
        entry.source_operation_code = step.source_operation_code;
        entry.source_instruction_count = if (step.source_operation_code == null) 0 else 1;
        entry.metadata_available = true;
    }
    return .{ .entry = entry, .implementation_owned = owned };
}

fn percentile(sorted: []const u64, numerator: usize, denominator: usize) u64 {
    const rank = (sorted.len * numerator + denominator - 1) / denominator;
    return sorted[@max(1, rank) - 1];
}

/// Serialize only facts observed on the Cyna route. A nullable count means
/// the backend did not measure it; zero means a measured absence.
pub fn serialize(
    allocator: std.mem.Allocator,
    source: *const compute.Executable,
    handle: *const core.DenseGraphHandle,
    records: []const compute.EvidenceRecord,
    collector_dropped: usize,
    cyna_dropped: usize,
    timing_scope: []const u8,
) ![]u8 {
    var mutable: std.ArrayList(MutableEntry) = .empty;
    defer {
        for (mutable.items) |*entry| entry.deinit(allocator);
        mutable.deinit(allocator);
    }
    const executable_id = source.id().int();
    const last_path = if (handle.runState()) |state| state.path else |_| core.DenseGraphHandle.ExecutionPath.unknown;
    const search_evidence = handle.searchEvidence() catch null;
    var preparation = handle.preparationEvidence(allocator) catch null;
    const resources = handle.resourceEvidence() catch null;
    defer {
        if (preparation) |*value| value.deinit(allocator);
    }
    const search: ?SearchEvidence = if (search_evidence) |evidence| .{
        .structural_states = evidence.limits.structural_states,
        .structural_expansions = evidence.limits.structural_expansions,
        .complete_evaluations = evidence.limits.complete_evaluations,
        .physical_work = evidence.limits.physical_work,
        .ready_work = evidence.limits.ready_work,
        .max_peak_active_bytes = evidence.limits.max_peak_active_bytes,
        .max_invocation_scratch_bytes = evidence.limits.max_invocation_scratch_bytes,
        .evaluated_states = evidence.evaluated_states,
        .applied_structural_expansions = evidence.structural_expansions,
        .applied_complete_evaluations = evidence.complete_evaluations,
        .truncated = evidence.truncated,
        .ready_truncated = evidence.ready_truncated,
    } else null;
    if (last_path == .prepared) {
        for (0..handle.physicalNodeCount()) |node_id|
            try mutable.append(allocator, try makeEntry(allocator, handle, .prepared_physical, node_id));
    } else if (last_path == .resident_view or last_path == .semantic) {
        const kind: compute.ExternalGraphKind = if (last_path == .resident_view) .resident_view else .resident_semantic;
        for (0..handle.semanticStepCount()) |node_id| {
            const step = try handle.describeSemanticStep(node_id);
            if (step.source_operation_code != null)
                try mutable.append(allocator, try makeEntry(allocator, handle, kind, node_id));
        }
    }
    var observed_runs: usize = 0;
    var observed_failed_runs: usize = 0;
    var profiler_marker: ?ProfilerMarkerAvailability = null;
    var hardware_metrics: std.ArrayList(HardwareMetricAvailability) = .empty;
    defer hardware_metrics.deinit(allocator);
    for (records) |record| switch (record.event) {
        .execution_started => |event| {
            if (event.executable_id == executable_id) observed_runs += 1;
        },
        .execution_failed => |event| {
            if (event.executable_id == executable_id) observed_failed_runs += 1;
        },
        .external_node_observed => |event| {
            if (event.executable_id != executable_id) continue;
            const index = findEntry(mutable.items, event.graph_kind, event.node_id) orelse blk: {
                try mutable.append(allocator, try makeEntry(allocator, handle, event.graph_kind, event.node_id));
                break :blk mutable.items.len - 1;
            };
            const item = &mutable.items[index];
            if (item.entry.invocation_count == 0) {
                item.entry.observed_schedule_position = event.schedule_position;
            } else if (item.entry.observed_schedule_position != event.schedule_position) {
                item.schedule_position_conflict = true;
            }
            item.entry.invocation_count += 1;
            item.entry.kernel_count = (item.entry.kernel_count orelse 0) +| event.kernel_count;
            if (event.succeeded) item.entry.succeeded_count += 1 else item.entry.failed_count += 1;
            if (event.materialized_view) {
                item.entry.materialized_view_count += 1;
                if (item.entry.materialized_view_bytes) |bytes| {
                    item.entry.materialized_view_bytes = bytes +| (item.entry.physical_output_bytes orelse 0);
                }
            }
            if (event.elapsed_ns) |elapsed| {
                try item.samples.append(allocator, elapsed);
            } else item.entry.timing_unavailable_count += 1;
        },
        .profiler_marker_availability => |event| {
            profiler_marker = .{ .provider = event.provider, .available = event.available };
        },
        .hardware_metric_availability => |event| {
            try hardware_metrics.append(allocator, .{
                .metric = @tagName(event.metric),
                .availability = @tagName(event.availability),
            });
        },
        else => {},
    };
    const output_entries = try allocator.alloc(Entry, mutable.items.len);
    defer allocator.free(output_entries);
    for (mutable.items, output_entries) |*item, *entry| {
        if (item.schedule_position_conflict) item.entry.observed_schedule_position = null;
        std.mem.sort(u64, item.samples.items, {}, std.sort.asc(u64));
        item.entry.timing_scope = timing_scope;
        if (std.mem.eql(u8, timing_scope, "kernel")) {
            item.entry.kernel_timing_sample_count = item.samples.items.len;
        } else {
            item.entry.invocation_timing_sample_count = item.samples.items.len;
        }
        for (item.samples.items) |sample| item.entry.total_ns +|= sample;
        if (item.samples.items.len != 0) {
            item.entry.minimum_ns = item.samples.items[0];
            item.entry.median_ns = percentile(item.samples.items, 1, 2);
            item.entry.p95_ns = percentile(item.samples.items, 95, 100);
            item.entry.maximum_ns = item.samples.items[item.samples.items.len - 1];
        }
        entry.* = item.entry;
    }
    const dropped = collector_dropped +| cyna_dropped;
    const kernel_count_evidence = if (dropped == 0) "exact" else "unavailable";
    const program_id = source.programId().int();
    const program_id_text = try std.fmt.allocPrint(allocator, "{d}", .{program_id});
    defer allocator.free(program_id_text);
    const executable_id_text = try std.fmt.allocPrint(allocator, "{d}", .{executable_id});
    defer allocator.free(executable_id_text);
    const Report = struct {
        schema_version: u32 = 10,
        suite: []const u8 = "compute_observability",
        program_id: u64,
        program_id_text: []const u8,
        executable_id: u64,
        executable_id_text: []const u8,
        backend: []const u8 = "cyna_cpu",
        timing_scope: []const u8,
        execution_path: []const u8,
        fallback_reason: []const u8,
        prepared_cache_hits: usize,
        prepared_cache_misses: usize,
        prepared_cache_bytes: usize,
        search: ?SearchEvidence,
        preparation: ?core.DenseGraphHandle.PreparationEvidence,
        resources: ?core.DenseGraphHandle.ResourceEvidence,
        entries: []const Entry,
        observed_run_count: usize,
        observed_failed_run_count: usize,
        dropped_event_count: usize,
        run_count_complete: bool,
        profiler_marker_availability: ?ProfilerMarkerAvailability,
        hardware_metric_availability: []const HardwareMetricAvailability,
        kernel_count_evidence: []const u8,
        phase_memory_available: bool = false,
        phase_memory: []const usize = &.{},
        cost_validations: []const usize = &.{},
    };
    return std.json.Stringify.valueAlloc(allocator, Report{
        .program_id = program_id,
        .program_id_text = program_id_text,
        .executable_id = executable_id,
        .executable_id_text = executable_id_text,
        .timing_scope = timing_scope,
        .entries = output_entries,
        .execution_path = @tagName(last_path),
        .fallback_reason = if (handle.runState()) |state| @tagName(state.fallback_reason) else |_| "unknown",
        .prepared_cache_hits = handle.cacheHits(),
        .prepared_cache_misses = handle.cacheMisses(),
        .prepared_cache_bytes = handle.cacheBytes(),
        .search = search,
        .preparation = preparation,
        .resources = resources,
        .observed_run_count = observed_runs,
        .observed_failed_run_count = observed_failed_runs,
        .dropped_event_count = dropped,
        .run_count_complete = dropped == 0,
        .kernel_count_evidence = kernel_count_evidence,
        .profiler_marker_availability = profiler_marker,
        .hardware_metric_availability = hardware_metrics.items,
    }, .{ .whitespace = .indent_2 });
}
