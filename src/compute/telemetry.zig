const std = @import("std");
const shared = @import("zig_libs").telemetry;

pub const MetricKind = shared.MetricKind;
pub const MetricUnit = shared.MetricUnit;
pub const Group = enum {
    execution,
    storage,
    memory,
};
pub const MetricDefinition = shared.MetricDefinition;
pub const Backend = shared.Backend;
pub const Scope = shared.Scope;
pub const TraceContext = shared.TraceContext;
pub const TraceId = shared.TraceId;
pub const SpanId = shared.SpanId;
pub const SpanHandle = shared.SpanHandle;
pub const TraceStatus = shared.TraceStatus;
pub const SpanKind = shared.SpanKind;
pub const Attribute = shared.Attribute;
pub const AttributeValue = shared.AttributeValue;

pub const traces = struct {
    pub const run = "run";
    pub const region = "region";
    pub const fallback = "fallback";
};

pub fn groupName(group: Group) []const u8 {
    return switch (group) {
        .execution => "compute.execution",
        .storage => "compute.storage",
        .memory => "compute.memory",
    };
}

fn counter(group: Group, name: []const u8) MetricDefinition {
    return .{ .group = groupName(group), .name = name, .kind = .counter, .unit = .count };
}

fn bytes(group: Group, name: []const u8) MetricDefinition {
    return .{ .group = groupName(group), .name = name, .kind = .counter, .unit = .bytes };
}

fn gauge(group: Group, name: []const u8, unit: MetricUnit) MetricDefinition {
    return .{ .group = groupName(group), .name = name, .kind = .gauge, .unit = unit };
}

pub const metrics = struct {
    pub const execution = struct {
        pub const transfer_to_host_count = counter(.execution, "transfer_to_host_count");
        pub const transfer_to_host_bytes = bytes(.execution, "transfer_to_host_bytes");
        pub const transfer_from_host_count = counter(.execution, "transfer_from_host_count");
        pub const transfer_from_host_bytes = bytes(.execution, "transfer_from_host_bytes");
        pub const contiguity_fixup_count = counter(.execution, "contiguity_fixup_count");
        pub const contiguity_fixup_bytes = bytes(.execution, "contiguity_fixup_bytes");
        pub const contiguous_input_required_count = counter(.execution, "contiguous_input_required_count");
        pub const storage_input_required_count = counter(.execution, "storage_input_required_count");
        pub const fusion_eligible_region_count = counter(.execution, "fusion_eligible_region_count");
        pub const fusion_fallback_count = counter(.execution, "fusion_fallback_count");
    };

    pub const storage = struct {
        pub const allocation_count = counter(.storage, "allocation_count");
        pub const free_count = counter(.storage, "free_count");
        pub const reuse_count = counter(.storage, "reuse_count");
        pub const live_object_count = gauge(.storage, "live_object_count", .count);
        pub const live_bytes = gauge(.storage, "live_bytes", .bytes);
        pub const peak_bytes = gauge(.storage, "peak_bytes", .bytes);
        pub const live_cpu_bytes = gauge(.storage, "live_cpu_bytes", .bytes);
        pub const live_metal_bytes = gauge(.storage, "live_metal_bytes", .bytes);
        pub const peak_cpu_bytes = gauge(.storage, "peak_cpu_bytes", .bytes);
        pub const peak_metal_bytes = gauge(.storage, "peak_metal_bytes", .bytes);
    };

    pub const memory = struct {
        pub const pool_hit_count = counter(.memory, "metal_pool_hit_count");
        pub const pool_miss_count = counter(.memory, "metal_pool_miss_count");
        pub const pool_store_count = counter(.memory, "metal_pool_store_count");
        pub const pool_drop_count = counter(.memory, "metal_pool_drop_count");
        pub const pool_trim_count = counter(.memory, "metal_pool_trim_count");
        pub const pool_live_bytes = gauge(.memory, "metal_pool_live_bytes", .bytes);
        pub const pool_live_buffer_count = gauge(.memory, "metal_pool_live_buffer_count", .count);
        pub const pool_peak_bytes = gauge(.memory, "metal_pool_peak_bytes", .bytes);
        pub const pool_trim_bytes = counter(.memory, "metal_pool_trim_bytes");
        pub const region_host_owned_bytes = gauge(.memory, "region_host_owned_bytes", .bytes);
        pub const region_host_scratch_bytes = gauge(.memory, "region_host_scratch_bytes", .bytes);
        pub const region_cpu_owned_bytes = gauge(.memory, "region_cpu_owned_bytes", .bytes);
        pub const region_cpu_scratch_bytes = gauge(.memory, "region_cpu_scratch_bytes", .bytes);
        pub const region_metal_pool_bytes = gauge(.memory, "region_metal_pool_bytes", .bytes);
        pub const region_metal_scratch_bytes = gauge(.memory, "region_metal_scratch_bytes", .bytes);
    };
};

pub fn install(next: Backend) void {
    shared.install(next);
}

pub fn current() Backend {
    return shared.current();
}

pub fn add(definition: MetricDefinition, delta: i64) void {
    shared.add(definition, delta);
}

pub fn set(definition: MetricDefinition, value: i64) void {
    shared.set(definition, value);
}

pub fn addCounter(group: Group, name: []const u8, delta: i64) void {
    add(counter(group, name), delta);
}

pub fn addBytes(group: Group, name: []const u8, delta: i64) void {
    add(bytes(group, name), delta);
}

pub fn beginTrace(group: Group, name: []const u8) Scope {
    var full_name: [128]u8 = undefined;
    const qualified_name = std.fmt.bufPrint(&full_name, "{s}/{s}", .{ groupName(group), name }) catch name;
    return startSpan(.root, qualified_name, .internal, &.{});
}

pub fn startSpan(
    parent: TraceContext,
    name: []const u8,
    kind: SpanKind,
    attributes: []const Attribute,
) Scope {
    return shared.startSpan(parent, name, kind, attributes);
}

test "telemetry forwards counter, gauge, and span operations" {
    const Collector = struct {
        adds: usize = 0,
        sets: usize = 0,
        begins: usize = 0,
        events: usize = 0,
        ends: usize = 0,
        last_definition: ?MetricDefinition = null,
        last_value: i64 = 0,

        fn metricAdd(context: *anyopaque, definition: MetricDefinition, delta: i64) void {
            const self: *@This() = @ptrCast(@alignCast(context));
            self.adds += 1;
            self.last_definition = definition;
            self.last_value = delta;
        }

        fn metricSet(context: *anyopaque, definition: MetricDefinition, value: i64) void {
            const self: *@This() = @ptrCast(@alignCast(context));
            self.sets += 1;
            self.last_definition = definition;
            self.last_value = value;
        }

        fn startSpan(
            context: *anyopaque,
            parent: TraceContext,
            timestamp_ns: i128,
            name: []const u8,
            kind: SpanKind,
            attributes: []const Attribute,
        ) SpanHandle {
            const self: *@This() = @ptrCast(@alignCast(context));
            self.begins += 1;
            _ = parent;
            _ = timestamp_ns;
            _ = kind;
            _ = attributes;
            std.debug.assert(std.mem.eql(u8, "compute.execution/run", name));
            return .{ .id = 7, .context = .root };
        }

        fn endSpan(context: *anyopaque, handle: usize, timestamp_ns: i128, status: TraceStatus) void {
            const self: *@This() = @ptrCast(@alignCast(context));
            self.ends += 1;
            _ = timestamp_ns;
            _ = status;
            std.debug.assert(handle == 7);
        }

        fn addEvent(
            context: *anyopaque,
            handle: usize,
            timestamp_ns: i128,
            name: []const u8,
            attributes: []const Attribute,
        ) void {
            const self: *@This() = @ptrCast(@alignCast(context));
            self.events += 1;
            _ = timestamp_ns;
            _ = attributes;
            std.debug.assert(handle == 7);
            std.debug.assert(std.mem.eql(u8, "fusion_hit", name));
        }
    };

    var collector = Collector{};
    const vtable = Backend.VTable{
        .metric_add = Collector.metricAdd,
        .metric_set = Collector.metricSet,
        .start_span = Collector.startSpan,
        .add_event = Collector.addEvent,
        .end_span = Collector.endSpan,
    };
    install(.{ .context = &collector, .vtable = &vtable });
    defer install(.{});

    add(metrics.execution.fusion_fallback_count, 2);
    try std.testing.expectEqual(@as(usize, 1), collector.adds);
    try std.testing.expectEqual(@as(i64, 2), collector.last_value);
    try std.testing.expectEqualStrings("compute.execution", collector.last_definition.?.group);

    set(metrics.storage.live_bytes, 1024);
    try std.testing.expectEqual(@as(usize, 1), collector.sets);
    try std.testing.expectEqual(@as(i64, 1024), collector.last_value);
    try std.testing.expectEqual(MetricKind.gauge, collector.last_definition.?.kind);

    var scope = beginTrace(.execution, traces.run);
    scope.addEventNow("fusion_hit", &.{});
    scope.end();
    try std.testing.expectEqual(@as(usize, 1), collector.begins);
    try std.testing.expectEqual(@as(usize, 1), collector.events);
    try std.testing.expectEqual(@as(usize, 1), collector.ends);
}
