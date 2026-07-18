const std = @import("std");
const compat = @import("../support/compat.zig");
const shared = @import("zig_libs").telemetry.interface;

pub const MetricKind = shared.MetricKind;
pub const MetricUnit = shared.MetricUnit;
pub const Group = enum {
    execution,
    storage,
    memory,
};
pub const MetricDefinition = shared.MetricDefinition;
pub const Interface = shared.Interface;
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
        .execution => "execution",
        .storage => "storage",
        .memory => "memory",
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
        pub const input_required_count = counter(.execution, "input_required_count");
        pub const storage_required_count = counter(.execution, "storage_required_count");
        pub const fusion_groups_eligible = counter(.execution, "fusion_groups_eligible");
        pub const fusion_fallback = counter(.execution, "fusion_fallback");
    };

    pub const storage = struct {
        pub const allocations = counter(.storage, "allocations");
        pub const frees = counter(.storage, "frees");
        pub const reuses = counter(.storage, "reuses");
        pub const live_objects = gauge(.storage, "live_objects", .count);
        pub const live_bytes = gauge(.storage, "live_bytes", .bytes);
        pub const peak_bytes = gauge(.storage, "peak_bytes", .bytes);
        pub const live_cpu_bytes = gauge(.storage, "live_cpu_bytes", .bytes);
        pub const live_metal_bytes = gauge(.storage, "live_metal_bytes", .bytes);
    };

    pub const memory = struct {
        pub const pool_hits = counter(.memory, "metal_pool_hits");
        pub const pool_misses = counter(.memory, "metal_pool_misses");
        pub const pool_stores = counter(.memory, "metal_pool_stores");
        pub const pool_drops = counter(.memory, "metal_pool_drops");
        pub const pool_trims = counter(.memory, "metal_pool_trims");
        pub const pool_live_bytes = gauge(.memory, "metal_pool_live_bytes", .bytes);
        pub const pool_live_buffers = gauge(.memory, "metal_pool_live_buffers", .count);
        pub const pool_trim_bytes = counter(.memory, "metal_pool_trim_bytes");
        pub const region_host_owned_bytes = gauge(.memory, "region_compute_host_owned_bytes", .bytes);
        pub const region_host_scratch_bytes = gauge(.memory, "region_compute_host_scratch_bytes", .bytes);
        pub const region_cpu_owned_bytes = gauge(.memory, "region_compute_cpu_owned_bytes", .bytes);
        pub const region_cpu_scratch_bytes = gauge(.memory, "region_compute_cpu_scratch_bytes", .bytes);
        pub const region_metal_pool_bytes = gauge(.memory, "region_compute_metal_pool_bytes", .bytes);
        pub const region_metal_scratch_bytes = gauge(.memory, "region_compute_metal_scratch_bytes", .bytes);
    };
};

var active: Interface = .{};

pub fn init(next: Interface) void {
    active = next;
}

pub fn current() Interface {
    return active;
}

pub fn add(definition: MetricDefinition, delta: i64) void {
    active.add(definition, delta);
}

pub fn set(definition: MetricDefinition, value: i64) void {
    active.set(definition, value);
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
    return active.startSpan(parent, compat.nanoTimestamp(), name, kind, attributes);
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
            std.debug.assert(std.mem.eql(u8, "execution/run", name));
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
    const vtable = Interface.VTable{
        .metric_add = Collector.metricAdd,
        .metric_set = Collector.metricSet,
        .start_span = Collector.startSpan,
        .add_event = Collector.addEvent,
        .end_span = Collector.endSpan,
    };
    init(.{ .context = &collector, .vtable = &vtable });
    defer init(.{});

    add(metrics.execution.fusion_fallback, 2);
    try std.testing.expectEqual(@as(usize, 1), collector.adds);
    try std.testing.expectEqual(@as(i64, 2), collector.last_value);
    try std.testing.expectEqualStrings("execution", collector.last_definition.?.group);

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
