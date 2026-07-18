const std = @import("std");
const compat = @import("../support/compat.zig");
const common = @import("common.zig");

pub const MetricKind = common.MetricKind;
pub const MetricUnit = common.MetricUnit;
pub const Mechanism = common.Mechanism;
pub const TelemetryChannel = common.TelemetryChannel;
pub const Domain = common.Domain;

pub const max_metrics = 128;

pub const Definition = struct {
    domain: Domain = .compute,
    group: []const u8,
    name: []const u8,
    kind: MetricKind,
    unit: MetricUnit,
    mechanism: Mechanism = .telemetry,
    channel: TelemetryChannel = .metrics,
};

pub const Snapshot = struct {
    domain: Domain,
    group: []const u8,
    name: []const u8,
    kind: MetricKind,
    unit: MetricUnit,
    mechanism: Mechanism,
    channel: TelemetryChannel,
    value: i64,
};

pub const Id = u16;

const Registry = struct {
    mutex: compat.Mutex = .{},
    count: usize = 0,
    definitions: [max_metrics]Definition = undefined,
    values: [max_metrics]std.atomic.Value(i64) = initAtomicValues(),
};

var registry = Registry{};

pub const RuntimeObject = enum {
    tensor,
    ndarray,
    graph_node,
};

const RuntimeObjectIds = struct {
    live_tensor: Id,
    live_ndarray: Id,
    live_graph_node: Id,
    finalized_tensor: Id,
    finalized_ndarray: Id,
};

var runtime_object_mu: compat.Mutex = .{};
var runtime_object_ids: ?RuntimeObjectIds = null;

fn initAtomicValues() [max_metrics]std.atomic.Value(i64) {
    var values: [max_metrics]std.atomic.Value(i64) = undefined;
    for (&values) |*value| value.* = std.atomic.Value(i64).init(0);
    return values;
}

pub fn register(definition: Definition) !Id {
    validateTaxonomy(definition.group, definition.name);
    registry.mutex.lock();
    defer registry.mutex.unlock();

    var i: usize = 0;
    while (i < registry.count) : (i += 1) {
        const existing = registry.definitions[i];
        if (existing.domain == definition.domain and
            std.mem.eql(u8, existing.group, definition.group) and
            std.mem.eql(u8, existing.name, definition.name) and
            existing.kind == definition.kind and
            existing.unit == definition.unit)
        {
            return @intCast(i);
        }
    }

    if (registry.count >= max_metrics) return error.TooManyMetrics;
    const id: Id = @intCast(registry.count);
    registry.definitions[registry.count] = definition;
    registry.values[registry.count].store(0, .monotonic);
    registry.count += 1;
    return id;
}

fn validateTaxonomy(group: []const u8, name: []const u8) void {
    std.debug.assert(group.len > 0);
    std.debug.assert(name.len > 0);
    std.debug.assert(std.mem.indexOfScalar(u8, name, '.') == null);
}

pub fn set(id: Id, value: i64) void {
    registry.values[id].store(value, .monotonic);
}

pub fn get(id: Id) i64 {
    return registry.values[id].load(.monotonic);
}

pub fn add(id: Id, delta: i64) void {
    _ = registry.values[id].fetchAdd(delta, .monotonic);
}

pub fn updateMax(id: Id, candidate: i64) void {
    var current = registry.values[id].load(.monotonic);
    while (candidate > current) {
        const previous = registry.values[id].cmpxchgWeak(current, candidate, .monotonic, .monotonic);
        if (previous == null) return;
        current = previous.?;
    }
}

pub fn snapshot(buffer: []Snapshot) []const Snapshot {
    registry.mutex.lock();
    defer registry.mutex.unlock();

    const count = @min(buffer.len, registry.count);
    for (buffer[0..count], 0..) |*entry, i| {
        const def = registry.definitions[i];
        entry.* = .{
            .domain = def.domain,
            .group = def.group,
            .name = def.name,
            .kind = def.kind,
            .unit = def.unit,
            .mechanism = def.mechanism,
            .channel = def.channel,
            .value = registry.values[i].load(.monotonic),
        };
    }
    return buffer[0..count];
}

pub fn ensureRuntimeObjectMetrics() void {
    _ = runtimeObjectIds() catch {};
}

pub fn recordRuntimeObjectCreate(kind: RuntimeObject) void {
    const ids = runtimeObjectIds() catch return;
    switch (kind) {
        .tensor => add(ids.live_tensor, 1),
        .ndarray => add(ids.live_ndarray, 1),
        .graph_node => add(ids.live_graph_node, 1),
    }
}

pub fn recordRuntimeObjectFinalize(kind: RuntimeObject) void {
    const ids = runtimeObjectIds() catch return;
    switch (kind) {
        .tensor => {
            add(ids.live_tensor, -1);
            add(ids.finalized_tensor, 1);
        },
        .ndarray => {
            add(ids.live_ndarray, -1);
            add(ids.finalized_ndarray, 1);
        },
        .graph_node => add(ids.live_graph_node, -1),
    }
}

fn runtimeObjectIds() !RuntimeObjectIds {
    runtime_object_mu.lock();
    defer runtime_object_mu.unlock();

    if (runtime_object_ids) |ids| return ids;

    const ids = RuntimeObjectIds{
        .live_tensor = try register(.{
            .domain = .runtime,
            .group = "objects.live",
            .name = "tensor",
            .kind = .gauge,
            .unit = .count,
        }),
        .live_ndarray = try register(.{
            .domain = .runtime,
            .group = "objects.live",
            .name = "ndarray",
            .kind = .gauge,
            .unit = .count,
        }),
        .live_graph_node = try register(.{
            .domain = .runtime,
            .group = "objects.live",
            .name = "graph_node",
            .kind = .gauge,
            .unit = .count,
        }),
        .finalized_tensor = try register(.{
            .domain = .runtime,
            .group = "objects.finalized",
            .name = "tensor",
            .kind = .counter,
            .unit = .count,
        }),
        .finalized_ndarray = try register(.{
            .domain = .runtime,
            .group = "objects.finalized",
            .name = "ndarray",
            .kind = .counter,
            .unit = .count,
        }),
    };
    runtime_object_ids = ids;
    return ids;
}

test "metrics register and snapshot" {
    const id = try register(.{
        .domain = .memory,
        .group = "allocator.events",
        .name = "allocations",
        .kind = .counter,
        .unit = .count,
    });
    add(id, 2);

    var buffer: [max_metrics]Snapshot = undefined;
    const view = snapshot(&buffer);
    try std.testing.expect(view.len >= 1);
    var found = false;
    for (view) |entry| {
        if (entry.domain != .memory) continue;
        if (!std.mem.eql(u8, entry.group, "allocator.events")) continue;
        if (!std.mem.eql(u8, entry.name, "allocations")) continue;
        found = true;
        try std.testing.expectEqual(TelemetryChannel.metrics, entry.channel);
        try std.testing.expectEqual(Domain.memory, entry.domain);
    }
    try std.testing.expect(found);
}

test "runtime object metrics are registered and mutable" {
    ensureRuntimeObjectMetrics();
    recordRuntimeObjectCreate(.tensor);
    recordRuntimeObjectFinalize(.tensor);

    var buffer: [max_metrics]Snapshot = undefined;
    const view = snapshot(&buffer);

    var found_live_tensor = false;
    var found_finalized_tensor = false;
    for (view) |entry| {
        if (entry.domain == .runtime and std.mem.eql(u8, entry.group, "objects.live") and std.mem.eql(u8, entry.name, "tensor")) {
            found_live_tensor = true;
            try std.testing.expectEqual(@as(i64, 0), entry.value);
        }
        if (entry.domain == .runtime and std.mem.eql(u8, entry.group, "objects.finalized") and std.mem.eql(u8, entry.name, "tensor")) {
            found_finalized_tensor = true;
            try std.testing.expect(entry.value >= 1);
        }
    }
    try std.testing.expect(found_live_tensor);
    try std.testing.expect(found_finalized_tensor);
}
