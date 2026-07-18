pub const common = @import("common.zig");
pub const metrics = @import("metrics.zig");
pub const trace = @import("trace.zig");

test {
    _ = @import("common.zig");
    _ = @import("metrics.zig");
    _ = @import("trace.zig");
}

test "obs contract: taxonomy enum names stay stable" {
    const std = @import("std");

    try std.testing.expectEqualStrings("gauge", @tagName(common.MetricKind.gauge));
    try std.testing.expectEqualStrings("counter", @tagName(common.MetricKind.counter));

    try std.testing.expectEqualStrings("count", @tagName(common.MetricUnit.count));
    try std.testing.expectEqualStrings("bytes", @tagName(common.MetricUnit.bytes));
    try std.testing.expectEqualStrings("nanoseconds", @tagName(common.MetricUnit.nanoseconds));

    try std.testing.expectEqualStrings("diagnostic", @tagName(common.Mechanism.diagnostic));
    try std.testing.expectEqualStrings("telemetry", @tagName(common.Mechanism.telemetry));

    try std.testing.expectEqualStrings("traces", @tagName(common.TelemetryChannel.traces));
    try std.testing.expectEqualStrings("logs", @tagName(common.TelemetryChannel.logs));
    try std.testing.expectEqualStrings("metrics", @tagName(common.TelemetryChannel.metrics));

    try std.testing.expectEqualStrings("runtime", @tagName(common.Domain.runtime));
    try std.testing.expectEqualStrings("compute", @tagName(common.Domain.compute));
    try std.testing.expectEqualStrings("memory", @tagName(common.Domain.memory));
    try std.testing.expectEqualStrings("support", @tagName(common.Domain.support));
}

test "obs contract: metrics defaults and idempotent register" {
    const std = @import("std");

    const def: metrics.Definition = .{
        .domain = .compute,
        .group = "contract.obs",
        .name = "counter",
        .kind = .counter,
        .unit = .count,
    };
    const id1 = try metrics.register(def);
    const id2 = try metrics.register(def);
    try std.testing.expectEqual(id1, id2);

    metrics.set(id1, 5);
    metrics.add(id1, 2);

    var buf: [metrics.max_metrics]metrics.Snapshot = undefined;
    const snaps = metrics.snapshot(&buf);
    var found = false;
    for (snaps) |snap| {
        if (snap.domain != .compute) continue;
        if (!std.mem.eql(u8, snap.group, "contract.obs")) continue;
        if (!std.mem.eql(u8, snap.name, "counter")) continue;
        found = true;
        try std.testing.expectEqual(common.Mechanism.telemetry, snap.mechanism);
        try std.testing.expectEqual(common.TelemetryChannel.metrics, snap.channel);
        try std.testing.expectEqual(common.Domain.compute, snap.domain);
        try std.testing.expectEqual(@as(i64, 7), snap.value);
    }
    try std.testing.expect(found);
}

test "obs contract: trace begin uses domain+group+name" {
    const std = @import("std");

    var s1 = trace.begin(.compute, "allocator", "default");
    s1.end();
    var s2 = trace.begin(.memory, "allocator", "contextual");
    s2.end();

    var events: [16]trace.Event = undefined;
    const recent = trace.recent(&events);
    try std.testing.expect(recent.len >= 2);

    const last = recent[recent.len - 1];
    try std.testing.expectEqual(common.Domain.memory, last.domain);
    try std.testing.expectEqual(common.TelemetryChannel.traces, last.channel);
    try std.testing.expect(std.mem.eql(u8, last.group, "allocator"));
}
