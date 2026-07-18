const std = @import("std");
const compat = @import("../support/compat.zig");
const common = @import("common.zig");

pub const Mechanism = common.Mechanism;
pub const TelemetryChannel = common.TelemetryChannel;
pub const Domain = common.Domain;

pub const Policy = struct {
    enabled: bool = true,
    min_elapsed_ns: u64 = 0,
    sample_rate_per_mille: u16 = 1000,
};

pub const Event = struct {
    cursor: u64 = 0,
    name: []const u8,
    mechanism: Mechanism,
    channel: TelemetryChannel,
    domain: Domain,
    group: []const u8,
    ended_at_unix_ms: i64,
    elapsed_ns: u64,
};

const max_name_len = 96;
const max_group_len = 64;

const StoredEvent = struct {
    cursor: u64 = 0,
    mechanism: Mechanism,
    channel: TelemetryChannel,
    domain: Domain,
    ended_at_unix_ms: i64,
    elapsed_ns: u64,
    name_len: usize,
    group_len: usize,
    name_buf: [max_name_len]u8,
    group_buf: [max_group_len]u8,
};

pub const Scope = struct {
    name_len: usize,
    name_buf: [max_name_len]u8,
    mechanism: Mechanism,
    channel: TelemetryChannel,
    domain: Domain,
    group_len: usize,
    group_buf: [max_group_len]u8,
    start_ns: i128,

    pub fn end(self: Scope) void {
        const elapsed_ns = elapsedSinceNs(self.start_ns);
        record(.{
            .name = self.name_buf[0..self.name_len],
            .mechanism = self.mechanism,
            .channel = self.channel,
            .domain = self.domain,
            .group = self.group_buf[0..self.group_len],
            .ended_at_unix_ms = compat.milliTimestamp(),
            .elapsed_ns = elapsed_ns,
        });
    }
};

pub const max_events = 128;
var events_mu: compat.Mutex = .{};
var events: [max_events]StoredEvent = undefined;
var event_head: usize = 0;
var event_count: usize = 0;
var next_cursor: u64 = 0;
var policy_mu: compat.Mutex = .{};
var policy: Policy = .{};

pub const RecentView = struct {
    events: []const Event,
    next_cursor: u64,
    dropped_before_cursor: bool,
};

pub fn begin(domain: Domain, group: []const u8, name: []const u8) Scope {
    validateTaxonomy(group, name);
    var scope = Scope{
        .name_len = @min(name.len, max_name_len),
        .name_buf = std.mem.zeroes([max_name_len]u8),
        .mechanism = .telemetry,
        .channel = .traces,
        .domain = domain,
        .group_len = @min(group.len, max_group_len),
        .group_buf = std.mem.zeroes([max_group_len]u8),
        .start_ns = compat.nanoTimestamp(),
    };
    @memcpy(scope.name_buf[0..scope.name_len], name[0..scope.name_len]);
    @memcpy(scope.group_buf[0..scope.group_len], group[0..scope.group_len]);
    return scope;
}

fn elapsedSinceNs(start_ns: i128) u64 {
    const elapsed = compat.nanoTimestamp() - start_ns;
    if (elapsed <= 0) return 0;
    const max = std.math.maxInt(u64);
    if (elapsed > max) return max;
    return @intCast(elapsed);
}

fn validateTaxonomy(group: []const u8, name: []const u8) void {
    std.debug.assert(group.len > 0);
    std.debug.assert(name.len > 0);
    std.debug.assert(std.mem.indexOfScalar(u8, name, '.') == null);
}

pub fn recent(buffer: []Event) []const Event {
    events_mu.lock();
    defer events_mu.unlock();

    const count = @min(buffer.len, event_count);
    for (0..count) |i| {
        const logical_index = event_count - count + i;
        const stored = &events[(event_head + logical_index) % max_events];
        buffer[i] = toPublicEvent(stored);
    }
    return buffer[0..count];
}

pub fn recentSince(buffer: []Event, since: u64) RecentView {
    events_mu.lock();
    defer events_mu.unlock();

    const current_next_cursor = next_cursor;
    const earliest_cursor = current_next_cursor - event_count;
    const dropped_before_cursor = since < earliest_cursor;
    const effective_since = if (dropped_before_cursor) earliest_cursor else since;

    var out_count: usize = 0;
    for (0..event_count) |i| {
        const stored = &events[(event_head + i) % max_events];
        if (stored.cursor < effective_since) continue;
        if (out_count >= buffer.len) break;
        buffer[out_count] = toPublicEvent(stored);
        out_count += 1;
    }

    return .{
        .events = buffer[0..out_count],
        .next_cursor = current_next_cursor,
        .dropped_before_cursor = dropped_before_cursor,
    };
}

pub fn flushAndDeinit() void {}

pub fn currentPolicy() Policy {
    policy_mu.lock();
    defer policy_mu.unlock();
    return policy;
}

pub fn setPolicy(next: Policy) void {
    policy_mu.lock();
    defer policy_mu.unlock();
    policy = next;
}

pub fn updatePolicyEnabled(enabled: bool) void {
    policy_mu.lock();
    defer policy_mu.unlock();
    policy.enabled = enabled;
}

pub fn updatePolicyMinElapsedNs(min_elapsed_ns: u64) void {
    policy_mu.lock();
    defer policy_mu.unlock();
    policy.min_elapsed_ns = min_elapsed_ns;
}

pub fn updatePolicySampleRate(sample_rate: f64) void {
    const clamped = if (sample_rate < 0) 0 else if (sample_rate > 1) 1 else sample_rate;
    const per_mille: u16 = @intFromFloat(@round(clamped * 1000.0));
    policy_mu.lock();
    defer policy_mu.unlock();
    policy.sample_rate_per_mille = per_mille;
}

fn record(event: Event) void {
    const active_policy = currentPolicy();
    if (!active_policy.enabled) return;
    if (event.elapsed_ns < active_policy.min_elapsed_ns) return;

    events_mu.lock();
    defer events_mu.unlock();

    if (active_policy.sample_rate_per_mille < 1000) {
        if ((next_cursor % 1000) >= active_policy.sample_rate_per_mille) return;
    }

    var stored = fromPublicEvent(event);
    stored.cursor = next_cursor;
    next_cursor += 1;

    if (event_count < max_events) {
        events[(event_head + event_count) % max_events] = stored;
        event_count += 1;
        return;
    }

    events[event_head] = stored;
    event_head = (event_head + 1) % max_events;
}

fn toPublicEvent(stored: *const StoredEvent) Event {
    return .{
        .cursor = stored.cursor,
        .name = stored.name_buf[0..stored.name_len],
        .mechanism = stored.mechanism,
        .channel = stored.channel,
        .domain = stored.domain,
        .group = stored.group_buf[0..stored.group_len],
        .ended_at_unix_ms = stored.ended_at_unix_ms,
        .elapsed_ns = stored.elapsed_ns,
    };
}

fn fromPublicEvent(event: Event) StoredEvent {
    var stored = StoredEvent{
        .mechanism = event.mechanism,
        .channel = event.channel,
        .domain = event.domain,
        .ended_at_unix_ms = event.ended_at_unix_ms,
        .elapsed_ns = event.elapsed_ns,
        .name_len = @min(event.name.len, max_name_len),
        .group_len = @min(event.group.len, max_group_len),
        .name_buf = std.mem.zeroes([max_name_len]u8),
        .group_buf = std.mem.zeroes([max_group_len]u8),
    };
    @memcpy(stored.name_buf[0..stored.name_len], event.name[0..stored.name_len]);
    @memcpy(stored.group_buf[0..stored.group_len], event.group[0..stored.group_len]);
    return stored;
}

test "trace recent exposes completed events" {
    var scope = begin(.compute, "execution", "dispatch");
    scope.end();

    var buffer: [4]Event = undefined;
    const view = recent(&buffer);
    try std.testing.expect(view.len >= 1);
}

test "context-aware trace scope records domain and group" {
    var scope = begin(.memory, "storage", "alloc");
    scope.end();

    var buffer: [8]Event = undefined;
    const view = recent(&buffer);
    try std.testing.expect(view.len >= 1);
    const event = view[view.len - 1];
    try std.testing.expectEqual(Domain.memory, event.domain);
    try std.testing.expectEqual(TelemetryChannel.traces, event.channel);
    try std.testing.expect(std.mem.eql(u8, "storage", event.group));
    try std.testing.expect(event.ended_at_unix_ms > 0);
}

test "trace recentSince uses cursor and reports drop" {
    var baseline_buffer: [1]Event = undefined;
    const baseline = recentSince(&baseline_buffer, std.math.maxInt(u64)).next_cursor;

    var first = begin(.compute, "observer", "first");
    first.end();
    var second = begin(.compute, "observer", "second");
    second.end();

    var buffer: [8]Event = undefined;
    const all = recentSince(&buffer, baseline);
    try std.testing.expect(all.events.len >= 2);
    try std.testing.expectEqual(false, all.dropped_before_cursor);

    const since_last = all.events[all.events.len - 1].cursor + 1;
    const none = recentSince(&buffer, since_last);
    try std.testing.expectEqual(@as(usize, 0), none.events.len);
    try std.testing.expectEqual(false, none.dropped_before_cursor);
}

test "trace policy can suppress recording by threshold" {
    const original = currentPolicy();
    defer setPolicy(original);

    setPolicy(.{ .enabled = true, .min_elapsed_ns = std.math.maxInt(u64), .sample_rate_per_mille = 1000 });

    var baseline_buffer: [1]Event = undefined;
    const baseline = recentSince(&baseline_buffer, std.math.maxInt(u64)).next_cursor;
    var scope = begin(.compute, "test", "suppressed");
    scope.end();
    var buffer: [4]Event = undefined;
    const view = recentSince(&buffer, baseline);
    try std.testing.expectEqual(@as(usize, 0), view.events.len);
}
