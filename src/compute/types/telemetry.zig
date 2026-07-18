const compat = @import("../../support/compat.zig");
const shared_trace = @import("zig_libs").telemetry.trace;

pub const MetricKind = enum {
    counter,
    gauge,
};

pub const MetricUnit = enum {
    count,
    bytes,
    nanoseconds,
};

pub const MetricDefinition = struct {
    group: []const u8,
    name: []const u8,
    kind: MetricKind = .counter,
    unit: MetricUnit = .count,
};

pub const TraceId = shared_trace.TraceId;
pub const SpanId = shared_trace.SpanId;

pub const TraceStatus = shared_trace.Status;

pub const SpanKind = shared_trace.Kind;

pub const AttributeValue = shared_trace.AttributeValue;
pub const Attribute = shared_trace.Attribute;
pub const TraceContext = shared_trace.Context;

pub const SpanHandle = struct {
    id: usize = 0,
    context: TraceContext = .root,
};

pub const Interface = struct {
    context: ?*anyopaque = null,
    vtable: ?*const VTable = null,

    pub const VTable = struct {
        metric_add: ?*const fn (*anyopaque, MetricDefinition, i64) void = null,
        metric_set: ?*const fn (*anyopaque, MetricDefinition, i64) void = null,
        start_span: ?*const fn (*anyopaque, TraceContext, i128, []const u8, SpanKind, []const Attribute) SpanHandle = null,
        add_event: ?*const fn (*anyopaque, usize, i128, []const u8, []const Attribute) void = null,
        end_span: ?*const fn (*anyopaque, usize, i128, TraceStatus) void = null,
    };

    pub fn add(self: Interface, definition: MetricDefinition, delta: i64) void {
        const vtable = self.vtable orelse return;
        const callback = vtable.metric_add orelse return;
        const context = self.context orelse return;
        callback(context, definition, delta);
    }

    pub fn set(self: Interface, definition: MetricDefinition, value: i64) void {
        const vtable = self.vtable orelse return;
        const callback = vtable.metric_set orelse return;
        const context = self.context orelse return;
        callback(context, definition, value);
    }

    pub fn startSpan(
        self: Interface,
        parent: TraceContext,
        timestamp_ns: i128,
        name: []const u8,
        kind: SpanKind,
        attributes: []const Attribute,
    ) Scope {
        const vtable = self.vtable orelse return .{ .telemetry = self };
        const callback = vtable.start_span orelse return .{ .telemetry = self };
        const context = self.context orelse return .{ .telemetry = self };
        const span = callback(context, parent, timestamp_ns, name, kind, attributes);
        return .{ .telemetry = self, .handle = span.id, .context = span.context };
    }
};

pub const Scope = struct {
    telemetry: Interface,
    handle: usize = 0,
    context: TraceContext = .root,

    pub fn end(self: Scope) void {
        self.endAt(compat.nanoTimestamp(), .ok);
    }

    pub fn child(
        self: Scope,
        name: []const u8,
        kind: SpanKind,
        attributes: []const Attribute,
    ) Scope {
        return self.telemetry.startSpan(self.context, compat.nanoTimestamp(), name, kind, attributes);
    }

    pub fn endAt(self: Scope, timestamp_ns: i128, status: TraceStatus) void {
        const vtable = self.telemetry.vtable orelse return;
        const callback = vtable.end_span orelse return;
        const context = self.telemetry.context orelse return;
        callback(context, self.handle, timestamp_ns, status);
    }

    pub fn addEvent(self: Scope, timestamp_ns: i128, name: []const u8, attributes: []const Attribute) void {
        const vtable = self.telemetry.vtable orelse return;
        const callback = vtable.add_event orelse return;
        const context = self.telemetry.context orelse return;
        callback(context, self.handle, timestamp_ns, name, attributes);
    }

    pub fn addEventNow(self: Scope, name: []const u8, attributes: []const Attribute) void {
        self.addEvent(compat.nanoTimestamp(), name, attributes);
    }
};
