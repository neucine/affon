const std = @import("std");
const compat = @import("../../../support/compat.zig");
const Device = @import("device.zig").Device;
const mm = @import("../../memory.zig");
const telemetry = @import("../../telemetry.zig");
const metal_common = @import("../../backend/metal/common.zig");

// Runtime backing for tensor values.
//
// Storage owns allocation, device placement, refcounting, memory metrics, and
// host/device byte movement. It lives under tensor while Tensor directly embeds a
// backing pointer, but it is a runtime concern rather than a logical tensor fact.
const max_events = 128;

const EventKind = enum {
    alloc,
    retain,
    release,
    free,
    reuse,
    trim,
};

const StorageReason = enum {
    op_output,
    temporary,
    workspace,
    transfer_staging,
    constant,
    manual,
};

const StorageSource = enum {
    eager,
    graph,
    backend,
    runtime,
    facade,
    manual,
};

const Event = struct {
    kind: EventKind,
    storage_id: u64,
    device: Device,
    region: mm.Region,
    bytes: usize,
    mutable: bool,
    reason: StorageReason,
    source: StorageSource,
    ref_count: usize,
};

var events_mu: compat.Mutex = .{};
var events: [max_events]Event = undefined;
var event_count: usize = 0;
var live_bytes = std.atomic.Value(i64).init(0);
var live_objects = std.atomic.Value(i64).init(0);
var peak_bytes = std.atomic.Value(i64).init(0);
var live_cpu_bytes = std.atomic.Value(i64).init(0);
var live_metal_bytes = std.atomic.Value(i64).init(0);

fn updateMax(target: *std.atomic.Value(i64), value: i64) void {
    var current = target.load(.monotonic);
    while (value > current) {
        current = target.cmpxchgWeak(current, value, .monotonic, .monotonic) orelse return;
    }
}

fn updateStorageMetrics(device: Device, delta: i64) void {
    const live = live_bytes.fetchAdd(delta, .monotonic) + delta;
    updateMax(&peak_bytes, live);
    telemetry.set(telemetry.metrics.storage.live_bytes, live);
    telemetry.set(telemetry.metrics.storage.peak_bytes, peak_bytes.load(.monotonic));

    const device_live = switch (device) {
        .cpu => live_cpu_bytes.fetchAdd(delta, .monotonic) + delta,
        .metal => live_metal_bytes.fetchAdd(delta, .monotonic) + delta,
    };
    telemetry.set(switch (device) {
        .cpu => telemetry.metrics.storage.live_cpu_bytes,
        .metal => telemetry.metrics.storage.live_metal_bytes,
    }, device_live);
}

fn recordEvent(event: Event) void {
    events_mu.lock();
    defer events_mu.unlock();
    if (event_count < max_events) {
        events[event_count] = event;
        event_count += 1;
        return;
    }
    std.mem.copyForwards(Event, events[0 .. max_events - 1], events[1..max_events]);
    events[max_events - 1] = event;
}

fn recentEvents(buffer: []Event) []const Event {
    events_mu.lock();
    defer events_mu.unlock();
    const count = @min(buffer.len, event_count);
    const start = event_count - count;
    for (0..count) |i| buffer[i] = events[start + i];
    return buffer[0..count];
}

pub const Storage = struct {
    pub const Id = u64;
    pub const AllocationPolicy = mm.AllocationPolicy;
    pub const Reason = StorageReason;
    pub const Source = StorageSource;

    pub const Metadata = struct {
        mutable: bool = true,
        policy: AllocationPolicy = .owned,
        reason: Reason = .manual,
        source: Source = .manual,
    };

    pub const Handle = union(enum) {
        managed: mm.Block,
        borrowed_cpu: []align(8) u8,
    };

    allocator: std.mem.Allocator,
    id: Id,
    bytes: usize,
    handle: Handle,
    ref_count: std.atomic.Value(usize),
    mutable: bool,
    policy: AllocationPolicy,
    reason: Reason,
    source: Source,

    var next_id = std.atomic.Value(u64).init(1);

    pub fn createCpu(allocator: std.mem.Allocator, bytes: usize, zeroed: bool) !*Storage {
        return createCpuWithMetadata(allocator, bytes, zeroed, .{});
    }

    pub fn createCpuWithMetadata(
        allocator: std.mem.Allocator,
        bytes: usize,
        zeroed: bool,
        metadata: Metadata,
    ) !*Storage {
        const memory_region = try mm.resolveRegionFor(.cpu, metadata.policy);
        const block = try mm.allocate(memory_region, .compute_storage_metadata, bytes, .{
            .zeroed = zeroed,
            .host_allocator = allocator,
        });
        errdefer mm.release(block);
        return initOwned(allocator, bytes, .{ .managed = block }, metadata);
    }

    pub fn wrapCpuBuffer(
        allocator: std.mem.Allocator,
        buffer: []align(8) u8,
        metadata: Metadata,
    ) !*Storage {
        return initOwned(allocator, buffer.len, .{ .borrowed_cpu = buffer }, metadata);
    }

    pub fn createMetalWithMetadata(
        allocator: std.mem.Allocator,
        bytes: usize,
        metadata: Metadata,
    ) !*Storage {
        const memory_region = try mm.resolveRegionFor(.metal, metadata.policy);
        const block = try mm.allocate(memory_region, .compute_storage_metadata, bytes, .{});
        errdefer mm.release(block);
        return initOwned(allocator, bytes, .{ .managed = block }, metadata);
    }

    fn initOwned(allocator: std.mem.Allocator, bytes: usize, handle: Handle, metadata: Metadata) !*Storage {
        const self = try allocator.create(Storage);
        self.* = .{
            .allocator = allocator,
            .id = next_id.fetchAdd(1, .monotonic),
            .bytes = bytes,
            .handle = handle,
            .ref_count = std.atomic.Value(usize).init(1),
            .mutable = metadata.mutable,
            .policy = metadata.policy,
            .reason = metadata.reason,
            .source = metadata.source,
        };

        const storage_region = try self.region();
        _ = storage_region;
        telemetry.add(telemetry.metrics.storage.allocations, 1);
        telemetry.set(telemetry.metrics.storage.live_objects, live_objects.fetchAdd(1, .monotonic) + 1);
        updateStorageMetrics(self.device(), @intCast(bytes));
        emitEvent(self, .alloc, 1);
        return self;
    }

    pub fn retain(self: *Storage) void {
        const new_count = self.ref_count.fetchAdd(1, .monotonic) + 1;
        emitEvent(self, .retain, new_count);
    }

    pub fn release(self: *Storage) void {
        const previous = self.ref_count.fetchSub(1, .acq_rel);
        const new_count = previous - 1;
        emitEvent(self, .release, new_count);
        if (previous != 1) return;

        switch (self.handle) {
            .managed => |block| mm.release(block),
            .borrowed_cpu => {},
        }
        const storage_region = self.region() catch null;
        _ = storage_region;
        telemetry.add(telemetry.metrics.storage.frees, 1);
        const new_live_objects = live_objects.fetchSub(1, .monotonic) - 1;
        telemetry.set(telemetry.metrics.storage.live_objects, new_live_objects);
        updateStorageMetrics(self.device(), -@as(i64, @intCast(self.bytes)));
        emitEvent(self, .free, 0);
        self.allocator.destroy(self);
    }

    pub fn noteReuse(self: *Storage) void {
        telemetry.add(telemetry.metrics.storage.reuses, 1);
        emitEvent(self, .reuse, self.ref_count.load(.monotonic));
    }

    pub fn noteTrim(self: *Storage) void {
        emitEvent(self, .trim, self.ref_count.load(.monotonic));
    }

    pub fn device(self: *const Storage) Device {
        return switch (self.handle) {
            .borrowed_cpu => .cpu,
            .managed => |block| switch (block.storage) {
                .host => .cpu,
                .device => .metal,
            },
        };
    }

    pub fn writableBytes(self: *Storage) ![]align(8) u8 {
        if (!self.mutable) return error.StorageImmutable;
        return switch (self.handle) {
            .borrowed_cpu => |buffer| buffer,
            .managed => |block| switch (block.storage) {
                .host => |buffer| buffer,
                .device => error.MetalNotImplemented,
            },
        };
    }

    pub fn readableBytes(self: *const Storage) ![]align(8) const u8 {
        return switch (self.handle) {
            .borrowed_cpu => |buffer| buffer,
            .managed => |block| switch (block.storage) {
                .host => |buffer| buffer,
                .device => error.MetalNotImplemented,
            },
        };
    }

    pub fn writeFromHost(self: *Storage, bytes: []const u8) !void {
        if (bytes.len != self.bytes) return error.SizeMismatch;
        switch (self.handle) {
            .borrowed_cpu => |buffer| @memcpy(buffer, bytes),
            .managed => |block| switch (block.storage) {
                .host => |buffer| @memcpy(buffer, bytes),
                .device => |handle| {
                    try metal_common.writeBuffer(handle, bytes);
                },
            },
        }
    }

    pub fn copyToHost(self: *const Storage, out: []u8) !void {
        if (out.len != self.bytes) return error.SizeMismatch;
        switch (self.handle) {
            .borrowed_cpu => |buffer| @memcpy(out, buffer),
            .managed => |block| switch (block.storage) {
                .host => |buffer| @memcpy(out, buffer),
                .device => |handle| {
                    try metal_common.readBuffer(handle, out);
                },
            },
        }
    }

    pub fn isMaterialized(self: *const Storage) bool {
        return switch (self.handle) {
            .borrowed_cpu => true,
            .managed => true,
        };
    }

    pub fn metalHandle(self: *const Storage) !*anyopaque {
        return switch (self.handle) {
            .managed => |block| switch (block.storage) {
                .device => |handle| handle,
                .host => error.DeviceMismatch,
            },
            .borrowed_cpu => error.DeviceMismatch,
        };
    }

    fn region(self: *const Storage) !mm.Region {
        return switch (self.handle) {
            .managed => |block| block.region,
            .borrowed_cpu => mm.resolveRegionFor(.cpu, self.policy),
        };
    }

    fn emitEvent(self: *const Storage, kind: EventKind, ref_count: usize) void {
        const storage_region = self.region() catch |err| {
            std.debug.panic("storage region resolution failed: {s}", .{@errorName(err)});
        };
        recordEvent(.{
            .kind = kind,
            .storage_id = self.id,
            .device = self.device(),
            .region = storage_region,
            .bytes = self.bytes,
            .mutable = self.mutable,
            .reason = self.reason,
            .source = self.source,
            .ref_count = ref_count,
        });
    }
};

test "storage lifecycle emits alloc and free events" {
    const allocator = std.testing.allocator;
    const storage = try Storage.createCpuWithMetadata(allocator, 32, true, .{
        .reason = .op_output,
        .source = .eager,
    });
    const storage_id = storage.id;
    storage.release();

    var recent: [8]Event = undefined;
    const view = recentEvents(&recent);
    var saw_alloc = false;
    var saw_release = false;
    var saw_free = false;
    for (view) |event| {
        if (event.storage_id != storage_id) continue;
        switch (event.kind) {
            .alloc => saw_alloc = true,
            .release => saw_release = true,
            .free => saw_free = true,
            else => {},
        }
    }
    try std.testing.expect(saw_alloc);
    try std.testing.expect(saw_release);
    try std.testing.expect(saw_free);
}

test "borrowed cpu buffer does not free external memory" {
    var backing: [16]u8 align(8) = [_]u8{0} ** 16;
    const storage = try Storage.wrapCpuBuffer(std.testing.allocator, backing[0..], .{
        .reason = .constant,
        .source = .runtime,
        .mutable = false,
    });
    defer storage.release();

    try std.testing.expectEqual(Device.cpu, storage.device());
    try std.testing.expectError(error.StorageImmutable, storage.writableBytes());
}
