const std = @import("std");
const compat = @import("../../support/compat.zig");
const Device = @import("../types/tensor/device.zig").Device;
const telemetry = @import("../telemetry.zig");
const policy_mod = @import("policy.zig");
const region_mod = @import("region.zig");

pub const AllocationPolicy = policy_mod.AllocationPolicy;
pub const Region = region_mod.Region;

extern fn affon_metal_buffer_create(byte_len: usize) ?*anyopaque;
extern fn affon_metal_buffer_destroy(handle: ?*anyopaque) void;
extern fn affon_metal_is_available() bool;
extern fn affon_metal_last_error() [*:0]const u8;

const max_per_size = 8;
const max_total_bytes = 1024 * 1024 * 1024;
var pool_oversize_threshold_bytes: usize = 64 * 1024 * 1024;

var mu: compat.Mutex = .{};
var pool = std.AutoHashMapUnmanaged(usize, std.ArrayListUnmanaged(*anyopaque)){};
var pooled_bytes = std.atomic.Value(usize).init(0);
var pooled_buffers = std.atomic.Value(usize).init(0);
var peak_pooled_bytes = std.atomic.Value(usize).init(0);

fn publishPoolGauges() void {
    telemetry.set(telemetry.metrics.memory.pool_live_bytes, @intCast(pooled_bytes.load(.monotonic)));
    telemetry.set(telemetry.metrics.memory.pool_live_buffer_count, @intCast(pooled_buffers.load(.monotonic)));
    telemetry.set(telemetry.metrics.memory.pool_peak_bytes, @intCast(peak_pooled_bytes.load(.monotonic)));
}

fn updatePoolPeak(value: usize) void {
    var current = peak_pooled_bytes.load(.monotonic);
    while (value > current) {
        current = peak_pooled_bytes.cmpxchgWeak(current, value, .monotonic, .monotonic) orelse return;
    }
}

pub fn resolveRegion(device: Device, policy: AllocationPolicy) !Region {
    return switch (device) {
        .cpu => switch (policy) {
            .owned => .compute_cpu_owned,
            .scratch => .compute_cpu_scratch,
            .pooled => error.CpuPolicyNotSupported,
        },
        .metal => switch (policy) {
            .pooled => .compute_metal_pool,
            .scratch => .compute_metal_scratch,
            .owned => error.MetalRequiresExplicitPolicy,
        },
    };
}

pub fn allocCpu(
    allocator: std.mem.Allocator,
    policy: AllocationPolicy,
    bytes: usize,
    zeroed: bool,
) ![]align(8) u8 {
    return switch (policy) {
        .owned => allocCpuOwned(allocator, bytes, zeroed),
        .scratch => allocCpuScratch(allocator, bytes, zeroed),
        .pooled => error.CpuPolicyNotSupported,
    };
}

pub fn freeCpu(allocator: std.mem.Allocator, policy: AllocationPolicy, buffer: []align(8) u8) void {
    switch (policy) {
        .owned => freeCpuOwned(allocator, buffer),
        .scratch => freeCpuScratch(allocator, buffer),
        .pooled => {},
    }
}

pub fn allocMetal(policy: AllocationPolicy, bytes: usize) !*anyopaque {
    return switch (policy) {
        .pooled => createPooledHandle(bytes) orelse error.OutOfMemory,
        .scratch => allocMetalScratch(bytes) orelse error.OutOfMemory,
        .owned => error.MetalRequiresExplicitPolicy,
    };
}

pub fn freeMetal(policy: AllocationPolicy, bytes: usize, handle: *anyopaque) void {
    switch (policy) {
        .pooled => releasePooledHandle(bytes, handle),
        .scratch => freeMetalScratch(handle),
        .owned => {},
    }
}

pub fn supports(device: Device, policy: AllocationPolicy) bool {
    return switch (device) {
        .cpu => policy != .pooled,
        .metal => policy != .owned,
    };
}

pub fn setPoolOversizeThreshold(bytes: usize) void {
    pool_oversize_threshold_bytes = bytes;
}

pub fn trimMetalPool() usize {
    var freed: usize = 0;
    var freed_count: usize = 0;
    mu.lock();
    defer mu.unlock();

    var it = pool.iterator();
    while (it.next()) |entry| {
        const byte_len = entry.key_ptr.*;
        const handles = &entry.value_ptr.*;
        for (handles.items) |handle| {
            affon_metal_buffer_destroy(handle);
            freed += byte_len;
            freed_count += 1;
        }
        handles.deinit(std.heap.page_allocator);
    }
    pool.clearAndFree(std.heap.page_allocator);

    if (freed > 0) {
        _ = pooled_bytes.fetchSub(freed, .monotonic);
        _ = pooled_buffers.fetchSub(freed_count, .monotonic);
        publishPoolGauges();
        telemetry.add(telemetry.metrics.memory.pool_trim_count, @intCast(freed_count));
        telemetry.add(telemetry.metrics.memory.pool_trim_bytes, @intCast(freed));
    }
    return freed;
}

pub fn isMetalAvailable() bool {
    return affon_metal_is_available();
}

pub fn metalLastError() []const u8 {
    return std.mem.span(affon_metal_last_error());
}

fn allocCpuOwned(allocator: std.mem.Allocator, bytes: usize, zeroed: bool) ![]align(8) u8 {
    const buffer = try allocator.alignedAlloc(u8, .@"8", bytes);
    if (zeroed) @memset(buffer, 0);
    return buffer;
}

fn freeCpuOwned(allocator: std.mem.Allocator, buffer: []align(8) u8) void {
    allocator.free(buffer);
}

fn allocCpuScratch(allocator: std.mem.Allocator, bytes: usize, zeroed: bool) ![]align(8) u8 {
    const buffer = try allocator.alignedAlloc(u8, .@"8", bytes);
    if (zeroed) @memset(buffer, 0);
    return buffer;
}

fn freeCpuScratch(allocator: std.mem.Allocator, buffer: []align(8) u8) void {
    allocator.free(buffer);
}

fn createPooledHandle(byte_len: usize) ?*anyopaque {
    if (!shouldPool(byte_len)) return directAlloc(byte_len);
    if (takePooled(byte_len)) |handle| return handle;
    telemetry.add(telemetry.metrics.memory.pool_miss_count, 1);

    const handle = affon_metal_buffer_create(byte_len) orelse blk: {
        _ = trimMetalPool();
        break :blk affon_metal_buffer_create(byte_len);
    };
    return handle;
}

fn releasePooledHandle(byte_len: usize, handle: *anyopaque) void {
    if (!shouldPool(byte_len)) {
        telemetry.add(telemetry.metrics.memory.pool_drop_count, 1);
        affon_metal_buffer_destroy(handle);
        return;
    }
    if (!returnPooled(byte_len, handle)) {
        telemetry.add(telemetry.metrics.memory.pool_drop_count, 1);
        affon_metal_buffer_destroy(handle);
    }
}

fn allocMetalScratch(byte_len: usize) ?*anyopaque {
    return affon_metal_buffer_create(byte_len);
}

fn freeMetalScratch(handle: *anyopaque) void {
    affon_metal_buffer_destroy(handle);
}

fn shouldPool(byte_len: usize) bool {
    return byte_len < pool_oversize_threshold_bytes;
}

fn directAlloc(byte_len: usize) ?*anyopaque {
    const handle = affon_metal_buffer_create(byte_len) orelse blk: {
        _ = trimMetalPool();
        break :blk affon_metal_buffer_create(byte_len);
    };
    return handle;
}

fn takePooled(byte_len: usize) ?*anyopaque {
    mu.lock();
    defer mu.unlock();

    const bucket = pool.getPtr(byte_len) orelse return null;
    const handle = bucket.pop() orelse return null;
    _ = pooled_bytes.fetchSub(byte_len, .monotonic);
    _ = pooled_buffers.fetchSub(1, .monotonic);
    telemetry.add(telemetry.metrics.memory.pool_hit_count, 1);
    publishPoolGauges();
    return handle;
}

fn returnPooled(byte_len: usize, handle: *anyopaque) bool {
    mu.lock();
    defer mu.unlock();

    if (pooled_bytes.load(.monotonic) + byte_len > max_total_bytes) return false;

    const gop = pool.getOrPut(std.heap.page_allocator, byte_len) catch return false;
    if (!gop.found_existing) gop.value_ptr.* = .empty;
    if (gop.value_ptr.items.len >= max_per_size) return false;
    gop.value_ptr.append(std.heap.page_allocator, handle) catch return false;

    _ = pooled_bytes.fetchAdd(byte_len, .monotonic);
    _ = pooled_buffers.fetchAdd(1, .monotonic);
    updatePoolPeak(pooled_bytes.load(.monotonic));
    telemetry.add(telemetry.metrics.memory.pool_store_count, 1);
    publishPoolGauges();
    return true;
}
