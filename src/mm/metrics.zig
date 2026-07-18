const std = @import("std");
const compat = @import("../support/compat.zig");
const Device = @import("../compute/types/tensor/device.zig").Device;
const Region = @import("region.zig").Region;
const obs = @import("../obs/index.zig");
const metrics = obs.metrics;

pub const Snapshot = struct {
    storage_allocations: usize,
    storage_reuses: usize,
    storage_frees: usize,
    live_storage_objects: usize,
    live_storage_bytes: usize,
    peak_storage_bytes: usize,
    live_cpu_storage_bytes: usize,
    live_metal_storage_bytes: usize,
    metal_pool_hits: usize,
    metal_pool_misses: usize,
    metal_pool_stores: usize,
    metal_pool_drops: usize,
    metal_pool_trims: usize,
    live_pooled_metal_bytes: usize,
    live_pooled_metal_buffers: usize,
    live_region_cpu_owned_bytes: usize,
    live_region_cpu_scratch_bytes: usize,
    live_region_metal_pool_bytes: usize,
    live_region_metal_scratch_bytes: usize,
    vm_phys_footprint_bytes: usize,
    vm_iokit_bytes: usize,
    vm_ioaccelerator_bytes: usize,
    qjs_malloc_bytes: usize,
    qjs_memory_used_bytes: usize,
    qjs_object_count: usize,
    qjs_string_count: usize,
};

const MetricIds = struct {
    allocations: metrics.Id,
    reuses: metrics.Id,
    frees: metrics.Id,
    live_objects: metrics.Id,
    live_bytes: metrics.Id,
    peak_bytes: metrics.Id,
    live_cpu_bytes: metrics.Id,
    live_metal_bytes: metrics.Id,
    metal_pool_hits: metrics.Id,
    metal_pool_misses: metrics.Id,
    metal_pool_stores: metrics.Id,
    metal_pool_drops: metrics.Id,
    metal_pool_trims: metrics.Id,
    metal_pool_live_bytes: metrics.Id,
    metal_pool_live_buffers: metrics.Id,
    live_region_cpu_owned_bytes: metrics.Id,
    live_region_cpu_scratch_bytes: metrics.Id,
    live_region_metal_pool_bytes: metrics.Id,
    live_region_metal_scratch_bytes: metrics.Id,
    vm_phys_footprint_bytes: metrics.Id,
    vm_iokit_bytes: metrics.Id,
    vm_ioaccelerator_bytes: metrics.Id,
    qjs_malloc_bytes: metrics.Id,
    qjs_memory_used_bytes: metrics.Id,
    qjs_object_count: metrics.Id,
    qjs_string_count: metrics.Id,
};

var metric_ids: ?MetricIds = null;
var metric_mu: compat.Mutex = .{};

pub fn noteStorageAlloc(device: Device, bytes: usize) void {
    const ids = ensureMetrics() catch return;
    metrics.add(ids.allocations, 1);
    metrics.add(ids.live_objects, 1);
    metrics.add(ids.live_bytes, toI64(bytes));
    addDeviceBytes(ids, device, bytes);
    const live_bytes = metrics.get(ids.live_bytes);
    metrics.updateMax(ids.peak_bytes, live_bytes);
}

pub fn noteStorageReuse() void {
    const ids = ensureMetrics() catch return;
    metrics.add(ids.reuses, 1);
}

pub fn noteStorageFree(device: Device, bytes: usize) void {
    const ids = ensureMetrics() catch return;
    metrics.add(ids.frees, 1);
    metrics.add(ids.live_objects, -1);
    metrics.add(ids.live_bytes, -toI64(bytes));
    subDeviceBytes(ids, device, bytes);
}

pub fn noteRegionAlloc(region: Region, bytes: usize) void {
    const ids = ensureMetrics() catch return;
    switch (region) {
        .runtime_host, .compute_host_owned, .compute_cpu_owned => metrics.add(ids.live_region_cpu_owned_bytes, toI64(bytes)),
        .runtime_host_scratch, .compute_host_scratch, .compute_cpu_scratch => metrics.add(ids.live_region_cpu_scratch_bytes, toI64(bytes)),
        .compute_metal_pool => metrics.add(ids.live_region_metal_pool_bytes, toI64(bytes)),
        .compute_metal_scratch => metrics.add(ids.live_region_metal_scratch_bytes, toI64(bytes)),
    }
}

pub fn noteRegionFree(region: Region, bytes: usize) void {
    const ids = ensureMetrics() catch return;
    switch (region) {
        .runtime_host, .compute_host_owned, .compute_cpu_owned => metrics.add(ids.live_region_cpu_owned_bytes, -toI64(bytes)),
        .runtime_host_scratch, .compute_host_scratch, .compute_cpu_scratch => metrics.add(ids.live_region_cpu_scratch_bytes, -toI64(bytes)),
        .compute_metal_pool => metrics.add(ids.live_region_metal_pool_bytes, -toI64(bytes)),
        .compute_metal_scratch => metrics.add(ids.live_region_metal_scratch_bytes, -toI64(bytes)),
    }
}

pub fn noteMetalPoolHit(byte_len: usize) void {
    const ids = ensureMetrics() catch return;
    metrics.add(ids.metal_pool_hits, 1);
    metrics.add(ids.metal_pool_live_bytes, -toI64(byte_len));
    metrics.add(ids.metal_pool_live_buffers, -1);
}

pub fn noteMetalPoolMiss() void {
    const ids = ensureMetrics() catch return;
    metrics.add(ids.metal_pool_misses, 1);
}

pub fn noteMetalPoolStore(byte_len: usize) void {
    const ids = ensureMetrics() catch return;
    metrics.add(ids.metal_pool_stores, 1);
    metrics.add(ids.metal_pool_live_bytes, toI64(byte_len));
    metrics.add(ids.metal_pool_live_buffers, 1);
}

pub fn noteMetalPoolDrop() void {
    const ids = ensureMetrics() catch return;
    metrics.add(ids.metal_pool_drops, 1);
}

pub fn noteMetalPoolTrim(freed_bytes: usize, freed_buffers: usize) void {
    const ids = ensureMetrics() catch return;
    metrics.add(ids.metal_pool_trims, 1);
    metrics.add(ids.metal_pool_live_bytes, -toI64(freed_bytes));
    metrics.add(ids.metal_pool_live_buffers, -toI64(freed_buffers));
}

pub fn setVmStats(phys_footprint: usize, iokit: usize, ioaccelerator: usize) void {
    const ids = ensureMetrics() catch return;
    metrics.set(ids.vm_phys_footprint_bytes, toI64(phys_footprint));
    metrics.set(ids.vm_iokit_bytes, toI64(iokit));
    metrics.set(ids.vm_ioaccelerator_bytes, toI64(ioaccelerator));
}

pub fn setQuickJsStats(usage: anytype) void {
    const ids = ensureMetrics() catch return;
    metrics.set(ids.qjs_malloc_bytes, toI64(usage.malloc_size));
    metrics.set(ids.qjs_memory_used_bytes, toI64(usage.memory_used_size));
    metrics.set(ids.qjs_object_count, toI64(usage.obj_count));
    metrics.set(ids.qjs_string_count, toI64(usage.str_count));
}

pub fn snapshot() Snapshot {
    const ids = ensureMetrics() catch return std.mem.zeroInit(Snapshot, .{});
    return .{
        .storage_allocations = fromI64(metrics.get(ids.allocations)),
        .storage_reuses = fromI64(metrics.get(ids.reuses)),
        .storage_frees = fromI64(metrics.get(ids.frees)),
        .live_storage_objects = fromI64(metrics.get(ids.live_objects)),
        .live_storage_bytes = fromI64(metrics.get(ids.live_bytes)),
        .peak_storage_bytes = fromI64(metrics.get(ids.peak_bytes)),
        .live_cpu_storage_bytes = fromI64(metrics.get(ids.live_cpu_bytes)),
        .live_metal_storage_bytes = fromI64(metrics.get(ids.live_metal_bytes)),
        .metal_pool_hits = fromI64(metrics.get(ids.metal_pool_hits)),
        .metal_pool_misses = fromI64(metrics.get(ids.metal_pool_misses)),
        .metal_pool_stores = fromI64(metrics.get(ids.metal_pool_stores)),
        .metal_pool_drops = fromI64(metrics.get(ids.metal_pool_drops)),
        .metal_pool_trims = fromI64(metrics.get(ids.metal_pool_trims)),
        .live_pooled_metal_bytes = fromI64(metrics.get(ids.metal_pool_live_bytes)),
        .live_pooled_metal_buffers = fromI64(metrics.get(ids.metal_pool_live_buffers)),
        .live_region_cpu_owned_bytes = fromI64(metrics.get(ids.live_region_cpu_owned_bytes)),
        .live_region_cpu_scratch_bytes = fromI64(metrics.get(ids.live_region_cpu_scratch_bytes)),
        .live_region_metal_pool_bytes = fromI64(metrics.get(ids.live_region_metal_pool_bytes)),
        .live_region_metal_scratch_bytes = fromI64(metrics.get(ids.live_region_metal_scratch_bytes)),
        .vm_phys_footprint_bytes = fromI64(metrics.get(ids.vm_phys_footprint_bytes)),
        .vm_iokit_bytes = fromI64(metrics.get(ids.vm_iokit_bytes)),
        .vm_ioaccelerator_bytes = fromI64(metrics.get(ids.vm_ioaccelerator_bytes)),
        .qjs_malloc_bytes = fromI64(metrics.get(ids.qjs_malloc_bytes)),
        .qjs_memory_used_bytes = fromI64(metrics.get(ids.qjs_memory_used_bytes)),
        .qjs_object_count = fromI64(metrics.get(ids.qjs_object_count)),
        .qjs_string_count = fromI64(metrics.get(ids.qjs_string_count)),
    };
}

fn addDeviceBytes(ids: MetricIds, device: Device, bytes: usize) void {
    switch (device) {
        .cpu => metrics.add(ids.live_cpu_bytes, toI64(bytes)),
        .metal => metrics.add(ids.live_metal_bytes, toI64(bytes)),
    }
}

fn subDeviceBytes(ids: MetricIds, device: Device, bytes: usize) void {
    switch (device) {
        .cpu => metrics.add(ids.live_cpu_bytes, -toI64(bytes)),
        .metal => metrics.add(ids.live_metal_bytes, -toI64(bytes)),
    }
}

fn toI64(value: usize) i64 {
    const capped = @min(value, @as(usize, @intCast(std.math.maxInt(i64))));
    return @intCast(capped);
}

fn fromI64(value: i64) usize {
    return if (value <= 0) 0 else @intCast(value);
}

fn ensureMetrics() !MetricIds {
    metric_mu.lock();
    defer metric_mu.unlock();

    if (metric_ids) |ids| return ids;
    const ids = MetricIds{
        .allocations = try metrics.register(.{
            .group = "allocator.events",
            .name = "allocations",
            .kind = .counter,
            .unit = .count,
            .domain = .memory,
        }),
        .reuses = try metrics.register(.{
            .group = "allocator.events",
            .name = "reuses",
            .kind = .counter,
            .unit = .count,
            .domain = .memory,
        }),
        .frees = try metrics.register(.{
            .group = "allocator.events",
            .name = "frees",
            .kind = .counter,
            .unit = .count,
            .domain = .memory,
        }),
        .live_objects = try metrics.register(.{
            .group = "owned.current_count",
            .name = "live_objects",
            .kind = .gauge,
            .unit = .count,
            .domain = .memory,
        }),
        .live_bytes = try metrics.register(.{
            .group = "owned.current_bytes",
            .name = "live_bytes",
            .kind = .gauge,
            .unit = .bytes,
            .domain = .memory,
        }),
        .peak_bytes = try metrics.register(.{
            .group = "owned.peak_bytes",
            .name = "peak_bytes",
            .kind = .gauge,
            .unit = .bytes,
            .domain = .memory,
        }),
        .live_cpu_bytes = try metrics.register(.{
            .group = "owned.current_bytes",
            .name = "live_cpu_bytes",
            .kind = .gauge,
            .unit = .bytes,
            .domain = .memory,
        }),
        .live_metal_bytes = try metrics.register(.{
            .group = "owned.current_bytes",
            .name = "live_metal_bytes",
            .kind = .gauge,
            .unit = .bytes,
            .domain = .memory,
        }),
        .metal_pool_hits = try metrics.register(.{
            .group = "allocator.events",
            .name = "hits",
            .kind = .counter,
            .unit = .count,
            .domain = .memory,
        }),
        .metal_pool_misses = try metrics.register(.{
            .group = "allocator.events",
            .name = "misses",
            .kind = .counter,
            .unit = .count,
            .domain = .memory,
        }),
        .metal_pool_stores = try metrics.register(.{
            .group = "allocator.events",
            .name = "stores",
            .kind = .counter,
            .unit = .count,
            .domain = .memory,
        }),
        .metal_pool_drops = try metrics.register(.{
            .group = "allocator.events",
            .name = "drops",
            .kind = .counter,
            .unit = .count,
            .domain = .memory,
        }),
        .metal_pool_trims = try metrics.register(.{
            .group = "allocator.events",
            .name = "trims",
            .kind = .counter,
            .unit = .count,
            .domain = .memory,
        }),
        .metal_pool_live_bytes = try metrics.register(.{
            .group = "owned.current_bytes",
            .name = "live_bytes",
            .kind = .gauge,
            .unit = .bytes,
            .domain = .memory,
        }),
        .metal_pool_live_buffers = try metrics.register(.{
            .group = "owned.current_count",
            .name = "live_buffers",
            .kind = .gauge,
            .unit = .count,
            .domain = .memory,
        }),
        .live_region_cpu_owned_bytes = try metrics.register(.{
            .group = "owned.current_bytes",
            .name = "live_cpu_owned_bytes",
            .kind = .gauge,
            .unit = .bytes,
            .domain = .memory,
        }),
        .live_region_cpu_scratch_bytes = try metrics.register(.{
            .group = "owned.current_bytes",
            .name = "live_cpu_scratch_bytes",
            .kind = .gauge,
            .unit = .bytes,
            .domain = .memory,
        }),
        .live_region_metal_pool_bytes = try metrics.register(.{
            .group = "owned.current_bytes",
            .name = "live_metal_pool_bytes",
            .kind = .gauge,
            .unit = .bytes,
            .domain = .memory,
        }),
        .live_region_metal_scratch_bytes = try metrics.register(.{
            .group = "owned.current_bytes",
            .name = "live_metal_scratch_bytes",
            .kind = .gauge,
            .unit = .bytes,
            .domain = .memory,
        }),
        .vm_phys_footprint_bytes = try metrics.register(.{
            .group = "observed.current_bytes",
            .name = "physical_footprint",
            .kind = .gauge,
            .unit = .bytes,
            .domain = .memory,
        }),
        .vm_iokit_bytes = try metrics.register(.{
            .group = "observed.current_bytes",
            .name = "iokit",
            .kind = .gauge,
            .unit = .bytes,
            .domain = .memory,
        }),
        .vm_ioaccelerator_bytes = try metrics.register(.{
            .group = "observed.current_bytes",
            .name = "ioaccelerator",
            .kind = .gauge,
            .unit = .bytes,
            .domain = .memory,
        }),
        .qjs_malloc_bytes = try metrics.register(.{
            .group = "observed.current_bytes",
            .name = "malloc_bytes",
            .kind = .gauge,
            .unit = .bytes,
            .domain = .memory,
        }),
        .qjs_memory_used_bytes = try metrics.register(.{
            .group = "observed.current_bytes",
            .name = "memory_used_bytes",
            .kind = .gauge,
            .unit = .bytes,
            .domain = .memory,
        }),
        .qjs_object_count = try metrics.register(.{
            .group = "observed.current_count",
            .name = "object_count",
            .kind = .gauge,
            .unit = .count,
            .domain = .memory,
        }),
        .qjs_string_count = try metrics.register(.{
            .group = "observed.current_count",
            .name = "string_count",
            .kind = .gauge,
            .unit = .count,
            .domain = .memory,
        }),
    };
    metric_ids = ids;
    return ids;
}

test "mm metrics use obs registry as the source of truth" {
    noteStorageAlloc(.cpu, 16);
    noteStorageReuse();
    noteStorageFree(.cpu, 16);

    const snap = snapshot();
    try std.testing.expect(snap.storage_allocations >= 1);
    try std.testing.expect(snap.storage_reuses >= 1);
    try std.testing.expect(snap.storage_frees >= 1);
    try std.testing.expectEqual(@as(usize, 0), snap.live_storage_bytes);
}
