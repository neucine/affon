const std = @import("std");
const compat = @import("../support/compat.zig");
const Device = @import("types/tensor/device.zig").Device;
const policy = @import("memory/policy.zig");
const region = @import("memory/region.zig");
const intention_mod = @import("memory/intention.zig");
const backend = @import("memory/backend.zig");
const telemetry = @import("telemetry.zig");

pub const AllocationPolicy = policy.AllocationPolicy;
pub const Region = region.Region;
pub const RegionMetadata = region.Metadata;
pub const Intention = intention_mod.Intention;

pub const setPoolOversizeThreshold = backend.setPoolOversizeThreshold;
pub const Block = struct {
    region: Region,
    policy: AllocationPolicy,
    bytes: usize,
    storage: Storage,
    host_allocator: ?std.mem.Allocator = null,

    pub const Storage = union(enum) {
        host: []align(8) u8,
        device: *anyopaque,
    };
};

pub const AllocOptions = struct {
    zeroed: bool = false,
    host_allocator: ?std.mem.Allocator = null,
};

pub const BoundaryAllocator = struct {
    intention: Intention,
    backing: std.mem.Allocator = std.heap.c_allocator,

    pub fn allocator(self: *BoundaryAllocator) std.mem.Allocator {
        return .{
            .ptr = self,
            .vtable = &.{
                .alloc = alloc,
                .resize = resize,
                .remap = remap,
                .free = free,
            },
        };
    }

    fn alloc(ctx: *anyopaque, len: usize, alignment: std.mem.Alignment, ret_addr: usize) ?[*]u8 {
        const self: *BoundaryAllocator = @ptrCast(@alignCast(ctx));
        return self.backing.rawAlloc(len, alignment, ret_addr) orelse return null;
    }

    fn resize(ctx: *anyopaque, memory_slice: []u8, alignment: std.mem.Alignment, new_len: usize, ret_addr: usize) bool {
        const self: *BoundaryAllocator = @ptrCast(@alignCast(ctx));
        return self.backing.rawResize(memory_slice, alignment, new_len, ret_addr);
    }

    fn remap(ctx: *anyopaque, memory_slice: []u8, alignment: std.mem.Alignment, new_len: usize, ret_addr: usize) ?[*]u8 {
        const self: *BoundaryAllocator = @ptrCast(@alignCast(ctx));
        return self.backing.rawRemap(memory_slice, alignment, new_len, ret_addr) orelse return null;
    }

    fn free(ctx: *anyopaque, memory_slice: []u8, alignment: std.mem.Alignment, ret_addr: usize) void {
        const self: *BoundaryAllocator = @ptrCast(@alignCast(ctx));
        self.backing.rawFree(memory_slice, alignment, ret_addr);
    }
};

pub const RegionConfig = struct {
    device: Device,
    policy: AllocationPolicy,
    ready: bool = true,
};

const RegionState = struct {
    registered: bool = false,
    config: RegionConfig = .{
        .device = .cpu,
        .policy = .owned,
        .ready = false,
    },
};

var region_mu: compat.Mutex = .{};
var region_state = std.EnumArray(Region, RegionState).initFill(.{});
var boundary_allocators_ready = false;
var boundary_allocators = std.EnumArray(Region, std.EnumArray(Intention, BoundaryAllocator)).initUndefined();
var default_regions_ready = false;
var region_host_owned_bytes = std.atomic.Value(i64).init(0);
var region_host_scratch_bytes = std.atomic.Value(i64).init(0);
var region_cpu_owned_bytes = std.atomic.Value(i64).init(0);
var region_cpu_scratch_bytes = std.atomic.Value(i64).init(0);
var region_metal_pool_bytes = std.atomic.Value(i64).init(0);
var region_metal_scratch_bytes = std.atomic.Value(i64).init(0);

fn regionCounter(memory_region: Region) *std.atomic.Value(i64) {
    return switch (memory_region) {
        .compute_host_owned => &region_host_owned_bytes,
        .compute_host_scratch => &region_host_scratch_bytes,
        .compute_cpu_owned => &region_cpu_owned_bytes,
        .compute_cpu_scratch => &region_cpu_scratch_bytes,
        .compute_metal_pool => &region_metal_pool_bytes,
        .compute_metal_scratch => &region_metal_scratch_bytes,
    };
}

fn regionMetric(memory_region: Region) telemetry.MetricDefinition {
    return switch (memory_region) {
        .compute_host_owned => telemetry.metrics.memory.region_host_owned_bytes,
        .compute_host_scratch => telemetry.metrics.memory.region_host_scratch_bytes,
        .compute_cpu_owned => telemetry.metrics.memory.region_cpu_owned_bytes,
        .compute_cpu_scratch => telemetry.metrics.memory.region_cpu_scratch_bytes,
        .compute_metal_pool => telemetry.metrics.memory.region_metal_pool_bytes,
        .compute_metal_scratch => telemetry.metrics.memory.region_metal_scratch_bytes,
    };
}

fn updateRegionBytes(memory_region: Region, delta: i64) void {
    const value = regionCounter(memory_region).fetchAdd(delta, .monotonic) + delta;
    telemetry.set(regionMetric(memory_region), value);
}

pub fn registerRegion(memory_region: Region, config: RegionConfig) void {
    region_mu.lock();
    defer region_mu.unlock();
    region_state.set(memory_region, .{
        .registered = true,
        .config = config,
    });
}

pub fn setRegionReady(memory_region: Region, ready: bool) !void {
    region_mu.lock();
    defer region_mu.unlock();
    var state = region_state.get(memory_region);
    if (!state.registered) return error.RegionNotRegistered;
    state.config.ready = ready;
    region_state.set(memory_region, state);
}

pub fn getRegionConfig(memory_region: Region) !RegionConfig {
    region_mu.lock();
    defer region_mu.unlock();
    const state = region_state.get(memory_region);
    if (!state.registered) return error.RegionNotRegistered;
    return state.config;
}

pub fn isRegionReady(memory_region: Region) bool {
    region_mu.lock();
    defer region_mu.unlock();
    const state = region_state.get(memory_region);
    return state.registered and state.config.ready;
}

pub fn allocator(memory_region: Region, intention: Intention) !std.mem.Allocator {
    region_mu.lock();
    defer region_mu.unlock();
    ensureDefaultRegions();

    const state = region_state.get(memory_region);
    if (!state.registered) return error.RegionNotRegistered;
    if (!state.config.ready) return error.RegionNotReady;

    ensureBoundaryAllocators();
    return boundary_allocators.getPtr(memory_region).getPtr(intention).allocator();
}

pub fn regionMetadata(memory_region: Region) RegionMetadata {
    return region.metadata(memory_region);
}

pub fn allocate(memory_region: Region, intention: Intention, bytes: usize, opts: AllocOptions) !Block {
    region_mu.lock();
    ensureDefaultRegions();
    region_mu.unlock();

    const config = try getRegionConfig(memory_region);
    if (!config.ready) return error.RegionNotReady;

    var block: Block = undefined;
    switch (config.device) {
        .cpu => {
            const alloc = opts.host_allocator orelse try allocator(memory_region, intention);
            const buffer = try backend.allocCpu(alloc, config.policy, bytes, opts.zeroed);
            block = .{
                .region = memory_region,
                .policy = config.policy,
                .bytes = bytes,
                .storage = .{ .host = buffer },
                .host_allocator = alloc,
            };
        },
        .metal => {
            const handle = try backend.allocMetal(config.policy, bytes);
            block = .{
                .region = memory_region,
                .policy = config.policy,
                .bytes = bytes,
                .storage = .{ .device = handle },
                .host_allocator = null,
            };
        },
    }
    updateRegionBytes(block.region, @intCast(block.bytes));
    return block;
}

pub fn release(block: Block) void {
    updateRegionBytes(block.region, -@as(i64, @intCast(block.bytes)));
    switch (block.storage) {
        .host => |buffer| {
            if (block.host_allocator) |alloc| {
                backend.freeCpu(alloc, block.policy, buffer);
            }
        },
        .device => |handle| backend.freeMetal(block.policy, block.bytes, handle),
    }
}

pub fn resolveRegionFor(device: Device, alloc_policy: AllocationPolicy) !Region {
    return backend.resolveRegion(device, alloc_policy);
}

pub fn allocCpu(
    alloc: std.mem.Allocator,
    alloc_policy: AllocationPolicy,
    bytes: usize,
    zeroed: bool,
) ![]align(8) u8 {
    return backend.allocCpu(alloc, alloc_policy, bytes, zeroed);
}

pub fn freeCpu(alloc: std.mem.Allocator, alloc_policy: AllocationPolicy, buffer: []align(8) u8) void {
    backend.freeCpu(alloc, alloc_policy, buffer);
}

pub fn allocMetal(alloc_policy: AllocationPolicy, bytes: usize) !*anyopaque {
    return backend.allocMetal(alloc_policy, bytes);
}

pub fn freeMetal(alloc_policy: AllocationPolicy, bytes: usize, handle: *anyopaque) void {
    backend.freeMetal(alloc_policy, bytes, handle);
}

fn ensureBoundaryAllocators() void {
    if (boundary_allocators_ready) return;
    inline for (@typeInfo(Region).@"enum".fields) |region_field| {
        const memory_region: Region = @enumFromInt(region_field.value);
        var by_intention = std.EnumArray(Intention, BoundaryAllocator).initUndefined();
        inline for (@typeInfo(Intention).@"enum".fields) |intention_field| {
            const intention: Intention = @enumFromInt(intention_field.value);
            by_intention.set(intention, .{
                .intention = intention,
            });
        }
        boundary_allocators.set(memory_region, by_intention);
    }
    boundary_allocators_ready = true;
}

fn ensureDefaultRegions() void {
    if (default_regions_ready) return;
    region_state.set(.compute_host_owned, .{ .registered = true, .config = .{ .device = .cpu, .policy = .owned, .ready = true } });
    region_state.set(.compute_host_scratch, .{ .registered = true, .config = .{ .device = .cpu, .policy = .scratch, .ready = true } });
    region_state.set(.compute_cpu_owned, .{ .registered = true, .config = .{ .device = .cpu, .policy = .owned, .ready = true } });
    region_state.set(.compute_cpu_scratch, .{ .registered = true, .config = .{ .device = .cpu, .policy = .scratch, .ready = true } });
    region_state.set(.compute_metal_pool, .{ .registered = true, .config = .{ .device = .metal, .policy = .pooled, .ready = true } });
    region_state.set(.compute_metal_scratch, .{ .registered = true, .config = .{ .device = .metal, .policy = .scratch, .ready = true } });
    default_regions_ready = true;
}

test {
    _ = @import("memory/backend.zig");
    _ = @import("memory/policy.zig");
    _ = @import("memory/region.zig");
}
