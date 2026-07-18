const backend = @import("backend.zig");
const qjs = @import("../qjs.zig");
const metrics = @import("../obs/metrics.zig");

const MetricIds = struct {
    phys_footprint: metrics.Id,
    iokit: metrics.Id,
    ioaccelerator: metrics.Id,
    malloc_bytes: metrics.Id,
    memory_used_bytes: metrics.Id,
    object_count: metrics.Id,
    string_count: metrics.Id,
};

var metric_ids: ?MetricIds = null;

fn ensureMetrics() ?MetricIds {
    if (metric_ids) |ids| return ids;
    const ids = MetricIds{
        .phys_footprint = metrics.register(.{ .group = "runtime.memory", .name = "physical_footprint", .kind = .gauge, .unit = .bytes, .domain = .memory }) catch return null,
        .iokit = metrics.register(.{ .group = "runtime.memory", .name = "iokit", .kind = .gauge, .unit = .bytes, .domain = .memory }) catch return null,
        .ioaccelerator = metrics.register(.{ .group = "runtime.memory", .name = "ioaccelerator", .kind = .gauge, .unit = .bytes, .domain = .memory }) catch return null,
        .malloc_bytes = metrics.register(.{ .group = "runtime.memory", .name = "malloc_bytes", .kind = .gauge, .unit = .bytes, .domain = .memory }) catch return null,
        .memory_used_bytes = metrics.register(.{ .group = "runtime.memory", .name = "memory_used_bytes", .kind = .gauge, .unit = .bytes, .domain = .memory }) catch return null,
        .object_count = metrics.register(.{ .group = "runtime.memory", .name = "object_count", .kind = .gauge, .unit = .count, .domain = .memory }) catch return null,
        .string_count = metrics.register(.{ .group = "runtime.memory", .name = "string_count", .kind = .gauge, .unit = .count, .domain = .memory }) catch return null,
    };
    metric_ids = ids;
    return ids;
}

pub fn observeVm(
    phys_footprint_bytes: usize,
    iokit_bytes: usize,
    ioaccelerator_bytes: usize,
    libc_malloc_in_use_bytes: usize,
    libc_malloc_allocated_bytes: usize,
) void {
    const ids = ensureMetrics() orelse return;
    metrics.set(ids.phys_footprint, @intCast(phys_footprint_bytes));
    metrics.set(ids.iokit, @intCast(iokit_bytes));
    metrics.set(ids.ioaccelerator, @intCast(ioaccelerator_bytes));
    _ = libc_malloc_in_use_bytes;
    _ = libc_malloc_allocated_bytes;
}

pub fn observeQuickJs(snapshot: qjs.MemoryUsage) void {
    const ids = ensureMetrics() orelse return;
    metrics.set(ids.malloc_bytes, @intCast(snapshot.malloc_size));
    metrics.set(ids.memory_used_bytes, @intCast(snapshot.memory_used_size));
    metrics.set(ids.object_count, @intCast(snapshot.obj_count));
    metrics.set(ids.string_count, @intCast(snapshot.str_count));
}

pub fn trimMetalPool() usize {
    return backend.trimMetalPool();
}
