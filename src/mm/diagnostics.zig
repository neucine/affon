const mm_metrics = @import("metrics.zig");
const backend = @import("backend.zig");
const qjs = @import("../qjs.zig");

pub fn observeVm(
    phys_footprint_bytes: usize,
    iokit_bytes: usize,
    ioaccelerator_bytes: usize,
    libc_malloc_in_use_bytes: usize,
    libc_malloc_allocated_bytes: usize,
) void {
    mm_metrics.setVmStats(phys_footprint_bytes, iokit_bytes, ioaccelerator_bytes);
    _ = libc_malloc_in_use_bytes;
    _ = libc_malloc_allocated_bytes;
}

pub fn observeQuickJs(snapshot: qjs.MemoryUsage) void {
    mm_metrics.setQuickJsStats(snapshot);
}

pub fn trimMetalPool() usize {
    return backend.trimMetalPool();
}
