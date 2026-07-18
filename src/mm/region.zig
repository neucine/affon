pub const Region = enum {
    runtime_host,
    runtime_host_scratch,
    compute_host_owned,
    compute_host_scratch,
    compute_cpu_owned,
    compute_cpu_scratch,
    compute_metal_pool,
    compute_metal_scratch,
};

pub const Metadata = struct {
    domain: []const u8,
    component: []const u8,
    name: []const u8,
};

pub fn metadata(region: Region) Metadata {
    return switch (region) {
        .runtime_host => .{ .domain = "runtime", .component = "mm", .name = "runtime_host" },
        .runtime_host_scratch => .{ .domain = "runtime", .component = "mm", .name = "runtime_host_scratch" },
        .compute_host_owned => .{ .domain = "compute", .component = "mm", .name = "compute_host_owned" },
        .compute_host_scratch => .{ .domain = "compute", .component = "mm", .name = "compute_host_scratch" },
        .compute_cpu_owned => .{ .domain = "compute", .component = "mm", .name = "compute_cpu_owned" },
        .compute_cpu_scratch => .{ .domain = "compute", .component = "mm", .name = "compute_cpu_scratch" },
        .compute_metal_pool => .{ .domain = "compute", .component = "mm", .name = "compute_metal_pool" },
        .compute_metal_scratch => .{ .domain = "compute", .component = "mm", .name = "compute_metal_scratch" },
    };
}
