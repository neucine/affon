const std = @import("std");
const shared = @import("zig_libs").memory;

pub const Region = enum {
    hao_runtime,
};

pub const Snapshot = shared.Snapshot;
pub const Check = shared.Check;

const regions = shared.Regions(Region);

pub const init = regions.init;
pub const deinit = regions.deinit;
pub const allocator = regions.allocator;
pub const snapshot = regions.snapshot;
pub const writeLeakReport = regions.writeLeakReport;

test "Hao runtime allocations are attributed to their Affon region" {
    try init(std.testing.allocator);
    defer _ = deinit();

    const value = try allocator(.hao_runtime).alloc(u8, 32);
    try std.testing.expectEqual(@as(usize, 32), snapshot(.hao_runtime).live_bytes);
    allocator(.hao_runtime).free(value);
}
