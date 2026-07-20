const std = @import("std");
const SliceRange = @import("../types/operation/options.zig").SliceRange;

pub const SliceSelector = union(enum) {
    all,
    index: i64,
    range: RangeSelector,
};

pub const RangeSelector = struct {
    start: ?i64 = null,
    stop: ?i64 = null,
    step: isize = 1,
};

fn normalizeIndex(index: i64, dim: usize, allow_end: bool) !usize {
    const dim_i64: i64 = @intCast(dim);
    const normalized = if (index < 0) dim_i64 + index else index;
    if (normalized < 0) return error.InvalidSliceBound;
    if (normalized > dim_i64) return error.InvalidSliceBound;
    if (!allow_end and normalized == dim_i64) return error.InvalidSliceBound;
    return @intCast(normalized);
}

fn normalizeOptionalBound(value: ?i64, fallback: usize, dim: usize) !usize {
    return if (value) |index| try normalizeIndex(index, dim, true) else fallback;
}

pub fn normalizeSelectors(allocator: std.mem.Allocator, input_shape: []const usize, selectors: []const SliceSelector) ![]SliceRange {
    if (selectors.len > input_shape.len) return error.TooManySliceDimensions;
    const ranges = try allocator.alloc(SliceRange, selectors.len);
    errdefer allocator.free(ranges);
    for (selectors, 0..) |selector, i| {
        const dim = input_shape[i];
        ranges[i] = switch (selector) {
            .all => .{ .start = 0, .stop = dim, .step = 1 },
            .index => |index| blk: {
                const normalized = try normalizeIndex(index, dim, false);
                break :blk .{ .start = normalized, .stop = normalized + 1, .step = 1 };
            },
            .range => |range| blk: {
                if (range.step <= 0) return error.NegativeSliceStepNotYetSupported;
                const start = try normalizeOptionalBound(range.start, 0, dim);
                const stop = try normalizeOptionalBound(range.stop, dim, dim);
                if (start > stop) return error.InvalidSliceBound;
                break :blk .{ .start = start, .stop = stop, .step = range.step };
            },
        };
    }
    return ranges;
}

test "slice selectors normalize negative indexes" {
    const allocator = std.testing.allocator;
    const ranges = try normalizeSelectors(allocator, &.{ 4, 8 }, &.{
        .{ .index = -1 },
        .{ .range = .{ .start = -4, .stop = -1, .step = 2 } },
    });
    defer allocator.free(ranges);
    try std.testing.expectEqual(@as(usize, 3), ranges[0].start);
    try std.testing.expectEqual(@as(usize, 4), ranges[0].stop);
    try std.testing.expectEqual(@as(usize, 4), ranges[1].start);
    try std.testing.expectEqual(@as(usize, 7), ranges[1].stop);
    try std.testing.expectEqual(@as(isize, 2), ranges[1].step);
}
