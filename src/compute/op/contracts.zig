const std = @import("std");
const ValueSpec = @import("../tensor/value_spec.zig").ValueSpec;

pub const RankContract = union(enum) {
    any: void,
    exact: usize,
    range: struct {
        min: usize,
        max: usize,
    },
};

pub fn requireInputCount(inputs: []const ValueSpec, expected: usize) !void {
    if (inputs.len != expected) return error.InvalidInputCount;
}

pub fn requireMinInputCount(inputs: []const ValueSpec, minimum: usize) !void {
    if (inputs.len < minimum) return error.InvalidInputCount;
}

pub fn requireRank(value: ValueSpec, contract: RankContract) !void {
    const rank = value.shape.rank();
    switch (contract) {
        .any => {},
        .exact => |expected| if (rank != expected) return error.UnsupportedRank,
        .range => |range| if (rank < range.min or rank > range.max) return error.UnsupportedRank,
    }
}

pub fn requireAllRanks(inputs: []const ValueSpec, contract: RankContract) !void {
    for (inputs) |input| try requireRank(input, contract);
}

pub fn requireSameRank(inputs: []const ValueSpec) !void {
    if (inputs.len == 0) return error.InvalidInputCount;
    const rank = inputs[0].shape.rank();
    for (inputs[1..]) |input| {
        if (input.shape.rank() != rank) return error.ShapeMismatch;
    }
}

pub fn requireAxisInBounds(rank: usize, axis: usize, allow_end: bool) !void {
    if (allow_end) {
        if (axis > rank) return error.InvalidAxis;
    } else if (axis >= rank) {
        return error.InvalidAxis;
    }
}

pub fn requirePermutation(allocator: std.mem.Allocator, axes: []const usize, rank: usize) !void {
    if (axes.len != rank) return error.InvalidPermutation;
    const seen = try allocator.alloc(bool, rank);
    defer allocator.free(seen);
    @memset(seen, false);
    for (axes) |axis| {
        if (axis >= rank) return error.InvalidPermutation;
        if (seen[axis]) return error.InvalidPermutation;
        seen[axis] = true;
    }
}

test "rank contract exact and range" {
    const allocator = std.testing.allocator;
    const value = try @import("../tensor/value.zig").Value.fromSliceF32(allocator, &.{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer value.deinit();
    const spec = try value.spec();

    try requireRank(spec, .{ .exact = 2 });
    try requireRank(spec, .{ .range = .{ .min = 1, .max = 4 } });
    try std.testing.expectError(error.UnsupportedRank, requireRank(spec, .{ .exact = 3 }));
}

test "axis bounds and permutation validation" {
    try requireAxisInBounds(3, 2, false);
    try requireAxisInBounds(3, 3, true);
    try std.testing.expectError(error.InvalidAxis, requireAxisInBounds(3, 3, false));

    const allocator = std.testing.allocator;
    try requirePermutation(allocator, &.{ 2, 0, 1 }, 3);
    try std.testing.expectError(error.InvalidPermutation, requirePermutation(allocator, &.{ 0, 0, 1 }, 3));
}
