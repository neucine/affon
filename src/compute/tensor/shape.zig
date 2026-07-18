const std = @import("std");

pub const Shape = struct {
    dims: []usize,
    allocator: std.mem.Allocator,

    pub fn initCopy(allocator: std.mem.Allocator, dims: []const usize) !Shape {
        const owned = try allocator.alloc(usize, dims.len);
        @memcpy(owned, dims);
        return .{
            .dims = owned,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Shape) void {
        self.allocator.free(self.dims);
        self.* = undefined;
    }

    pub fn rank(self: Shape) usize {
        return self.dims.len;
    }

    pub fn numel(self: Shape) usize {
        var total: usize = 1;
        for (self.dims) |dim| total *= dim;
        return total;
    }

    pub fn eql(a: Shape, b: Shape) bool {
        return std.mem.eql(usize, a.dims, b.dims);
    }
};

test "shape rank and numel" {
    const allocator = std.testing.allocator;
    var shape = try Shape.initCopy(allocator, &.{ 2, 3, 4 });
    defer shape.deinit();

    try std.testing.expectEqual(@as(usize, 3), shape.rank());
    try std.testing.expectEqual(@as(usize, 24), shape.numel());
}
