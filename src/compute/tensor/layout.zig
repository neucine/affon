const std = @import("std");
const Shape = @import("shape.zig").Shape;

pub const Layout = struct {
    strides: []isize,
    offset: usize = 0,
    allocator: std.mem.Allocator,

    pub fn initCopy(allocator: std.mem.Allocator, strides_in: []const isize, offset: usize) !Layout {
        const strides = try allocator.alloc(isize, strides_in.len);
        @memcpy(strides, strides_in);
        return .{
            .strides = strides,
            .offset = offset,
            .allocator = allocator,
        };
    }

    pub fn initContiguous(allocator: std.mem.Allocator, shape: Shape) !Layout {
        const strides = try allocator.alloc(isize, shape.rank());
        if (shape.rank() > 0) {
            strides[shape.rank() - 1] = 1;
            if (shape.rank() > 1) {
                var i = shape.rank() - 1;
                while (i > 0) {
                    i -= 1;
                    strides[i] = strides[i + 1] * @as(isize, @intCast(shape.dims[i + 1]));
                }
            }
        }
        return .{
            .strides = strides,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Layout) void {
        self.allocator.free(self.strides);
        self.* = undefined;
    }

    pub fn isContiguous(self: Layout, shape: Shape) bool {
        var expected: isize = 1;
        var i = shape.rank();
        while (i > 0) {
            i -= 1;
            if (self.strides[i] != expected) return false;
            expected *= @intCast(shape.dims[i]);
        }
        return true;
    }
};

test "contiguous layout for rank-2 shape" {
    const allocator = std.testing.allocator;
    var shape = try Shape.initCopy(allocator, &.{ 2, 3 });
    defer shape.deinit();
    var layout = try Layout.initContiguous(allocator, shape);
    defer layout.deinit();

    try std.testing.expectEqual(@as(isize, 3), layout.strides[0]);
    try std.testing.expectEqual(@as(isize, 1), layout.strides[1]);
    try std.testing.expect(layout.isContiguous(shape));
}

test "layout copy preserves strides and offset" {
    const allocator = std.testing.allocator;
    var layout = try Layout.initCopy(allocator, &.{ 6, 2, 1 }, 3);
    defer layout.deinit();

    try std.testing.expectEqual(@as(isize, 6), layout.strides[0]);
    try std.testing.expectEqual(@as(usize, 3), layout.offset);
}
