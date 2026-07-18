const std = @import("std");
const DType = @import("../../tensor/dtype.zig").DType;
const Storage = @import("../../tensor/storage.zig").Storage;

pub fn run(allocator: std.mem.Allocator, dtype: DType, input: *const Storage, values_out: *Storage, indices_out: *Storage, shape: []const usize, axis: usize, k: usize, largest: bool) !void {
    switch (dtype) {
        .f32 => try runTyped(allocator, f32, input, values_out, indices_out, shape, axis, k, largest),
        .f64 => try runTyped(allocator, f64, input, values_out, indices_out, shape, axis, k, largest),
        .i64 => try runTyped(allocator, i64, input, values_out, indices_out, shape, axis, k, largest),
    }
}

fn runTyped(allocator: std.mem.Allocator, comptime T: type, input: *const Storage, values_out: *Storage, indices_out: *Storage, shape: []const usize, axis: usize, k: usize, largest: bool) !void {
    const Candidate = struct { v: T, i: usize };
    const src = std.mem.bytesAsSlice(T, try input.readableBytes());
    const values = std.mem.bytesAsSlice(T, try values_out.writableBytes());
    const indices = std.mem.bytesAsSlice(i64, try indices_out.writableBytes());

    var outer: usize = 1;
    var inner: usize = 1;
    const axis_len = shape[axis];
    for (shape[0..axis]) |d| outer *= d;
    for (shape[axis + 1 ..]) |d| inner *= d;

    var candidates = try allocator.alloc(Candidate, axis_len);
    defer allocator.free(candidates);

    for (0..outer) |o| {
        for (0..inner) |inn| {
            for (0..axis_len) |a| {
                candidates[a] = .{ .v = src[o * axis_len * inner + a * inner + inn], .i = a };
            }
            std.mem.sort(Candidate, candidates, largest, struct {
                fn lessThan(largest_local: bool, lhs: Candidate, rhs: Candidate) bool {
                    return if (largest_local) lhs.v > rhs.v else lhs.v < rhs.v;
                }
            }.lessThan);

            for (0..k) |kk| {
                const out_index = o * k * inner + kk * inner + inn;
                values[out_index] = candidates[kk].v;
                indices[out_index] = @intCast(candidates[kk].i);
            }
        }
    }
}
