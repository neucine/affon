const std = @import("std");
const DType = @import("../../tensor/dtype.zig").DType;
const Storage = @import("../../tensor/storage.zig").Storage;

pub fn run(dtype: DType, a: *const Storage, b: *const Storage, out: *Storage) !void {
    const out_typed = std.mem.bytesAsSlice(i64, try out.writableBytes());
    switch (dtype) {
        .f32 => {
            const a_typed = std.mem.bytesAsSlice(f32, try a.readableBytes());
            const b_typed = std.mem.bytesAsSlice(f32, try b.readableBytes());
            for (out_typed, a_typed, b_typed) |*dst, lhs, rhs| dst.* = @intFromBool(lhs > rhs);
        },
        .f64 => {
            const a_typed = std.mem.bytesAsSlice(f64, try a.readableBytes());
            const b_typed = std.mem.bytesAsSlice(f64, try b.readableBytes());
            for (out_typed, a_typed, b_typed) |*dst, lhs, rhs| dst.* = @intFromBool(lhs > rhs);
        },
        .i64 => {
            const a_typed = std.mem.bytesAsSlice(i64, try a.readableBytes());
            const b_typed = std.mem.bytesAsSlice(i64, try b.readableBytes());
            for (out_typed, a_typed, b_typed) |*dst, lhs, rhs| dst.* = @intFromBool(lhs > rhs);
        },
    }
}
