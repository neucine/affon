const std = @import("std");
const DType = @import("../../types/tensor/dtype.zig").DType;
const Storage = @import("../../types/tensor/storage.zig").Storage;

pub fn run(dtype: DType, input: *const Storage, out: *Storage, min_val: f64, max_val: f64) !void {
    switch (dtype) {
        .f32 => {
            const minv: f32 = @floatCast(min_val);
            const maxv: f32 = @floatCast(max_val);
            const src = std.mem.bytesAsSlice(f32, try input.readableBytes());
            const dst = std.mem.bytesAsSlice(f32, try out.writableBytes());
            for (dst, src) |*d, s| d.* = @min(@max(s, minv), maxv);
        },
        .f64 => {
            const src = std.mem.bytesAsSlice(f64, try input.readableBytes());
            const dst = std.mem.bytesAsSlice(f64, try out.writableBytes());
            for (dst, src) |*d, s| d.* = @min(@max(s, min_val), max_val);
        },
        .i64 => {
            const min_trunc = @trunc(min_val);
            const max_trunc = @trunc(max_val);
            if (min_trunc != min_val or max_trunc != max_val) return error.InvalidClampBounds;
            if (min_trunc < @as(f64, @floatFromInt(std.math.minInt(i64))) or min_trunc > @as(f64, @floatFromInt(std.math.maxInt(i64)))) return error.InvalidClampBounds;
            if (max_trunc < @as(f64, @floatFromInt(std.math.minInt(i64))) or max_trunc > @as(f64, @floatFromInt(std.math.maxInt(i64)))) return error.InvalidClampBounds;

            const minv: i64 = @intFromFloat(min_trunc);
            const maxv: i64 = @intFromFloat(max_trunc);

            const src = std.mem.bytesAsSlice(i64, try input.readableBytes());
            const dst = std.mem.bytesAsSlice(i64, try out.writableBytes());
            for (dst, src) |*d, s| d.* = @min(@max(s, minv), maxv);
        },
    }
}
