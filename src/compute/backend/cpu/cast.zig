const std = @import("std");
const DType = @import("../../types/tensor/dtype.zig").DType;
const Storage = @import("../../types/tensor/storage.zig").Storage;

pub fn run(from: DType, to: DType, input: *const Storage, out: *Storage) !void {
    if (from == to) {
        const src = try input.readableBytes();
        const dst = try out.writableBytes();
        @memcpy(dst, src);
        return;
    }
    switch (from) {
        .f32 => {
            const src = std.mem.bytesAsSlice(f32, try input.readableBytes());
            switch (to) {
                .f64 => {
                    const dst = std.mem.bytesAsSlice(f64, try out.writableBytes());
                    for (dst, src) |*d, s| d.* = @as(f64, s);
                },
                .i64 => {
                    const dst = std.mem.bytesAsSlice(i64, try out.writableBytes());
                    for (dst, src) |*d, s| d.* = @intFromFloat(@trunc(s));
                },
                else => unreachable,
            }
        },
        .f64 => {
            const src = std.mem.bytesAsSlice(f64, try input.readableBytes());
            switch (to) {
                .f32 => {
                    const dst = std.mem.bytesAsSlice(f32, try out.writableBytes());
                    for (dst, src) |*d, s| d.* = @floatCast(s);
                },
                .i64 => {
                    const dst = std.mem.bytesAsSlice(i64, try out.writableBytes());
                    for (dst, src) |*d, s| d.* = @intFromFloat(@trunc(s));
                },
                else => unreachable,
            }
        },
        .i64 => {
            const src = std.mem.bytesAsSlice(i64, try input.readableBytes());
            switch (to) {
                .f32 => {
                    const dst = std.mem.bytesAsSlice(f32, try out.writableBytes());
                    for (dst, src) |*d, s| d.* = @floatFromInt(s);
                },
                .f64 => {
                    const dst = std.mem.bytesAsSlice(f64, try out.writableBytes());
                    for (dst, src) |*d, s| d.* = @floatFromInt(s);
                },
                else => unreachable,
            }
        },
    }
}
