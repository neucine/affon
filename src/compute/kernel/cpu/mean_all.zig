const std = @import("std");
const DType = @import("../../tensor/dtype.zig").DType;
const Storage = @import("../../tensor/storage.zig").Storage;

pub fn run(dtype: DType, input: *const Storage, out: *Storage) !void {
    switch (dtype) {
        .f32 => {
            const src = std.mem.bytesAsSlice(f32, try input.readableBytes());
            const dst = std.mem.bytesAsSlice(f32, try out.writableBytes());
            var acc: f32 = 0;
            for (src) |v| acc += v;
            dst[0] = acc / @as(f32, @floatFromInt(src.len));
        },
        .f64 => {
            const src = std.mem.bytesAsSlice(f64, try input.readableBytes());
            const dst = std.mem.bytesAsSlice(f64, try out.writableBytes());
            var acc: f64 = 0;
            for (src) |v| acc += v;
            dst[0] = acc / @as(f64, @floatFromInt(src.len));
        },
        .i64 => return error.ExecutionNotImplemented,
    }
}
