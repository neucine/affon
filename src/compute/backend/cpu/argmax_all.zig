const std = @import("std");
const DType = @import("../../shared/types/tensor/dtype.zig").DType;
const Storage = @import("../../shared/types/tensor/storage.zig").Storage;

pub fn run(input_dtype: DType, input: *const Storage, out: *Storage) !void {
    const dst = std.mem.bytesAsSlice(i64, try out.writableBytes());
    switch (input_dtype) {
        .f32 => {
            const src = std.mem.bytesAsSlice(f32, try input.readableBytes());
            var idx: usize = 0;
            var maxv = src[0];
            for (src[1..], 1..) |v, i| {
                if (v > maxv) {
                    maxv = v;
                    idx = i;
                }
            }
            dst[0] = @intCast(idx);
        },
        .f64 => {
            const src = std.mem.bytesAsSlice(f64, try input.readableBytes());
            var idx: usize = 0;
            var maxv = src[0];
            for (src[1..], 1..) |v, i| {
                if (v > maxv) {
                    maxv = v;
                    idx = i;
                }
            }
            dst[0] = @intCast(idx);
        },
        .i64 => {
            const src = std.mem.bytesAsSlice(i64, try input.readableBytes());
            var idx: usize = 0;
            var maxv = src[0];
            for (src[1..], 1..) |v, i| {
                if (v > maxv) {
                    maxv = v;
                    idx = i;
                }
            }
            dst[0] = @intCast(idx);
        },
    }
}
