const std = @import("std");
const DType = @import("../../shared/types/tensor/dtype.zig").DType;
const Storage = @import("../../shared/types/tensor/storage.zig").Storage;

pub fn run(dtype: DType, input: *const Storage, out: *Storage) !void {
    switch (dtype) {
        .f32 => {
            const src = std.mem.bytesAsSlice(f32, try input.readableBytes());
            const dst = std.mem.bytesAsSlice(f32, try out.writableBytes());
            var minv = src[0];
            for (src[1..]) |v| {
                if (v < minv) minv = v;
            }
            dst[0] = minv;
        },
        .f64 => {
            const src = std.mem.bytesAsSlice(f64, try input.readableBytes());
            const dst = std.mem.bytesAsSlice(f64, try out.writableBytes());
            var minv = src[0];
            for (src[1..]) |v| {
                if (v < minv) minv = v;
            }
            dst[0] = minv;
        },
        .i64 => {
            const src = std.mem.bytesAsSlice(i64, try input.readableBytes());
            const dst = std.mem.bytesAsSlice(i64, try out.writableBytes());
            var minv = src[0];
            for (src[1..]) |v| {
                if (v < minv) minv = v;
            }
            dst[0] = minv;
        },
    }
}
