const std = @import("std");
const DType = @import("../../shared/types/tensor/dtype.zig").DType;
const Storage = @import("../../shared/types/tensor/storage.zig").Storage;

pub fn run(input_dtype: DType, mask_dtype: DType, input: *const Storage, mask: *const Storage, out: *Storage, fill_value: f64) !void {
    const mask_bytes = try mask.readableBytes();
    switch (input_dtype) {
        .f32 => {
            const src = std.mem.bytesAsSlice(f32, try input.readableBytes());
            const dst = std.mem.bytesAsSlice(f32, try out.writableBytes());
            const fv: f32 = @floatCast(fill_value);
            for (dst, 0..) |*d, i| d.* = if (maskAt(mask_dtype, mask_bytes, i)) fv else src[i];
        },
        .f64 => {
            const src = std.mem.bytesAsSlice(f64, try input.readableBytes());
            const dst = std.mem.bytesAsSlice(f64, try out.writableBytes());
            for (dst, 0..) |*d, i| d.* = if (maskAt(mask_dtype, mask_bytes, i)) fill_value else src[i];
        },
        .i64 => {
            const src = std.mem.bytesAsSlice(i64, try input.readableBytes());
            const dst = std.mem.bytesAsSlice(i64, try out.writableBytes());
            const fv: i64 = @intFromFloat(fill_value);
            for (dst, 0..) |*d, i| d.* = if (maskAt(mask_dtype, mask_bytes, i)) fv else src[i];
        },
    }
}

fn maskAt(dtype: DType, mask_bytes: []align(8) const u8, i: usize) bool {
    return switch (dtype) {
        .f32 => std.mem.bytesAsSlice(f32, mask_bytes)[i] != 0,
        .f64 => std.mem.bytesAsSlice(f64, mask_bytes)[i] != 0,
        .i64 => std.mem.bytesAsSlice(i64, mask_bytes)[i] != 0,
    };
}
