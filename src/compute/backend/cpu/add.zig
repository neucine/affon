const std = @import("std");
const DType = @import("../../shared/types/tensor/dtype.zig").DType;
const Storage = @import("../../shared/types/tensor/storage.zig").Storage;

pub fn run(dtype: DType, a: *const Storage, b: *const Storage, out: *Storage) !void {
    switch (dtype) {
        .f32 => {
            const a_typed = std.mem.bytesAsSlice(f32, try a.readableBytes());
            const b_typed = std.mem.bytesAsSlice(f32, try b.readableBytes());
            const out_typed = std.mem.bytesAsSlice(f32, try out.writableBytes());
            if (a_typed.len != out_typed.len or b_typed.len != out_typed.len) return error.InvalidStorageLength;
            for (out_typed, a_typed, b_typed) |*dst, lhs, rhs| dst.* = lhs + rhs;
        },
        .f64 => {
            const a_typed = std.mem.bytesAsSlice(f64, try a.readableBytes());
            const b_typed = std.mem.bytesAsSlice(f64, try b.readableBytes());
            const out_typed = std.mem.bytesAsSlice(f64, try out.writableBytes());
            if (a_typed.len != out_typed.len or b_typed.len != out_typed.len) return error.InvalidStorageLength;
            for (out_typed, a_typed, b_typed) |*dst, lhs, rhs| dst.* = lhs + rhs;
        },
        .i64 => {
            const a_typed = std.mem.bytesAsSlice(i64, try a.readableBytes());
            const b_typed = std.mem.bytesAsSlice(i64, try b.readableBytes());
            const out_typed = std.mem.bytesAsSlice(i64, try out.writableBytes());
            if (a_typed.len != out_typed.len or b_typed.len != out_typed.len) return error.InvalidStorageLength;
            for (out_typed, a_typed, b_typed) |*dst, lhs, rhs| dst.* = lhs + rhs;
        },
    }
}
