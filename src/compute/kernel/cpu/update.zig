const std = @import("std");
const DType = @import("../../tensor/dtype.zig").DType;
const Storage = @import("../../tensor/storage.zig").Storage;

pub fn muladdInplace(dtype: DType, target: *Storage, addend: *const Storage, scale: f64) !void {
    switch (dtype) {
        .f32 => {
            const t = std.mem.bytesAsSlice(f32, try target.writableBytes());
            const a = std.mem.bytesAsSlice(f32, try addend.readableBytes());
            const s: f32 = @floatCast(scale);
            for (t, 0..) |v, i| t[i] = s * v + a[i];
        },
        .f64 => {
            const t = std.mem.bytesAsSlice(f64, try target.writableBytes());
            const a = std.mem.bytesAsSlice(f64, try addend.readableBytes());
            for (t, 0..) |v, i| t[i] = scale * v + a[i];
        },
        .i64 => return error.UnsupportedDType,
    }
}

pub fn axpyInplace(dtype: DType, target: *Storage, addend: *const Storage, scale: f64) !void {
    switch (dtype) {
        .f32 => {
            const t = std.mem.bytesAsSlice(f32, try target.writableBytes());
            const a = std.mem.bytesAsSlice(f32, try addend.readableBytes());
            const s: f32 = @floatCast(scale);
            for (t, 0..) |v, i| t[i] = v + s * a[i];
        },
        .f64 => {
            const t = std.mem.bytesAsSlice(f64, try target.writableBytes());
            const a = std.mem.bytesAsSlice(f64, try addend.readableBytes());
            for (t, 0..) |v, i| t[i] = v + scale * a[i];
        },
        .i64 => return error.UnsupportedDType,
    }
}

pub fn subInplace(dtype: DType, target: *Storage, delta: *const Storage) !void {
    switch (dtype) {
        .f32 => {
            const t = std.mem.bytesAsSlice(f32, try target.writableBytes());
            const d = std.mem.bytesAsSlice(f32, try delta.readableBytes());
            for (t, 0..) |v, i| t[i] = v - d[i];
        },
        .f64 => {
            const t = std.mem.bytesAsSlice(f64, try target.writableBytes());
            const d = std.mem.bytesAsSlice(f64, try delta.readableBytes());
            for (t, 0..) |v, i| t[i] = v - d[i];
        },
        .i64 => {
            const t = std.mem.bytesAsSlice(i64, try target.writableBytes());
            const d = std.mem.bytesAsSlice(i64, try delta.readableBytes());
            for (t, 0..) |v, i| t[i] = v - d[i];
        },
    }
}

pub fn scaleInplace(dtype: DType, target: *Storage, scale: f64) !void {
    switch (dtype) {
        .f32 => {
            const t = std.mem.bytesAsSlice(f32, try target.writableBytes());
            const s: f32 = @floatCast(scale);
            for (t) |*x| x.* *= s;
        },
        .f64 => {
            const t = std.mem.bytesAsSlice(f64, try target.writableBytes());
            for (t) |*x| x.* *= scale;
        },
        .i64 => return error.UnsupportedDType,
    }
}
