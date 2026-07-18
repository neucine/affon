const std = @import("std");
const DType = @import("../../types/tensor/dtype.zig").DType;
const Storage = @import("../../types/tensor/storage.zig").Storage;

pub fn run(dtype: DType, a: *const Storage, b: *const Storage, bias: *const Storage, out: *Storage, m: usize, n: usize, k: usize) !void {
    switch (dtype) {
        .f32 => try runTyped(f32, a, b, bias, out, m, n, k),
        .f64 => try runTyped(f64, a, b, bias, out, m, n, k),
        .i64 => try runTyped(i64, a, b, bias, out, m, n, k),
    }
}

pub fn runOffset(
    dtype: DType,
    a: *const Storage,
    b: *const Storage,
    bias: *const Storage,
    out: *Storage,
    a_offset_bytes: usize,
    b_offset_bytes: usize,
    bias_offset_bytes: usize,
    out_offset_bytes: usize,
    m: usize,
    n: usize,
    k: usize,
) !void {
    switch (dtype) {
        .f32 => try runTypedOffset(f32, a, b, bias, out, a_offset_bytes, b_offset_bytes, bias_offset_bytes, out_offset_bytes, m, n, k),
        .f64 => try runTypedOffset(f64, a, b, bias, out, a_offset_bytes, b_offset_bytes, bias_offset_bytes, out_offset_bytes, m, n, k),
        .i64 => try runTypedOffset(i64, a, b, bias, out, a_offset_bytes, b_offset_bytes, bias_offset_bytes, out_offset_bytes, m, n, k),
    }
}

fn runTyped(comptime T: type, a: *const Storage, b: *const Storage, bias: *const Storage, out: *Storage, m: usize, n: usize, k: usize) !void {
    const av = std.mem.bytesAsSlice(T, try a.readableBytes());
    const bv = std.mem.bytesAsSlice(T, try b.readableBytes());
    const biasv = std.mem.bytesAsSlice(T, try bias.readableBytes());
    const dst = std.mem.bytesAsSlice(T, try out.writableBytes());
    for (0..m) |i| {
        for (0..n) |j| {
            var acc: T = 0;
            for (0..k) |kk| acc += av[i * k + kk] * bv[kk * n + j];
            dst[i * n + j] = acc + biasv[i * n + j];
        }
    }
}

fn runTypedOffset(
    comptime T: type,
    a: *const Storage,
    b: *const Storage,
    bias: *const Storage,
    out: *Storage,
    a_offset_bytes: usize,
    b_offset_bytes: usize,
    bias_offset_bytes: usize,
    out_offset_bytes: usize,
    m: usize,
    n: usize,
    k: usize,
) !void {
    const av = std.mem.bytesAsSlice(T, try a.readableBytes());
    const bv = std.mem.bytesAsSlice(T, try b.readableBytes());
    const biasv = std.mem.bytesAsSlice(T, try bias.readableBytes());
    const dst = std.mem.bytesAsSlice(T, try out.writableBytes());
    const a_base = a_offset_bytes / @sizeOf(T);
    const b_base = b_offset_bytes / @sizeOf(T);
    const bias_base = bias_offset_bytes / @sizeOf(T);
    const out_base = out_offset_bytes / @sizeOf(T);
    for (0..m) |i| {
        for (0..n) |j| {
            var acc: T = 0;
            for (0..k) |kk| acc += av[a_base + i * k + kk] * bv[b_base + kk * n + j];
            dst[out_base + i * n + j] = acc + biasv[bias_base + i * n + j];
        }
    }
}
