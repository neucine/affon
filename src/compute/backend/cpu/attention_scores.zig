const std = @import("std");
const DType = @import("../../types/tensor/dtype.zig").DType;
const Storage = @import("../../types/tensor/storage.zig").Storage;

pub fn run(
    allocator: std.mem.Allocator,
    dtype: DType,
    q: *const Storage,
    k_t: *const Storage,
    scale: *const Storage,
    mask: *const Storage,
    out: *Storage,
    mask_fill_value: f64,
    m: usize,
    n: usize,
    k: usize,
) !void {
    return runOffset(allocator, dtype, q, k_t, scale, mask, out, 0, 0, 0, 0, 0, m, n, k, mask_fill_value);
}

pub fn runOffset(
    allocator: std.mem.Allocator,
    dtype: DType,
    q: *const Storage,
    k_t: *const Storage,
    scale: *const Storage,
    mask: *const Storage,
    out: *Storage,
    q_offset_bytes: usize,
    k_t_offset_bytes: usize,
    scale_offset_bytes: usize,
    mask_offset_bytes: usize,
    out_offset_bytes: usize,
    m: usize,
    n: usize,
    k: usize,
    mask_fill_value: f64,
) !void {
    switch (dtype) {
        .f32 => try runTyped(allocator, f32, q, k_t, scale, mask, out, q_offset_bytes, k_t_offset_bytes, scale_offset_bytes, mask_offset_bytes, out_offset_bytes, m, n, k, @floatCast(mask_fill_value)),
        .f64 => try runTyped(allocator, f64, q, k_t, scale, mask, out, q_offset_bytes, k_t_offset_bytes, scale_offset_bytes, mask_offset_bytes, out_offset_bytes, m, n, k, mask_fill_value),
        .i64 => return error.ExecutionNotImplemented,
    }
}

fn runTyped(
    allocator: std.mem.Allocator,
    comptime T: type,
    q: *const Storage,
    k_t: *const Storage,
    scale: *const Storage,
    mask: *const Storage,
    out: *Storage,
    q_offset_bytes: usize,
    k_t_offset_bytes: usize,
    scale_offset_bytes: usize,
    mask_offset_bytes: usize,
    out_offset_bytes: usize,
    m: usize,
    n: usize,
    k: usize,
    mask_fill_value: T,
) !void {
    const qv = std.mem.bytesAsSlice(T, try q.readableBytes());
    const kv = std.mem.bytesAsSlice(T, try k_t.readableBytes());
    const sv = std.mem.bytesAsSlice(T, try scale.readableBytes());
    const mv = std.mem.bytesAsSlice(i64, try mask.readableBytes());
    const ov = std.mem.bytesAsSlice(T, try out.writableBytes());

    const qb = q_offset_bytes / @sizeOf(T);
    const kb = k_t_offset_bytes / @sizeOf(T);
    const sb = scale_offset_bytes / @sizeOf(T);
    const mb = mask_offset_bytes / @sizeOf(i64);
    const ob = out_offset_bytes / @sizeOf(T);

    var row_buf = try allocator.alloc(T, n);
    defer allocator.free(row_buf);
    for (0..m) |r| {
        var max_v: T = -std.math.inf(T);
        for (0..n) |c| {
            var dot: T = 0;
            for (0..k) |kk| dot += qv[qb + r * k + kk] * kv[kb + kk * n + c];
            var v = dot * sv[sb + r * n + c];
            if (mv[mb + r * n + c] != 0) v = mask_fill_value;
            row_buf[c] = v;
            if (v > max_v) max_v = v;
        }
        var sum_exp: T = 0;
        for (0..n) |c| {
            const e = @exp(row_buf[c] - max_v);
            row_buf[c] = e;
            sum_exp += e;
        }
        for (0..n) |c| ov[ob + r * n + c] = row_buf[c] / sum_exp;
    }
}
