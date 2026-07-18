const std = @import("std");
const DType = @import("../../tensor/dtype.zig").DType;
const Storage = @import("../../tensor/storage.zig").Storage;
const common = @import("common.zig");

extern fn affon_metal_matmul_f32(a_handle: *anyopaque, b_handle: *anyopaque, out_handle: *anyopaque, m: usize, n: usize, k: usize) c_int;
extern fn affon_metal_matmul_i64(a_handle: *anyopaque, b_handle: *anyopaque, out_handle: *anyopaque, m: usize, n: usize, k: usize) c_int;
extern fn affon_metal_matmul_offset_f32(
    a_handle: *anyopaque,
    b_handle: *anyopaque,
    out_handle: *anyopaque,
    a_offset_bytes: usize,
    b_offset_bytes: usize,
    out_offset_bytes: usize,
    m: usize,
    n: usize,
    k: usize,
) c_int;
extern fn affon_metal_matmul_offset_many_f32(
    a_handle: *anyopaque,
    b_handle: *anyopaque,
    out_handle: *anyopaque,
    a_offsets_bytes: [*]const usize,
    b_offsets_bytes: [*]const usize,
    out_offsets_bytes: [*]const usize,
    batch_count: usize,
    m: usize,
    n: usize,
    k: usize,
) c_int;
extern fn affon_metal_matmul_offset_i64(
    a_handle: *anyopaque,
    b_handle: *anyopaque,
    out_handle: *anyopaque,
    a_offset_bytes: usize,
    b_offset_bytes: usize,
    out_offset_bytes: usize,
    m: usize,
    n: usize,
    k: usize,
) c_int;
extern fn affon_metal_matmul_offset_many_i64(
    a_handle: *anyopaque,
    b_handle: *anyopaque,
    out_handle: *anyopaque,
    a_offsets_bytes: [*]const usize,
    b_offsets_bytes: [*]const usize,
    out_offsets_bytes: [*]const usize,
    batch_count: usize,
    m: usize,
    n: usize,
    k: usize,
) c_int;
extern fn affon_metal_matmul_strided_f32(
    a_handle: *anyopaque,
    b_handle: *anyopaque,
    out_handle: *anyopaque,
    a_offset_bytes: usize,
    b_offset_bytes: usize,
    out_offset_bytes: usize,
    a_row_stride: isize,
    a_col_stride: isize,
    b_row_stride: isize,
    b_col_stride: isize,
    m: usize,
    n: usize,
    k: usize,
) c_int;
extern fn affon_metal_matmul_strided_many_f32(
    a_handle: *anyopaque,
    b_handle: *anyopaque,
    out_handle: *anyopaque,
    a_offsets_bytes: [*]const usize,
    b_offsets_bytes: [*]const usize,
    out_offsets_bytes: [*]const usize,
    batch_count: usize,
    a_row_stride: isize,
    a_col_stride: isize,
    b_row_stride: isize,
    b_col_stride: isize,
    m: usize,
    n: usize,
    k: usize,
) c_int;
extern fn affon_metal_matmul_strided_i64(
    a_handle: *anyopaque,
    b_handle: *anyopaque,
    out_handle: *anyopaque,
    a_offset_bytes: usize,
    b_offset_bytes: usize,
    out_offset_bytes: usize,
    a_row_stride: isize,
    a_col_stride: isize,
    b_row_stride: isize,
    b_col_stride: isize,
    m: usize,
    n: usize,
    k: usize,
) c_int;
extern fn affon_metal_matmul_strided_many_i64(
    a_handle: *anyopaque,
    b_handle: *anyopaque,
    out_handle: *anyopaque,
    a_offsets_bytes: [*]const usize,
    b_offsets_bytes: [*]const usize,
    out_offsets_bytes: [*]const usize,
    batch_count: usize,
    a_row_stride: isize,
    a_col_stride: isize,
    b_row_stride: isize,
    b_col_stride: isize,
    m: usize,
    n: usize,
    k: usize,
) c_int;

pub fn run(dtype: DType, a: *const Storage, b: *const Storage, out: *Storage, a_shape: []const usize, b_shape: []const usize) !void {
    if (!common.isAvailable()) return error.MetalUnavailable;
    if (dtype != .f32 and dtype != .i64) return error.ExecutionNotImplemented;
    const elem_size: usize = switch (dtype) {
        .f32 => @sizeOf(f32),
        .i64 => @sizeOf(i64),
        else => unreachable,
    };
    const a_handle = try a.metalHandle();
    const b_handle = try b.metalHandle();
    const out_handle = try out.metalHandle();

    if (a_shape.len == 2 and b_shape.len == 2) {
        if (a_shape[1] != b_shape[0]) return error.ShapeMismatch;
        const rc = switch (dtype) {
            .f32 => affon_metal_matmul_f32(a_handle, b_handle, out_handle, a_shape[0], b_shape[1], a_shape[1]),
            .i64 => affon_metal_matmul_i64(a_handle, b_handle, out_handle, a_shape[0], b_shape[1], a_shape[1]),
            else => unreachable,
        };
        if (rc != 0) return error.MetalKernelLaunchFailed;
        return;
    }

    if (a_shape.len >= 2 and b_shape.len >= 2) {
        const a_rank = a_shape.len;
        const b_rank = b_shape.len;
        const m = a_shape[a_rank - 2];
        const k = a_shape[a_rank - 1];
        const b_k = b_shape[b_rank - 2];
        const n = b_shape[b_rank - 1];
        if (k != b_k) return error.ShapeMismatch;

        const a_batch = a_shape[0 .. a_rank - 2];
        const b_batch = b_shape[0 .. b_rank - 2];
        const batch_rank = @max(a_batch.len, b_batch.len);

        var out_batch_shape: [8]usize = [_]usize{1} ** 8;
        var a_batch_shape: [8]usize = [_]usize{1} ** 8;
        var b_batch_shape: [8]usize = [_]usize{1} ** 8;
        if (batch_rank > 8) return error.ExecutionNotImplemented;

        var i: usize = 0;
        while (i < batch_rank) : (i += 1) {
            const ai = if (i + a_batch.len >= batch_rank) a_batch[i + a_batch.len - batch_rank] else 1;
            const bi = if (i + b_batch.len >= batch_rank) b_batch[i + b_batch.len - batch_rank] else 1;
            if (ai != bi and ai != 1 and bi != 1) return error.ExecutionNotImplemented;
            out_batch_shape[i] = @max(ai, bi);
            a_batch_shape[i] = ai;
            b_batch_shape[i] = bi;
        }

        var out_batch_strides: [8]usize = [_]usize{1} ** 8;
        var a_batch_strides: [8]usize = [_]usize{1} ** 8;
        var b_batch_strides: [8]usize = [_]usize{1} ** 8;
        if (batch_rank > 0) {
            out_batch_strides[batch_rank - 1] = 1;
            a_batch_strides[batch_rank - 1] = 1;
            b_batch_strides[batch_rank - 1] = 1;
            var d = batch_rank - 1;
            while (d > 0) {
                d -= 1;
                out_batch_strides[d] = out_batch_strides[d + 1] * out_batch_shape[d + 1];
                a_batch_strides[d] = a_batch_strides[d + 1] * a_batch_shape[d + 1];
                b_batch_strides[d] = b_batch_strides[d + 1] * b_batch_shape[d + 1];
            }
        }

        var batch: usize = 1;
        for (out_batch_shape[0..batch_rank]) |d| batch *= d;
        const a_batch_bytes = m * k * elem_size;
        const b_batch_bytes = k * n * elem_size;
        const out_batch_bytes = m * n * elem_size;

        const a_offsets = try std.heap.page_allocator.alloc(usize, batch);
        defer std.heap.page_allocator.free(a_offsets);
        const b_offsets = try std.heap.page_allocator.alloc(usize, batch);
        defer std.heap.page_allocator.free(b_offsets);
        const out_offsets = try std.heap.page_allocator.alloc(usize, batch);
        defer std.heap.page_allocator.free(out_offsets);

        var coord: [8]usize = [_]usize{0} ** 8;
        for (0..batch) |bi| {
            var rem = bi;
            for (0..batch_rank) |r| {
                const d = batch_rank - 1 - r;
                coord[d] = rem % out_batch_shape[d];
                rem /= out_batch_shape[d];
            }

            var a_batch_idx: usize = 0;
            var b_batch_idx: usize = 0;
            for (0..batch_rank) |d| {
                const ac = if (a_batch_shape[d] == 1) 0 else coord[d];
                const bc = if (b_batch_shape[d] == 1) 0 else coord[d];
                a_batch_idx += ac * a_batch_strides[d];
                b_batch_idx += bc * b_batch_strides[d];
            }

            a_offsets[bi] = a_batch_idx * a_batch_bytes;
            b_offsets[bi] = b_batch_idx * b_batch_bytes;
            out_offsets[bi] = bi * out_batch_bytes;
        }
        const rc = switch (dtype) {
            .f32 => affon_metal_matmul_offset_many_f32(
                a_handle,
                b_handle,
                out_handle,
                a_offsets.ptr,
                b_offsets.ptr,
                out_offsets.ptr,
                batch,
                m,
                n,
                k,
            ),
            .i64 => affon_metal_matmul_offset_many_i64(
                a_handle,
                b_handle,
                out_handle,
                a_offsets.ptr,
                b_offsets.ptr,
                out_offsets.ptr,
                batch,
                m,
                n,
                k,
            ),
            else => unreachable,
        };
        if (rc != 0) return error.MetalKernelLaunchFailed;
        return;
    }

    return error.ExecutionNotImplemented;
}

pub fn runStrided(
    dtype: DType,
    a: *const Storage,
    b: *const Storage,
    out: *Storage,
    a_offset_bytes: usize,
    b_offset_bytes: usize,
    out_offset_bytes: usize,
    a_row_stride: isize,
    a_col_stride: isize,
    b_row_stride: isize,
    b_col_stride: isize,
    m: usize,
    n: usize,
    k: usize,
) !void {
    if (!common.isAvailable()) return error.MetalUnavailable;
    if (dtype != .f32 and dtype != .i64) return error.ExecutionNotImplemented;
    const rc = switch (dtype) {
        .f32 => affon_metal_matmul_strided_f32(
            try a.metalHandle(),
            try b.metalHandle(),
            try out.metalHandle(),
            a_offset_bytes,
            b_offset_bytes,
            out_offset_bytes,
            a_row_stride,
            a_col_stride,
            b_row_stride,
            b_col_stride,
            m,
            n,
            k,
        ),
        .i64 => affon_metal_matmul_strided_i64(
            try a.metalHandle(),
            try b.metalHandle(),
            try out.metalHandle(),
            a_offset_bytes,
            b_offset_bytes,
            out_offset_bytes,
            a_row_stride,
            a_col_stride,
            b_row_stride,
            b_col_stride,
            m,
            n,
            k,
        ),
        else => unreachable,
    };
    if (rc != 0) return error.MetalKernelLaunchFailed;
}

pub fn runStridedMany(
    dtype: DType,
    a: *const Storage,
    b: *const Storage,
    out: *Storage,
    a_offsets_bytes: []const usize,
    b_offsets_bytes: []const usize,
    out_offsets_bytes: []const usize,
    a_row_stride: isize,
    a_col_stride: isize,
    b_row_stride: isize,
    b_col_stride: isize,
    m: usize,
    n: usize,
    k: usize,
) !void {
    if (!common.isAvailable()) return error.MetalUnavailable;
    if (dtype != .f32 and dtype != .i64) return error.ExecutionNotImplemented;
    if (a_offsets_bytes.len != b_offsets_bytes.len or a_offsets_bytes.len != out_offsets_bytes.len) return error.ShapeMismatch;
    if (a_offsets_bytes.len == 0) return;
    const rc = switch (dtype) {
        .f32 => affon_metal_matmul_strided_many_f32(
            try a.metalHandle(),
            try b.metalHandle(),
            try out.metalHandle(),
            a_offsets_bytes.ptr,
            b_offsets_bytes.ptr,
            out_offsets_bytes.ptr,
            a_offsets_bytes.len,
            a_row_stride,
            a_col_stride,
            b_row_stride,
            b_col_stride,
            m,
            n,
            k,
        ),
        .i64 => affon_metal_matmul_strided_many_i64(
            try a.metalHandle(),
            try b.metalHandle(),
            try out.metalHandle(),
            a_offsets_bytes.ptr,
            b_offsets_bytes.ptr,
            out_offsets_bytes.ptr,
            a_offsets_bytes.len,
            a_row_stride,
            a_col_stride,
            b_row_stride,
            b_col_stride,
            m,
            n,
            k,
        ),
        else => unreachable,
    };
    if (rc != 0) return error.MetalKernelLaunchFailed;
}
