const DType = @import("../../shared/types/tensor/dtype.zig").DType;
const Storage = @import("../../shared/types/tensor/storage.zig").Storage;
const common = @import("common.zig");

extern fn affon_metal_where_f32(cond_handle: *anyopaque, a_handle: *anyopaque, b_handle: *anyopaque, out_handle: *anyopaque, len: usize) c_int;
extern fn affon_metal_where_i64_f32(cond_handle: *anyopaque, a_handle: *anyopaque, b_handle: *anyopaque, out_handle: *anyopaque, len: usize) c_int;
extern fn affon_metal_where_f32_i64(cond_handle: *anyopaque, a_handle: *anyopaque, b_handle: *anyopaque, out_handle: *anyopaque, len: usize) c_int;
extern fn affon_metal_where_i64_i64(cond_handle: *anyopaque, a_handle: *anyopaque, b_handle: *anyopaque, out_handle: *anyopaque, len: usize) c_int;
extern fn affon_metal_where_broadcast_f32(
    cond_handle: *anyopaque,
    a_handle: *anyopaque,
    b_handle: *anyopaque,
    out_handle: *anyopaque,
    ndim: usize,
    shape: [*]const u32,
    cond_strides: [*]const u32,
    a_strides: [*]const u32,
    b_strides: [*]const u32,
    cond_offset: usize,
    a_offset: usize,
    b_offset: usize,
    len: usize,
) c_int;
extern fn affon_metal_where_broadcast_i64_f32(
    cond_handle: *anyopaque,
    a_handle: *anyopaque,
    b_handle: *anyopaque,
    out_handle: *anyopaque,
    ndim: usize,
    shape: [*]const u32,
    cond_strides: [*]const u32,
    a_strides: [*]const u32,
    b_strides: [*]const u32,
    cond_offset: usize,
    a_offset: usize,
    b_offset: usize,
    len: usize,
) c_int;

pub fn run(cond_dtype: DType, value_dtype: DType, cond: *const Storage, on_true: *const Storage, on_false: *const Storage, out: *Storage) !void {
    if (!common.isAvailable()) return error.MetalUnavailable;
    const a_handle = try on_true.metalHandle();
    const b_handle = try on_false.metalHandle();
    const out_handle = try out.metalHandle();
    const len = switch (value_dtype) {
        .f32 => out.bytes / @sizeOf(f32),
        .i64 => out.bytes / @sizeOf(i64),
        else => return error.ExecutionNotImplemented,
    };

    const cond_handle = try cond.metalHandle();

    const rc = switch (value_dtype) {
        .f32 => switch (cond_dtype) {
            .f32 => affon_metal_where_f32(cond_handle, a_handle, b_handle, out_handle, len),
            .i64 => affon_metal_where_i64_f32(cond_handle, a_handle, b_handle, out_handle, len),
            else => return error.ExecutionNotImplemented,
        },
        .i64 => switch (cond_dtype) {
            .f32 => affon_metal_where_f32_i64(cond_handle, a_handle, b_handle, out_handle, len),
            .i64 => affon_metal_where_i64_i64(cond_handle, a_handle, b_handle, out_handle, len),
            else => return error.ExecutionNotImplemented,
        },
        else => return error.ExecutionNotImplemented,
    };
    if (rc != 0) return error.MetalKernelLaunchFailed;
}

pub fn runBroadcastF32(
    cond_dtype: DType,
    cond: *const Storage,
    on_true: *const Storage,
    on_false: *const Storage,
    out: *Storage,
    shape: []const usize,
    cond_strides: []const isize,
    a_strides: []const isize,
    b_strides: []const isize,
    cond_offset: usize,
    a_offset: usize,
    b_offset: usize,
) !void {
    if (!common.isAvailable()) return error.MetalUnavailable;
    if (shape.len == 0 or shape.len > 8) return error.ExecutionNotImplemented;
    if (cond_strides.len != shape.len or a_strides.len != shape.len or b_strides.len != shape.len) return error.ExecutionNotImplemented;
    var shape_u32: [8]u32 = [_]u32{1} ** 8;
    var cond_u32: [8]u32 = [_]u32{0} ** 8;
    var a_u32: [8]u32 = [_]u32{0} ** 8;
    var b_u32: [8]u32 = [_]u32{0} ** 8;
    var len: usize = 1;
    for (shape, 0..) |dim, i| {
        shape_u32[i] = @intCast(dim);
        len *= dim;
        if (cond_strides[i] < 0 or a_strides[i] < 0 or b_strides[i] < 0) return error.ExecutionNotImplemented;
        cond_u32[i] = @intCast(cond_strides[i]);
        a_u32[i] = @intCast(a_strides[i]);
        b_u32[i] = @intCast(b_strides[i]);
    }
    const rc = switch (cond_dtype) {
        .f32 => affon_metal_where_broadcast_f32(
            try cond.metalHandle(),
            try on_true.metalHandle(),
            try on_false.metalHandle(),
            try out.metalHandle(),
            shape.len,
            &shape_u32,
            &cond_u32,
            &a_u32,
            &b_u32,
            cond_offset,
            a_offset,
            b_offset,
            len,
        ),
        .i64 => affon_metal_where_broadcast_i64_f32(
            try cond.metalHandle(),
            try on_true.metalHandle(),
            try on_false.metalHandle(),
            try out.metalHandle(),
            shape.len,
            &shape_u32,
            &cond_u32,
            &a_u32,
            &b_u32,
            cond_offset,
            a_offset,
            b_offset,
            len,
        ),
        else => return error.ExecutionNotImplemented,
    };
    if (rc != 0) return error.MetalKernelLaunchFailed;
}
